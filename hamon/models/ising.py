# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import contextlib
import logging

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Bool, Key, Shaped

from hamon.device import (
    DeviceLike,
    free_node_count,
    resolve_entry_device,
    tree_device_put,
)
from hamon.block_sampling import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    SuperBlock,
    sample_with_observation,
)
from hamon.factor import FactorSamplingProgram
from hamon.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from hamon.models.ebm import AbstractFactorizedEBM, EBMFactor
from hamon.observers import MomentAccumulatorObserver
from hamon.pgm import AbstractNode, SpinNode

logger = logging.getLogger(__name__)

Edge = tuple[AbstractNode, AbstractNode]


# ``IsingEBM.factors`` runs once per chain per ``nrpt`` call (via
# ``with_beta``/``with_ebm``), and its node-group Blocks depend only on the
# ``nodes``/``edges`` lists — which ``with_beta`` passes through by reference —
# never on β or the weights. Rebuilding them cost two O(|edges|) list
# comprehensions plus three O(|nodes|) Block type scans per rebuild. Keyed on
# list identity; entries pin their key lists (no id reuse while live) and the
# cache is a bounded FIFO.
_FACTOR_BLOCK_CACHE: dict = {}
_FACTOR_BLOCK_CACHE_MAX = 32


def _ising_factor_blocks(
    nodes: list[AbstractNode], edges: list[Edge]
) -> tuple[Block, Block, Block]:
    """Build (cached) the bias Block and the edge head/tail Blocks."""
    key = (id(nodes), id(edges))
    hit = _FACTOR_BLOCK_CACHE.get(key)
    if hit is not None and hit[0] is nodes and hit[1] is edges:
        return hit[2], hit[3], hit[4]

    bias_block = Block(nodes)
    head_block = Block([x[0] for x in edges])
    tail_block = Block([x[1] for x in edges])

    if len(_FACTOR_BLOCK_CACHE) >= _FACTOR_BLOCK_CACHE_MAX:
        _FACTOR_BLOCK_CACHE.pop(next(iter(_FACTOR_BLOCK_CACHE)))
    _FACTOR_BLOCK_CACHE[key] = (nodes, edges, bias_block, head_block, tail_block)
    return bias_block, head_block, tail_block


class IsingEBM(AbstractFactorizedEBM):
    r"""An EBM with the energy function,

    $$\mathcal{E}(s) = -\beta \left( \sum_{i \in S_1} b_i s_i + \sum_{(i, j) \in S_2} J_{ij} s_i s_j \right)$$

    where $S_1$ and $S_2$ are the sets of biases and weights that make up the model, respectively.
    $b_i$ represents the bias associated with the spin $s_i$ and $J_{ij}$ is a weight that couples
    $s_i$ and $s_j$. $\beta$ is the usual temperature parameter.

    **Attributes:**

    - `nodes`: the nodes that have an associated bias (i.e $S_1$)
    - `biases`: the bias associated with each node in `nodes`.
    - `edges`: the edges that have an associated weight (i.e $S_2$)
    - `weights`: the weight associated with each pair of nodes in `edges`.
    - `beta`: the scalar temperature parameter for the model.
    """

    nodes: list[AbstractNode]
    biases: Array
    edges: list[Edge]
    weights: Array
    beta: Array

    # .factors must recompute β*weights on every call — caching breaks AD tracer flow.

    def __init__(
        self,
        nodes: list[AbstractNode],
        edges: list[Edge],
        biases: Array,
        weights: Array,
        beta: Array,
    ):
        sd_map = {nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool_)}
        super().__init__(sd_map)
        self.nodes = nodes
        self.edges = edges
        # β must match the float dtype of the parameters it scales: a strong
        # float64 β (e.g. jnp.array(1.0) with x64 enabled in the host
        # application) would otherwise promote every β·W interaction tensor —
        # and with it the whole device sampling loop — to float64.
        param_dtype = jnp.result_type(biases, weights)
        if jnp.issubdtype(param_dtype, jnp.floating):
            beta = jnp.asarray(beta, dtype=param_dtype)
        self.beta = beta
        self.weights = weights
        self.biases = biases

    def with_beta(self, beta: Array) -> "IsingEBM":
        return IsingEBM(self.nodes, self.edges, self.biases, self.weights, beta)

    @property
    def factors(self) -> list[EBMFactor]:
        bias_block, head_block, tail_block = _ising_factor_blocks(
            self.nodes, self.edges
        )
        return [
            SpinEBMFactor([bias_block], self.beta * self.biases),
            SpinEBMFactor([head_block, tail_block], self.beta * self.weights),
        ]


class IsingSamplingProgram(FactorSamplingProgram):
    """A very thin wrapper on FactorSamplingProgram that specializes it to the case of an Ising Model."""

    def __init__(
        self,
        ebm: IsingEBM,
        free_blocks: list[SuperBlock],
        clamped_blocks: list[Block],
        *,
        _gibbs_spec: BlockGibbsSpec | None = None,
    ):
        samp = SpinGibbsConditional()
        # _gibbs_spec: internal fast path for with_ebm — the spec is pure
        # structure (no β, no weights), so a rebuild over the same blocks would
        # reproduce it node for node while paying an O(|nodes|) location-map
        # construction.
        spec = (
            _gibbs_spec
            if _gibbs_spec is not None
            else BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)
        )
        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])

    def with_ebm(self, ebm: IsingEBM) -> "IsingSamplingProgram":
        return IsingSamplingProgram(
            ebm,
            list(self.gibbs_spec.superblocks),
            self.gibbs_spec.clamped_blocks,
            _gibbs_spec=self.gibbs_spec,
        )


class IsingTrainingSpec(eqx.Module):
    """Contains a complete specification of an Ising EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for Monte Carlo estimation of the gradient of the KL-divergence between the model and a data distribution.
    """

    ebm: IsingEBM
    program_positive: IsingSamplingProgram
    program_negative: IsingSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule

    def __init__(
        self,
        ebm: IsingEBM,
        data_blocks: list[Block],
        conditioning_blocks: list[Block],
        positive_sampling_blocks: list[SuperBlock],
        negative_sampling_blocks: list[SuperBlock],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ):
        self.ebm = ebm
        self.program_positive = IsingSamplingProgram(
            ebm, positive_sampling_blocks, data_blocks + conditioning_blocks
        )
        self.program_negative = IsingSamplingProgram(
            ebm, negative_sampling_blocks, conditioning_blocks
        )
        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


@eqx.filter_jit
def hinton_init(
    key: Key[Array, ""],
    model: IsingEBM,
    blocks: list[Block[AbstractNode]],
    batch_shape: tuple[int, ...],
) -> list[Bool[Array, "batch_size block_size"]]:
    r"""
    Initialize the blocks according to the marginal bias.

    Each binary unit $i$ in a block is sampled independently as

    $$\mathbb{P}(S_i = 1) = \sigma(\beta h_i) = \frac{1}{1 + e^{-\beta h_i}}$$

    where $h_i$ is the bias of unit *i* and $\beta$ is the
    inverse-temperature scaling factor. See Hinton (2012) for a discussion of this initialization heuristic.

    Each block is sampled with its own Bernoulli draw; blocks may have
    different sizes.

    Arguments:
        key: the JAX PRNG key to use
        model: the Ising model to initialize for
        blocks: the blocks that are to be initialized
        batch_shape: the pre-pended batch dimension

    Returns:
        the initialized blocks as a list of bool arrays, one per block
    """
    node_map = {node: i for i, node in enumerate(model.nodes)}

    # Process each block independently to handle ragged block sizes correctly.
    keys = jax.random.split(key, len(blocks))

    result = []
    for block, k in zip(blocks, keys):
        indices = jnp.array([node_map[n] for n in block], dtype=jnp.int32)
        block_biases = model.biases[indices]  # (block_size,)
        probs = jax.nn.sigmoid(model.beta * block_biases)
        sample = jax.random.bernoulli(
            k, p=probs, shape=(*batch_shape, len(block))
        ).astype(jnp.bool_)
        result.append(sample)

    return result


def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
    *,
    device: DeviceLike = "auto",
):
    """
    Estimates the first and second moments of an Ising model Boltzmann distribution via sampling.

    Arguments:
        key: the jax PRNG key
        first_moment_nodes: the nodes that represent the variables we want to estimate the first moments of
        second_moment_edges: the edges that connect the variables we want to estimate the second moments of
        program: the `BlockSamplingProgram` to be used for sampling
        schedule: the schedule to use for sampling
        init_state: the variable values to use to initialize the sampling
        clamped_data: the variable values to assign to the clamped nodes
    Returns:
        the first and second moment data
    """
    moment_spec = []
    if first_moment_nodes:
        moment_spec.append([(node,) for node in first_moment_nodes])
    moment_spec.append(list(second_moment_edges))

    def _spin_transform(state, _):
        return [2 * x.astype(jnp.int8) - 1 for x in state]

    observer = MomentAccumulatorObserver(moment_spec, _spin_transform)
    init_mem = observer.init()

    moments, _ = sample_with_observation(
        key,
        program,
        schedule,
        init_state,
        clamped_data,
        init_mem,
        observer,
        device=device,
    )

    if first_moment_nodes:
        node_sums, edge_sums = moments
    else:
        node_sums = jnp.zeros(0)
        edge_sums = moments[0]

    node_moments = node_sums / schedule.n_samples
    edge_moments = edge_sums / schedule.n_samples

    return node_moments, edge_moments


def estimate_kl_grad(
    key: Key[Array, ""],
    training_spec: IsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    data: list[Array],
    conditioning_values: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
    *,
    device: DeviceLike = "auto",
) -> tuple:
    r"""
    Estimate the KL-gradients of an Ising model with respect to its weights and biases.

    Uses the standard two-term Monte Carlo estimator of the gradient of the KL-divergence between an Ising model and
    a data distribution.

    The gradients are:

    $$\Delta W = -\beta (\langle s_i s_j \rangle_{+} - \langle s_i s_j \rangle_{-})$$

    $$\Delta b = -\beta (\langle s_i \rangle_{+} - \langle s_i \rangle_{-})$$

    Here, $\langle\cdot\rangle_{+}$ denotes an expectation under the
    *positive* phase (data-clamped Boltzmann distribution) and
    $\langle\cdot\rangle_{-}$ under the *negative* phase (model
    distribution).

    Arguments:
        key: the JAX PRNG key
        training_spec: the Ising EBM for which to estimate the gradients
        bias_nodes: the nodes for which to estimate the bias gradients
        weight_edges: the edges for which to estimate the weight gradients
        data: The data values to use for the positive phase of the gradient estimate. Each array has shape [batch nodes]
        conditioning_values: values to assign to the nodes that the model is conditioned on.
         Each array has shape [nodes]
        init_state_positive: initial state for the positive sampling chain. Each array has
         shape [n_chains_pos batch nodes]
        init_state_negative: initial state for the negative sampling chain. Each array has
         shape [n_chains_neg nodes]
    Returns:
        the weight gradients and the bias gradients
    """
    n_chains_pos, batch_size = init_state_positive[0].shape[:2]
    n_chains_neg = init_state_negative[0].shape[0]

    dev = resolve_entry_device(
        device,
        n_chains=n_chains_pos * batch_size + n_chains_neg,
        n_nodes=free_node_count(training_spec.program_negative),
        arrays=(
            data,
            conditioning_values,
            init_state_positive,
            init_state_negative,
            key,
        ),
    )
    if dev is not None:
        (
            key,
            training_spec,
            data,
            conditioning_values,
            init_state_positive,
            init_state_negative,
        ) = tree_device_put(
            (
                key,
                training_spec,
                data,
                conditioning_values,
                init_state_positive,
                init_state_negative,
            ),
            dev,
        )
    device_ctx = (
        jax.default_device(dev) if dev is not None else contextlib.nullcontext()
    )

    with device_ctx:
        key_pos, key_neg = jax.random.split(key, 2)

        cond_batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (data[0].shape[0], *x.shape)),
            conditioning_values,
        )

        keys_pos = jax.random.split(key_pos, (n_chains_pos, batch_size))

        # The vmapped estimate_moments calls see tracers, so their own "auto"
        # routing is a no-op — placement is governed by this context.
        moms_b_pos, moms_w_pos = jax.vmap(
            lambda k_out, i_out: jax.vmap(
                lambda k, i, c: estimate_moments(
                    k,
                    bias_nodes,
                    weight_edges,
                    training_spec.program_positive,
                    training_spec.schedule_positive,
                    i,
                    c,
                )
            )(k_out, i_out, data + cond_batched_pos)
        )(keys_pos, init_state_positive)

        keys_neg = jax.random.split(key_neg, n_chains_neg)

        moms_b_neg, moms_w_neg = jax.vmap(
            lambda k, i: estimate_moments(
                k,
                bias_nodes,
                weight_edges,
                training_spec.program_negative,
                training_spec.schedule_negative,
                i,
                conditioning_values,
            )
        )(keys_neg, init_state_negative)

        float_type = training_spec.ebm.beta.dtype
        grad_b = -training_spec.ebm.beta * (
            jnp.mean(moms_b_pos, axis=(0, 1), dtype=float_type)
            - jnp.mean(moms_b_neg, axis=0, dtype=float_type)
        )
        grad_w = -training_spec.ebm.beta * (
            jnp.mean(moms_w_pos, axis=(0, 1), dtype=float_type)
            - jnp.mean(moms_w_neg, axis=0, dtype=float_type)
        )
        return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)


def ising_sample(
    biases: Shaped[Array, " n"],
    edges: Shaped[Array, "m 2"],
    weights: Shaped[Array, " m"],
    *,
    key: Key[Array, ""],
    beta: float = 1.0,
    n_samples: int = 1000,
    n_warmup: int = 500,
    steps_per_sample: int = 1,
    target_acceptance: float = 0.5,
    max_chains: int = 128,
    device: DeviceLike = "auto",
) -> tuple[Bool[Array, "n_samples n"], dict]:
    r"""Sample from an Ising model Boltzmann distribution via fully autotuned NRPT.

    A thin Ising-specific front end over :func:`hamon.autosample`: it builds and
    colours the graph, then **autotunes the full NRPT configuration** — chain
    count, local-exploration count (``gibbs_steps_per_round``), and schedule —
    before drawing from the cold chain. Unlike earlier versions, the
    exploration count is no longer a fixed argument; it is discovered (and
    device-calibrated) automatically.

    The energy function is

    $$\mathcal{E}(s) = -\beta \left( \sum_i b_i s_i
        + \sum_{(i,j)} J_{ij} s_i s_j \right)$$

    Args:
        biases: per-node bias array of shape ``(n,)``.
        edges: integer index pairs of shape ``(m, 2)``.
        weights: per-edge coupling of shape ``(m,)``.
        key: JAX PRNG key.
        beta: inverse temperature for the target distribution.
        n_samples: number of samples to return.
        n_warmup: warmup steps before collecting samples.
        steps_per_sample: Gibbs sweeps between recorded samples.
        target_acceptance: desired per-pair swap acceptance rate for the
            chain-count search. Default 0.5 — the round-trip-optimal r* = 1/2
            (N* ≈ 2Λ; Syed et al.).
        max_chains: ceiling on the discovered chain count.
        device: where to run — ``"auto"`` (default), ``"cpu"``/``"gpu"``, a
            concrete ``jax.Device``, or ``None`` to leave placement untouched.
            Resolved once and reused across all autotuning stages; the measured
            wall time on this device calibrates the chosen ``gibbs_steps_per_round``.

    Returns:
        A tuple ``(samples, diagnostics)`` where *samples* is a boolean array of
        shape ``(n_samples, n)`` (``True`` = spin up) and *diagnostics* is a dict
        with keys ``n_chains``, ``betas``, ``Lambda``, ``gibbs_steps_per_round``,
        ``mean_spins`` (average number of +1 spins per sample), ``device``,
        ``round_trip_diagnostics``, and ``report`` (the full
        :class:`hamon.AutotuneReport`).

    Warns:
        Logs a warning if all coupling weights are zero (NRPT is
        unnecessary) or if all biases are identical (model has no
        per-variable preference).
    """
    from hamon.graph_utils import rlf_coloring
    from hamon.autotune import autosample

    biases = jnp.asarray(biases)
    weights = jnp.asarray(weights)
    n = biases.shape[0]
    # Host (numpy) array: the graph is built on the host by indexing every edge
    # endpoint (``int(e[0])``). Keeping this on-device would make each of those
    # ~2·n_edges indexing ops a blocking device→host transfer (and an eager
    # slice dispatch); np.asarray pulls it to the host once instead.
    edges_np = np.asarray(edges)

    # --- degenerate model checks ---
    if edges_np.shape[0] == 0:
        logger.warning(
            "No edges provided — variables are independent. NRPT is unnecessary; single-chain Gibbs sampling suffices."
        )
    elif jnp.all(weights == 0):
        logger.warning(
            "All coupling weights are zero — variables are independent. "
            "NRPT is unnecessary; single-chain Gibbs sampling suffices."
        )
    bias_range = float(jnp.max(biases) - jnp.min(biases))
    if bias_range == 0 and biases.shape[0] > 1:
        logger.warning(
            "All biases are identical (spread = 0). The model has no "
            "per-variable preference; sampling results may be uninformative."
        )

    # --- build graph and colour it for block Gibbs ---
    nodes: list[AbstractNode] = [SpinNode() for _ in range(n)]
    # .tolist() converts the whole edge array to Python ints in C, instead of
    # 2·|edges| per-element int(...) casts on numpy scalars.
    node_edges: list[Edge] = [(nodes[u], nodes[v]) for u, v in edges_np.tolist()]

    # Recursive-Largest-First colouring of the variable graph: each colour class
    # is an independent set and becomes one block-Gibbs group. The colour count
    # is the number of sequential sample groups in the NRPT round loop, which
    # sets its XLA compile cost, so minimising colours directly cuts compile —
    # RLF does that more aggressively than greedy heuristics on dense graphs and
    # matches them on sparse/bipartite ones.
    coloring = rlf_coloring(n, edges_np)
    n_colors = (max(coloring) + 1) if n else 1
    color_groups: list[list[AbstractNode]] = [[] for _ in range(n_colors)]
    for idx in range(n):
        color_groups[coloring[idx]].append(nodes[idx])
    free_blocks: list[SuperBlock] = [Block(group) for group in color_groups]

    # --- template EBM & program ---
    ebm = IsingEBM(nodes, node_edges, biases, weights, jnp.array(float(beta)))
    program = IsingSamplingProgram(ebm, free_blocks, [])

    # --- autotune (N + exploration + schedule) and draw ---
    key, k_init, k_auto = jax.random.split(key, 3)

    def _init_factory(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        keys = jax.random.split(k_init, n_chains)
        return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]

    samples, report = autosample(
        k_auto,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
        ebm=ebm,
        program=program,
        init_factory=_init_factory,
        clamp_state=[],
        sample_nodes=nodes,
        beta_range=(0.0, float(beta)),
        target_acceptance=target_acceptance,
        max_chains=max_chains,
        device=device,
    )

    mean_spins = float(jnp.mean(jnp.sum(samples, axis=1).astype(jnp.float32)))
    diagnostics = {
        "n_chains": report.n_chains,
        "betas": report.betas,
        "Lambda": report.Lambda,
        "gibbs_steps_per_round": report.gibbs_steps_per_round,
        "mean_spins": mean_spins,
        "device": report.device,
        "round_trip_diagnostics": report.round_trip_diagnostics,
        "report": report,
    }
    return samples, diagnostics
