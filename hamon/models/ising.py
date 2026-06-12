# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import contextlib
import logging

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Key, Shaped

from hamon.device import (
    DeviceLike,
    free_node_count,
    resolve_device,
    resolve_entry_device,
    tree_device_put,
    work_score,
)
from hamon.block_sampling import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    SuperBlock,
    sample_states,
    sample_with_observation,
)
from hamon.factor import FactorSamplingProgram
from hamon.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from hamon.models.ebm import AbstractFactorizedEBM, EBMFactor
from hamon.observers import MomentAccumulatorObserver
from hamon.pgm import AbstractNode, SpinNode

logger = logging.getLogger(__name__)

Edge = tuple[AbstractNode, AbstractNode]


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
        return [
            SpinEBMFactor([Block(self.nodes)], self.beta * self.biases),
            SpinEBMFactor(
                [Block([x[0] for x in self.edges]), Block([x[1] for x in self.edges])],
                self.beta * self.weights,
            ),
        ]


class IsingSamplingProgram(FactorSamplingProgram):
    """A very thin wrapper on FactorSamplingProgram that specializes it to the case of an Ising Model."""

    def __init__(
        self, ebm: IsingEBM, free_blocks: list[SuperBlock], clamped_blocks: list[Block]
    ):
        samp = SpinGibbsConditional()
        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)
        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])

    def with_ebm(self, ebm: IsingEBM) -> "IsingSamplingProgram":
        return IsingSamplingProgram(
            ebm, list(self.gibbs_spec.superblocks), self.gibbs_spec.clamped_blocks
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
    gibbs_steps_per_round: int = 1,
    target_acceptance: float = 0.6,
    device: DeviceLike = "auto",
) -> tuple[Bool[Array, "n_samples n"], dict]:
    r"""Sample from an Ising model Boltzmann distribution via NRPT.

    Orchestrates the full pipeline: graph coloring, chain-count discovery,
    adaptive schedule tuning, and final sampling from the cold chain.

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
        gibbs_steps_per_round: Gibbs sweeps between NRPT swap attempts.
        target_acceptance: desired per-pair swap acceptance rate for
            chain-count discovery.
        device: where to run — ``"auto"`` (default; small workloads on CPU,
            large on a visible accelerator), ``"cpu"``/``"gpu"``, a concrete
            ``jax.Device``, or ``None`` to leave placement untouched. Under
            ``"auto"`` the device is re-resolved once after chain-count
            discovery, so cheap discovery probes may run on CPU while a large
            production run lands on the accelerator.

    Returns:
        A tuple ``(samples, diagnostics)`` where *samples* is a boolean
        array of shape ``(n_samples, n)`` (``True`` = spin up) and
        *diagnostics* is a dict with keys ``n_chains``, ``betas``,
        ``Lambda``, ``converged_reason``, ``acceptance_rate``,
        ``mean_spins`` (average number of +1 spins per sample), ``device``
        (the resolved production device, or None), and
        ``round_trip_diagnostics``.

    Warns:
        Logs a warning if all coupling weights are zero (NRPT is
        unnecessary) or if all biases are identical (model has no
        per-variable preference).
    """
    import networkx as nx

    from hamon.nrpt import discover_chain_count, nrpt_adaptive

    biases = jnp.asarray(biases)
    weights = jnp.asarray(weights)
    n = biases.shape[0]
    edges_np = jnp.asarray(edges)

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
    node_edges: list[Edge] = [(nodes[int(e[0])], nodes[int(e[1])]) for e in edges_np]

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(int(e[0]), int(e[1])) for e in edges_np])
    coloring = nx.coloring.greedy_color(g, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1 if coloring else 1
    color_groups: list[list[AbstractNode]] = [[] for _ in range(n_colors)]
    for idx, color in coloring.items():
        color_groups[color].append(nodes[idx])
    free_blocks: list[SuperBlock] = [Block(group) for group in color_groups]

    # --- template EBM & program ---
    ebm = IsingEBM(nodes, node_edges, biases, weights, jnp.array(float(beta)))
    program = IsingSamplingProgram(ebm, free_blocks, [])

    # --- discover chain count ---
    key, k_disc, k_nrpt, k_samp, k_init = jax.random.split(key, 5)

    def _init_factory(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        keys = jax.random.split(k_init, n_chains)
        return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]

    # Device routing, stage 1: discovery probes start at 8 chains
    # (discover_chain_count's initial_n default).
    dev = resolve_entry_device(
        device, n_chains=8, n_nodes=n, arrays=(biases, weights, key)
    )

    discovery = discover_chain_count(
        k_disc,
        ebm=ebm,
        program=program,
        init_factory=_init_factory,
        beta_range=(0.0, float(beta)),
        gibbs_steps_per_round=gibbs_steps_per_round,
        target_acceptance=target_acceptance,
        device=dev,
    )

    # --- adaptive NRPT ---
    betas = discovery["betas"]
    n_chains = len(betas)

    # Device routing, stage 2: re-resolve with the discovered chain count. The
    # chain-count change forces a recompile anyway, so this is the one place
    # routing may legitimately switch devices (cheap CPU probes, accelerator
    # production).
    if dev is not None and isinstance(device, str) and device.lower() == "auto":
        dev = resolve_device("auto", score=work_score(n_chains, n))

    init_ebms = [ebm.with_beta(jnp.array(float(b))) for b in betas]
    init_progs = [program.with_ebm(e) for e in init_ebms]
    init_states = _init_factory(n_chains, init_ebms, init_progs)

    warm_states, nrpt_stats = nrpt_adaptive(
        k_nrpt,
        ebm=ebm,
        program=program,
        init_states=init_states,
        clamp_state=[],
        n_rounds=500,
        gibbs_steps_per_round=gibbs_steps_per_round,
        initial_betas=betas,
        n_tune=5,
        rounds_per_tune=200,
        device=dev,
    )

    # --- sample from cold chain (last index = highest beta) ---
    cold_state = warm_states[-1]
    schedule = SamplingSchedule(n_warmup, n_samples, steps_per_sample)
    obs_block = Block(nodes)
    raw_samples = sample_states(
        k_samp, program, schedule, cold_state, [], [obs_block], device=dev
    )
    samples = raw_samples[0]  # (n_samples, n)

    mean_spins = float(jnp.mean(jnp.sum(samples, axis=1).astype(jnp.float32)))

    from hamon.diagnostics import report_nrpt_diagnostics

    health = report_nrpt_diagnostics(nrpt_stats, samples=samples)

    diagnostics = {
        "n_chains": discovery["n_chains"],
        "betas": nrpt_stats["betas"],
        "Lambda": discovery["Lambda"],
        "converged_reason": discovery["converged_reason"],
        "acceptance_rate": nrpt_stats["acceptance_rate"],
        "mean_spins": mean_spins,
        "device": str(dev) if dev is not None else None,
        "health": health,
    }
    if "round_trip_diagnostics" in nrpt_stats:
        diagnostics["round_trip_diagnostics"] = nrpt_stats["round_trip_diagnostics"]

    return samples, diagnostics
