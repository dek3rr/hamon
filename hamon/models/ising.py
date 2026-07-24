# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import contextlib
import logging
from collections.abc import Sequence
from typing import Literal, overload

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
from hamon.factor import ModelSamplingProgram
from hamon.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from hamon.models.ebm import AbstractFactorizedEBM, EBMFactor
from hamon.observers import AbstractObserver, MomentAccumulatorObserver
from hamon.pgm import AbstractNode, SpinNode, _as_identity_seq, _fifo_cache

logger = logging.getLogger(__name__)

Edge = tuple[AbstractNode, AbstractNode]


# ``IsingEBM.factors`` runs once per chain per ``nrpt`` call, but its
# node-group Blocks depend only on the ``nodes``/``edges`` lists (passed
# through by reference), never on β or the weights — so cache them.
_FACTOR_BLOCK_CACHE: dict = {}


def _ising_factor_blocks(
    nodes: Sequence[AbstractNode], edges: Sequence[Edge]
) -> tuple[Block, Block, Block]:
    """Build (cached) the bias Block and the edge head/tail Blocks."""

    def build():
        return (
            nodes,  # pin the id()-keyed objects (see _fifo_cache)
            edges,
            Block(nodes),
            Block([x[0] for x in edges]),
            Block([x[1] for x in edges]),
        )

    hit = _fifo_cache(_FACTOR_BLOCK_CACHE, 32, (id(nodes), id(edges)), build)
    return hit[2], hit[3], hit[4]


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

    ``nodes`` and ``edges`` are stored as immutable, identity-hashed
    sequences (a single pytree leaf each): the EBM is passed to jitted
    functions (``hinton_init``, the NRPT round loop), and flattening plain
    lists would visit and hash every node and edge endpoint — O(|graph|)
    host work — on every call. They still index, iterate, and ``len()``
    like lists; ``with_beta`` passes the same objects through, which is
    what keeps the jit cache hitting.
    """

    nodes: Sequence[AbstractNode]
    biases: Array
    edges: Sequence[Edge]
    weights: Array
    beta: Array

    # .factors must recompute β*weights on every call — caching breaks AD tracer flow.

    def __init__(
        self,
        nodes: Sequence[AbstractNode],
        edges: Sequence[Edge],
        biases: Array,
        weights: Array,
        beta: Array,
    ):
        sd_map = {nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool_)}
        super().__init__(sd_map)
        self.nodes = _as_identity_seq(nodes)
        self.edges = _as_identity_seq(edges)
        # Cast β to the weights' dtype: a strong float64 β (x64 host app)
        # would otherwise promote the whole device sampling loop to float64.
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


class IsingSamplingProgram(ModelSamplingProgram):
    """Thin wrapper specializing :class:`ModelSamplingProgram` to an Ising model."""

    def __init__(
        self,
        ebm: IsingEBM,
        free_blocks: list[SuperBlock],
        clamped_blocks: list[Block],
        *,
        _gibbs_spec: BlockGibbsSpec | None = None,
    ):
        super().__init__(
            ebm,
            free_blocks,
            clamped_blocks,
            SpinGibbsConditional(),
            _gibbs_spec=_gibbs_spec,
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

    Units are drawn independently across all blocks at once; blocks may have
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
    indices = jnp.array(
        [node_map[n] for block in blocks for n in block], dtype=jnp.int32
    )
    n_units = sum(len(block) for block in blocks)

    probs = jax.nn.sigmoid(model.beta * model.biases[indices])
    draw = jax.random.bernoulli(key, p=probs, shape=(*batch_shape, n_units)).astype(
        jnp.bool_
    )

    result = []
    offset = 0
    for block in blocks:
        result.append(draw[..., offset : offset + len(block)])
        offset += len(block)

    return result


class _MomentsAndStateObserver(AbstractObserver):
    """Wrap a moment accumulator so the carry also tracks the latest state.

    The final recorded free-block state is exactly the chain state when
    sampling ends, which is what persistent-chain (PCD) training needs to
    seed the next gradient step. The carry is ``(inner_carry, last_state)``;
    the state slot must be seeded with a same-structure pytree (the initial
    state) so the scan carry structure is fixed.
    """

    inner: MomentAccumulatorObserver

    def __call__(
        self,
        program,
        state_free,
        state_clamped,
        carry,
        iteration,
        global_state=None,
    ):
        mem, _ = carry
        mem, obs = self.inner(
            program, state_free, state_clamped, mem, iteration, global_state
        )
        return (mem, state_free), obs


@overload
def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
    *,
    return_state: Literal[False] = False,
    device: DeviceLike = "auto",
) -> tuple[Array, Array]: ...


@overload
def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
    *,
    return_state: Literal[True],
    device: DeviceLike = "auto",
) -> tuple[Array, Array, list[Array]]: ...


def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
    *,
    return_state: bool = False,
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
        return_state: when True, also return the final free-block chain state
            (the state at the last recorded sample), so callers can continue
            the chain — e.g. persistent-chain (PCD) training.
    Returns:
        the first and second moment data, plus the final chain state when
        ``return_state`` is set.
    """
    moment_spec = []
    if first_moment_nodes:
        moment_spec.append([(node,) for node in first_moment_nodes])
    moment_spec.append(list(second_moment_edges))

    def _spin_transform(state, _):
        return [2 * x.astype(jnp.int8) - 1 for x in state]

    observer = MomentAccumulatorObserver(moment_spec, _spin_transform)

    if return_state:
        state_observer = _MomentsAndStateObserver(observer)
        (moments, final_state), _ = sample_with_observation(
            key,
            program,
            schedule,
            init_state,
            clamped_data,
            (observer.init(), init_state),
            state_observer,
            device=device,
        )
    else:
        final_state = None
        moments, _ = sample_with_observation(
            key,
            program,
            schedule,
            init_state,
            clamped_data,
            observer.init(),
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

    if return_state:
        return node_moments, edge_moments, final_state
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
    return_negative_state: bool = False,
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
        return_negative_state: when True, append the negative chains' final
            states (same structure as ``init_state_negative``) to the returned
            tuple. Feeding them back as the next step's ``init_state_negative``
            gives persistent-chain (PCD) training: the chains track the slowly
            moving model distribution instead of re-warming from scratch every
            gradient step, so the negative schedule's ``n_warmup`` can drop to
            ~0. (The positive phase is clamped to per-batch data, so persisting
            it across batches is not meaningful and it is not returned.)
    Returns:
        the weight gradients and the bias gradients (plus the final negative
        chain state when ``return_negative_state`` is set)
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

        if return_negative_state:
            moms_b_neg, moms_w_neg, final_state_neg = jax.vmap(
                lambda k, i: estimate_moments(
                    k,
                    bias_nodes,
                    weight_edges,
                    training_spec.program_negative,
                    training_spec.schedule_negative,
                    i,
                    conditioning_values,
                    return_state=True,
                )
            )(keys_neg, init_state_negative)
        else:
            final_state_neg = None
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
        if return_negative_state:
            return (
                grad_w,
                grad_b,
                (moms_b_pos, moms_w_pos),
                (moms_b_neg, moms_w_neg),
                final_state_neg,
            )
        return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)


# Node identity keys every downstream cache, so without this content-keyed
# memo a repeat ising_sample call with the same graph retraces and recompiles
# every jitted kernel (~3.5 s of XLA at 128²).
_GRAPH_CACHE: dict = {}

# Ceiling on the log2-degree bucket index used to split color classes (see
# _ising_graph): nodes with degree >= 2**this share the top bucket. It only
# bounds the block count on pathological degrees (2**24 ~ 16M) and is never
# reached by real graphs.
_MAX_DEGREE_BUCKET = 24

# A degree-bucket split of a color class is kept only when it cuts that class's
# padded-gather work to <= this fraction of the unsplit cost: each extra block
# is a sequential Gibbs group (a write-back barrier and its own kernel), so a
# split that saves little is a loss. Declining one costs at most
# 1/_MIN_SPLIT_SAVING - 1 of a class's gather, and so of the graph's.
_MIN_SPLIT_SAVING = 0.9


def _ising_graph(n: int, edges_np: np.ndarray):
    """(nodes, node_edges, free_blocks) for a variable graph, memoized.

    The coloring is deterministic, so equal-content graphs always produce the
    same block structure; reusing the node objects is what lets the structure
    cache and every jit cache hit on repeat calls.
    """
    from hamon.graph_utils import rlf_coloring

    def build():
        nodes: list[AbstractNode] = [SpinNode() for _ in range(n)]
        # .tolist() converts the whole edge array to Python ints in C, instead
        # of 2·|edges| per-element int(...) casts on numpy scalars.
        node_edges: list[Edge] = [(nodes[u], nodes[v]) for u, v in edges_np.tolist()]

        # Recursive-Largest-First coloring: each color class is an independent
        # set, so it becomes a block-Gibbs group updated in parallel.
        coloring = rlf_coloring(n, edges_np)
        n_colors = (max(coloring) + 1) if n else 1
        color_groups: list[list[int]] = [[] for _ in range(n_colors)]
        for idx in range(n):
            color_groups[coloring[idx]].append(idx)

        # Split each color by degree into log2 buckets: the block-Gibbs
        # conditional pads every node's neighbor gather to the block's max
        # degree, so one hub can force a whole color to that width (measured
        # 30x wasted work on a scale-free graph). No-op on regular lattices.
        degree = (
            np.bincount(edges_np.reshape(-1), minlength=n)
            if edges_np.size
            else np.zeros(n, dtype=int)
        )
        free_blocks: list[SuperBlock] = []
        for group in color_groups:
            if not group:
                continue
            g = np.array(group)
            bucket = np.minimum(
                np.log2(np.maximum(degree[g], 1)).astype(int), _MAX_DEGREE_BUCKET
            )
            buckets = np.unique(bucket)
            if buckets.size > 1:
                # Only pay the extra sequential groups when the padding they
                # remove is worth it: a near-regular class saves almost
                # nothing, a hub class saves almost everything.
                whole = g.size * int(degree[g].max())
                split = sum(
                    int((bucket == b).sum()) * int(degree[g[bucket == b]].max())
                    for b in buckets
                )
                if split > _MIN_SPLIT_SAVING * whole:
                    buckets = buckets[:0]
            if buckets.size:
                for b in buckets:
                    free_blocks.append(Block([nodes[i] for i in g[bucket == b]]))
            else:
                free_blocks.append(Block([nodes[i] for i in g]))
        return nodes, node_edges, free_blocks

    key = (n, edges_np.shape, edges_np.tobytes())
    return _fifo_cache(_GRAPH_CACHE, 8, key, build)


def _is_forest(n: int, edges_np: np.ndarray) -> bool:
    """Union-find acyclicity check on the coupling graph."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges_np.tolist():
        ru, rv = find(int(u)), find(int(v))
        if ru == rv:
            return False
        parent[ru] = rv
    return True


# Sweeps the descent may spend without improving on the best energy found so
# far before it gives up. The random half-flip mask descends on sparse graphs
# but oscillates on dense ones, so the loop needs a progress test rather than
# only a sweep ceiling.
_DESCENT_PATIENCE = 32


def _descent_probe(
    biases_np: np.ndarray,
    edges_np: np.ndarray,
    weights_np: np.ndarray,
    *,
    n_replicas: int = 64,
    seed: int = 0,
    patience: int = _DESCENT_PATIENCE,
) -> tuple[np.ndarray, float]:
    """Excitation costs from greedy descents: (per-site min 2|local field|, best E).

    Vectorized over replicas; sweep-style descent with a random half-mask per
    iteration (avoids two-spin flip oscillations), forcing the single best
    flip for replicas the mask left empty. Stops when no replica can improve,
    when the best energy has stalled for ``patience`` sweeps, or at a 10*n
    backstop. Host-side numpy — no jit involvement.
    """
    rng = np.random.default_rng(seed)
    n = biases_np.shape[0]
    ea, eb = edges_np[:, 0], edges_np[:, 1]
    s = rng.choice([-1.0, 1.0], size=(n_replicas, n))

    # The local field is a scatter-add over the symmetric coupling graph, which
    # as a sparse matmul is one C loop with no temporaries.
    from scipy.sparse import coo_array

    coupling = coo_array(
        (
            np.concatenate([weights_np, weights_np]),
            (np.concatenate([ea, eb]), np.concatenate([eb, ea])),
        ),
        shape=(n, n),
    ).tocsr()  # coo -> csr sums duplicate edges rather than dropping them

    def local_field(s):
        return biases_np[None, :] + (coupling @ s.T).T

    best_e, stalled = np.inf, 0
    for _ in range(10 * n):
        lf = local_field(s)
        gains = -2.0 * s * lf
        improvable = gains > 1e-12
        if not improvable.any():
            break
        # Progress test, from the local field already in hand: summing
        # s_i·lf_i gives Σb·s + 2Σ_<ij>J s_i s_j, so E = -(Σb·s + Σs·lf)/2 —
        # no second pass over the edge list.
        e = float((-((s @ biases_np) + (s * lf).sum(1)) / 2.0).min())
        if e < best_e - 1e-12:
            best_e, stalled = e, 0
        else:
            stalled += 1
            if stalled >= patience:
                break
        mask = improvable & (rng.random((n_replicas, n)) < 0.5)
        empty = improvable.any(1) & ~mask.any(1)
        if empty.any():
            best = np.argmax(np.where(improvable, gains, -np.inf), axis=1)
            mask[empty, best[empty]] = True
        s = np.where(mask, -s, s)
    lf = local_field(s)
    energies = -(s @ biases_np)
    if len(weights_np):
        energies -= (weights_np[None, :] * s[:, ea] * s[:, eb]).sum(1)
    return 2.0 * np.abs(lf).min(axis=0), float(energies.min())


def ising_excitation_costs(
    biases, edges, weights, *, n_replicas: int = 64, seed: int = 0
) -> tuple[np.ndarray, float, str]:
    """Elementary excitation-cost spectrum of an Ising landscape.

    Returns ``(costs, energy_scale, method)``. Field-free forests are exact:
    bond defects are independent with cost ``2|J|`` and ``|E_GS| = Σ|J|``.
    Anything else uses the greedy-descent probe (``2|local field|`` per-site
    minima across replicas, energy scale = best minimum found). On highly
    regular graphs the probe minima can have a zero local field at every site
    (a genuine degeneracy), leaving no positive costs; there we fall back to
    the coupling-magnitude spectrum ``2|J|``.
    """
    biases_np = np.asarray(biases, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    edges_np = np.asarray(edges)
    n = biases_np.shape[0]
    if not biases_np.any() and _is_forest(n, edges_np):
        return 2.0 * np.abs(weights_np), float(np.abs(weights_np).sum()), "tree-exact"
    costs, best_e = _descent_probe(
        biases_np, edges_np, weights_np, n_replicas=n_replicas, seed=seed
    )
    if not (costs > 1e-12).any():
        return 2.0 * np.abs(weights_np), abs(best_e), "coupling-fallback"
    return costs, abs(best_e), "descent-probe"


def ising_estimate_beta(
    biases,
    edges,
    weights,
    *,
    gap_tol: float = 1e-3,
    n_replicas: int = 64,
    seed: int = 0,
):
    """Estimate the coldest useful β for ground-state search on an Ising model.

    A thin front end over :func:`hamon.estimate_beta_max`: extracts the
    excitation-cost spectrum (exact on field-free forests, greedy-descent
    probe elsewhere) and selects the smallest β whose predicted equilibrium
    excess energy is at most ``gap_tol`` of the ground-state scale. Runs on
    the host in milliseconds — no tuning, no compiles. Returns a
    :class:`hamon.BetaEstimate`.
    """
    from hamon.advisor import estimate_beta_max

    costs, scale, method = ising_excitation_costs(
        biases, edges, weights, n_replicas=n_replicas, seed=seed
    )
    return estimate_beta_max(costs, scale, gap_tol=gap_tol, method=method)


def ising_sample(
    biases: Shaped[Array, " n"],
    edges: Shaped[Array, "m 2"],
    weights: Shaped[Array, " m"],
    *,
    key: Key[Array, ""],
    beta: float | str = 1.0,
    n_samples: int = 1000,
    n_warmup: int = 500,
    steps_per_sample: int = 1,
    target_acceptance: float = 0.5,
    max_chains: int = 128,
    device: DeviceLike = "auto",
) -> tuple[Bool[Array, "n_samples n"], dict]:
    r"""Sample from an Ising model Boltzmann distribution via fully autotuned NRPT.

    A thin Ising-specific front end over :func:`hamon.autosample`: it builds and
    colors the graph, then **autotunes the full NRPT configuration** — chain
    count, local-exploration count (``gibbs_steps_per_round``), and schedule —
    before drawing from the cold chain. Unlike earlier versions, the
    exploration count is no longer a fixed argument; it is discovered (and
    device-calibrated) automatically.

    A warning is logged if all coupling weights are zero (NRPT is unnecessary)
    or if all biases are identical (the model has no per-variable preference).

    The energy function is

    $$\mathcal{E}(s) = -\beta \left( \sum_i b_i s_i
        + \sum_{(i,j)} J_{ij} s_i s_j \right)$$

    Args:
        biases: per-node bias array of shape ``(n,)``.
        edges: integer index pairs of shape ``(m, 2)``.
        weights: per-edge coupling of shape ``(m,)``.
        key: JAX PRNG key.
        beta: inverse temperature for the target distribution, or ``"auto"``
            to choose it for ground-state search: the excitation-cost spectrum
            of the landscape (exact on field-free forests, greedy-descent
            probe elsewhere) picks the smallest β whose predicted equilibrium
            excess energy is ≤ 0.1% of the ground-state scale — see
            :func:`ising_estimate_beta`. The estimate and its rationale are
            returned under ``diagnostics["beta_estimate"]``.
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
    """
    from hamon.autotune import autosample

    biases = jnp.asarray(biases)
    weights = jnp.asarray(weights)
    n = biases.shape[0]
    # Graph construction indexes every edge endpoint on the host; np.asarray
    # pulls the array over once instead of ~2·n_edges blocking transfers.
    edges_np = np.asarray(edges)

    # The cost spectrum picks beta under "auto" and tells the advisor where the
    # resolved beta's thermal floor sits (so a warm draw reads beta-limited even
    # while still improving). Host-side milliseconds.
    from hamon.advisor import estimate_beta_max, excess_energy

    if isinstance(beta, str) and beta != "auto":
        raise ValueError(f"beta must be a float or 'auto', got {beta!r}")
    beta_estimate = None
    search_context = None
    try:
        costs, e_scale, method = ising_excitation_costs(biases, edges_np, weights)
        estimate = estimate_beta_max(costs, e_scale, method=method)
        if isinstance(beta, str):
            beta_estimate = estimate
            beta = estimate.beta_max
            logger.info("beta='auto': %s", estimate.summary())
        pos = costs[costs > 1e-12]
        if e_scale > 0 and pos.size:
            search_context = {
                "predicted_floor_rel": excess_energy(pos, float(beta)) / e_scale,
                "estimator_beta": estimate.beta_max,
            }
    except ValueError:
        if isinstance(beta, str):
            raise  # "auto" needs a cost spectrum; plain draws don't

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

    # Build (or reuse) the colored graph: node identity keys every downstream
    # cache, so a repeat call must reuse the same node objects or every kernel
    # recompiles.
    nodes, node_edges, free_blocks = _ising_graph(n, edges_np)

    # --- template EBM & program ---
    ebm = IsingEBM(nodes, node_edges, biases, weights, jnp.array(float(beta)))
    program = IsingSamplingProgram(ebm, free_blocks, [])

    # --- autotune (N + exploration + schedule) and draw ---
    key, k_init, k_auto = jax.random.split(key, 3)

    def _init_factory(n_chains, ebms, programs):
        # One stacked draw at the max_chains ceiling, sliced to the live count:
        # a per-chain loop pays a stack compile plus a per-N key split, and a
        # direct (n_chains,) batch recompiles hinton_init per probe width.
        fb = programs[0].gibbs_spec.free_blocks
        full = hinton_init(k_init, ebms[0], fb, (max_chains,))
        return [b[:n_chains] for b in full]

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
        search_context=search_context,
        device=device,
    )

    if beta_estimate is not None:
        report.beta_estimate = beta_estimate

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
        "beta_estimate": beta_estimate,
        "search_advice": report.search_advice,
    }
    return samples, diagnostics
