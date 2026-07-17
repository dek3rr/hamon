# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)
# Changes: added global_state fast path to avoid redundant reconstruction; added dtype parameter to MomentAccumulatorObserver

import abc
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int, PyTree

from hamon.block_management import Block, block_state_to_global, from_global_state

if TYPE_CHECKING:
    from hamon.block_sampling import _State, BlockSamplingProgram

from hamon.pgm import AbstractNode

ObserveCarry = TypeVar("ObserveCarry", bound=PyTree)


class AbstractObserver(eqx.Module):
    """
    Interface for objects that inspect the sampling program while it is running.

    A concrete Observer is called once per block-sampling iteration and can maintain an
    arbitrary "carry" state across calls (e.g. running averages, histogram
    buffers, log-probs, etc.).
    """

    @abc.abstractmethod
    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: ObserveCarry,
        iteration: Int[Array, ""],
        global_state: list[PyTree[Array]] | None = None,
    ) -> tuple[ObserveCarry, PyTree]:
        """Make an observation.

        This function is called at the end of a block-sampling iteration and can record information about the
        current state of the sampling program that might be useful for something later.

        **Arguments:**

        - `program`: The sampling program that is running when this function is called.
        - `state_free`: The current state of the free nodes involved in the sampling program.
        - `state_clamped`: The state of the clamped nodes involved in the sampling program.
        - `carry`: The "memory" available to this observer.
        - `iteration`: How many iterations of block sampling have happened before this function was called.
        - `global_state`: The precomputed global state as returned by `_run_blocks`. When provided
            (always the case inside `sample_with_observation`), observers use it directly to avoid
            an extra `block_state_to_global` call. When ``None`` (user code calling an observer
            directly), observers reconstruct it internally.

        **Returns:**

        A tuple, where the first element is the updated carry, and the second is a PyTree that will be
        recorded by the sampler.
        """
        return NotImplemented

    def init(self) -> PyTree:
        """Initialize the memory for the observer. Defaults to None."""
        return None


class StateObserver(AbstractObserver):
    """
    Observer which logs the raw state of some set of nodes.

    This observer is stateless: its carry is always ``None`` and ``iteration``
    is ignored.

    **Attributes:**

    - `blocks_to_sample`: the list of `Block`s which the states are logged for
    """

    blocks_to_sample: list[Block]

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list["_State"],
        state_clamped: list["_State"],
        carry: None,
        iteration: Int[Array, ""],
        global_state: list[PyTree[Array]] | None = None,
    ) -> tuple[None, PyTree]:
        """Simply returns the state of the blocks that are being logged to be recorded by the sampler."""
        if global_state is None:
            global_state = block_state_to_global(
                state_free + state_clamped, program.gibbs_spec
            )
        sampled_state = from_global_state(
            global_state, program.gibbs_spec, self.blocks_to_sample
        )
        return None, sampled_state


def _f_identity(*x):
    return x[0]


class MomentAccumulatorObserver(AbstractObserver):
    r"""
    Observer that accumulates and updates the provided moments.

    It doesn't log any samples, and will only accumulate moments. Note that this observer does not
    scale the accumulated values by the number of times it was called. It simply records a running sum of a product
    of some state variables,

    $$\sum_i f(x_1^i) f(x_2^i) \dots f(x_N^i)$$

    **Attributes:**

    - `blocks_to_sample`: the blocks to accumulate the moments over. These
        are for constructing the final state, and aren't truly "blocks"
        in the algorithmic sense (they can be connected to each other).
        There is one block per node type.
    - `flat_nodes_list`: a list of all of the nodes in the moments (each
        occurring only once, so len(set(x)) = len(x)).
    - `flat_to_type_slices_list`: a list over node types in which each element
        is an array of indices of the `flat_node_list` which that type
        corresponds to
    - `flat_to_full_moment_slices`: a list over moment types in which each
        element is a 2D array, which matches the shape of the `moment_spec[i]`
        and of which each element is the index in the `flat_node_list`.
    - `f_transform`: the element-wise transformation $f$ to apply to sample values before
        accumulation.
    - `_flat_scatter_index`: precomputed concatenation of all `flat_to_type_slices_list`
        arrays, used to build `flat_state` in a single scatter call.
    - `_flat_scatter_sizes`: number of entries contributed by each node type, used to
        split the concatenated sampled state before scattering.
    - `_flat_value_order`: precomputed ``argsort(_flat_scatter_index)``; used in
        ``__call__`` to permute the concatenated sampled values into flat-node
        order without allocating a zeros array.
    - `_accumulate_dtype`: dtype for the accumulator, fixed at construction time.
    """

    blocks_to_sample: list[Block]
    flat_nodes_list: list[AbstractNode]
    flat_to_type_slices_list: list[Int[Array, " nodes_in_slice"]]
    flat_to_full_moment_slices: list[Int[Array, "num_groups nodes_in_moment"]]
    f_transform: Callable
    _flat_scatter_index: Array  # shape: (total_flat_nodes,)
    _flat_scatter_sizes: list[int]  # len == number of node types
    _flat_value_order: Array  # shape: (total_flat_nodes,) — argsort of scatter index
    _flat_state_size: int
    _accumulate_dtype: jnp.dtype

    def __init__(
        self,
        moment_spec: Sequence[Sequence[Sequence[AbstractNode]]],
        f_transform: Callable = _f_identity,
        dtype: jnp.dtype = jnp.float32,
    ):
        r"""
        Create a MomentAccumulatorObserver.

        **Arguments:**

        - `moment_spec`: A 3 depth sequence. The first is a sequence over different moment types.
            A given moment type should have the same number of nodes in each moment. Then for each
            moment type, there is a sequence over moments. Each given moment is defined by a certain
            set of nodes.

            For example, to get the first and second moments on a simple o-o graph:

            [
                [(node1,), (node2,)],
                [(node1, node2)]
            ]

        - `f_transform`: A function that takes in (state, blocks) and returns something with the same
            structure as state. Defines a transformation $y=f(x)$ so accumulated moments are
            $\langle f(x_1) f(x_2) \rangle$.

        - `dtype`: Accumulator dtype, fixed at construction. Defaults to `jnp.float32`. Use
            `jnp.float64` for double-precision models. Fixing this here avoids a per-step cast
            inside the scan body.
        """
        self.f_transform = f_transform
        self._accumulate_dtype = jax.dtypes.canonicalize_dtype(dtype)

        # --- Pass 1: deduplicate nodes and build moment index slices --------
        flat_nodes_list: list[AbstractNode] = []
        node_to_flat_idx: dict[AbstractNode, int] = {}
        flat_to_full_moment_slices: list[np.ndarray] = []

        for moment in moment_spec:
            shape = (len(moment), len(moment[0]))
            moment_slice = np.zeros(shape, dtype=int)

            for j, nodes in enumerate(moment):
                for k, node in enumerate(nodes):
                    idx = node_to_flat_idx.get(node, -1)
                    if idx == -1:
                        idx = len(flat_nodes_list)
                        node_to_flat_idx[node] = idx
                        flat_nodes_list.append(node)
                    moment_slice[j, k] = idx

            flat_to_full_moment_slices.append(moment_slice)

        # Pass 2: build blocks_to_sample and type slices from the
        # deduplicated flat_nodes_list, making _flat_scatter_index a true
        # permutation (no duplicate targets).
        nodes_by_type: dict[type, list[AbstractNode]] = defaultdict(list)
        flat_to_type_slices: dict[type, list[int]] = defaultdict(list)

        for idx, node in enumerate(flat_nodes_list):
            node_type = node.__class__
            nodes_by_type[node_type].append(node)
            flat_to_type_slices[node_type].append(idx)

        blocks_to_sample: list[Block] = []
        flat_to_type_slices_list: list[jnp.ndarray] = []

        for node_type, nodes in nodes_by_type.items():
            blocks_to_sample.append(Block(nodes))
            flat_to_type_slices_list.append(
                jnp.array(flat_to_type_slices[node_type], dtype=int)
            )

        self.flat_nodes_list = flat_nodes_list
        self.flat_to_full_moment_slices = [
            jnp.array(s, dtype=int) for s in flat_to_full_moment_slices
        ]
        self.blocks_to_sample = blocks_to_sample
        self.flat_to_type_slices_list = flat_to_type_slices_list

        # Precompute scatter index and its inverse (argsort).
        self._flat_scatter_index = (
            jnp.concatenate(flat_to_type_slices_list)
            if flat_to_type_slices_list
            else jnp.array([], dtype=int)
        )
        self._flat_scatter_sizes = [len(s) for s in flat_to_type_slices_list]
        self._flat_state_size = len(flat_nodes_list)

        # _flat_value_order[i] = source position for flat position i, turning
        # __call__ into a pure gather (no zeros + scatter).
        if self._flat_scatter_index.size > 0:
            self._flat_value_order = jnp.argsort(self._flat_scatter_index)
        else:
            self._flat_value_order = jnp.array([], dtype=int)

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: list[Array],
        iteration: Int[Array, ""],
        global_state: list[PyTree[Array]] | None = None,
    ) -> tuple[list[Array], PyTree]:
        """Accumulate the moments via `carry`. Does not return anything for the sampler to write down."""
        if global_state is None:
            global_state = block_state_to_global(
                state_free + state_clamped, program.gibbs_spec
            )

        sampled_state = from_global_state(
            global_state, program.gibbs_spec, self.blocks_to_sample
        )
        sampled_state = list(self.f_transform(sampled_state, self.blocks_to_sample))

        # Concatenate all sampled values (ordered by type-block), then permute
        # into flat-node order via a precomputed argsort — no zeros allocation.
        flat_values = jnp.concatenate([jnp.ravel(s) for s in sampled_state])
        flat_state = flat_values.astype(self._accumulate_dtype)[self._flat_value_order]

        def accumulate_moment(mem_entry, sl):
            update = jnp.prod(flat_state[sl], axis=1)
            return mem_entry + update

        mem = jax.tree.map(accumulate_moment, carry, self.flat_to_full_moment_slices)
        return mem, None

    def init(self) -> list[Array]:
        """Initialize the moment accumulators."""
        return [
            jnp.zeros(x.shape[0], dtype=self._accumulate_dtype)
            for x in self.flat_to_full_moment_slices
        ]


# ---------------------------------------------------------------------------
# NRPT observers
# ---------------------------------------------------------------------------


class AbstractNRPTObserver(eqx.Module):
    """Observer for NRPT rounds, called once per round after Gibbs sweeps and swaps.

    Concrete subclasses must implement ``__call__`` and may override ``init``
    to provide a non-trivial carry.  The ``observation`` returned by
    ``__call__`` is stacked by ``lax.scan`` into a pytree with a leading axis
    of size ``n_rounds``.  Return ``None`` as the observation for
    accumulate-only observers that do not need per-round storage.
    """

    @abc.abstractmethod
    def __call__(
        self,
        stacked_states: list[Array],
        base_energies: Array,
        round_idx: Int[Array, ""],
        carry: ObserveCarry,
    ) -> tuple[ObserveCarry, PyTree]:
        """Observe one NRPT round.

        **Arguments:**

        - `stacked_states`: Per-block arrays, each of shape ``(n_chains, ...)``.
          The cold chain (target) is at index ``-1``.
        - `base_energies`: Shape ``(n_chains,)`` base energies (no β factor),
          aligned with ``stacked_states`` — ``base_energies[c]`` is the energy
          of the state at chain position ``c`` after this round's swaps.
        - `round_idx`: Zero-based round counter.
        - `carry`: Arbitrary pytree state threaded across rounds.

        **Returns:**

        ``(updated_carry, observation)`` — *observation* is stacked by
        ``lax.scan``; use ``None`` for accumulate-only mode.
        """
        ...

    def init(self) -> PyTree:
        """Initialize the observer carry.  Defaults to ``None``."""
        return None

    @property
    def masking_safe(self) -> bool:
        """Whether this observer is correct under chain masking (``pad_chains_to``).

        ``True`` only for observers that read **exclusively live** ladder
        positions — not a raw ``-1`` tail index (which under padding records a
        divergent padding copy of the cold chain) and not an all-chains aggregate
        (which padding would pollute). Default ``False``; ``nrpt`` refuses to pad
        when an observer is not masking-safe.
        """
        return False


class NRPTStateObserver(AbstractNRPTObserver):
    """Collect raw chain states at specified chain indices each round.

    This observer is stateless (carry is always ``None``).  The returned
    observation is a list of arrays — one per free block — each of shape
    ``(len(chain_indices), ...)``.  After ``lax.scan`` stacking the leading
    axis becomes ``n_rounds``.

    **Attributes:**

    - `chain_indices`: Tuple of chain positions to record.  Use ``(-1,)``
      to collect only the cold chain (the default).
    """

    chain_indices: tuple[int, ...]

    def __init__(self, chain_indices: tuple[int, ...] = (-1,)):
        self.chain_indices = chain_indices

    def __call__(
        self,
        stacked_states: list[Array],
        base_energies: Array,
        round_idx: Int[Array, ""],
        carry: None,
    ) -> tuple[None, list[Array]]:
        idx = jnp.array(self.chain_indices)
        return None, [s[idx] for s in stacked_states]


class ColdIndexObserver(AbstractNRPTObserver):
    """Record one chain at a **traced** ladder index each round.

    Like ``NRPTStateObserver((i,))`` but ``i`` is traced data, not a static
    attribute, so a chain-masked (padded) draw at different *live* chain counts
    reuses ONE compiled observer round loop instead of recompiling per N — the
    live cold chain always sits at absolute index ``n_chains - 1`` of the padded
    ladder. Observation layout matches ``NRPTStateObserver`` with a single index:
    one ``(1, ...)`` array per free block, stacked to ``(n_rounds, 1, ...)`` by
    ``lax.scan``.
    """

    idx: Array  # traced scalar: absolute live position to record

    def __init__(self, idx):
        self.idx = jnp.asarray(idx, dtype=jnp.int32)

    @property
    def masking_safe(self) -> bool:
        # Reads a single caller-supplied live index (the cold chain at
        # n_chains-1), never the padding tail — safe under pad_chains_to.
        return True

    def __call__(
        self,
        stacked_states: list[Array],
        base_energies: Array,
        round_idx: Int[Array, ""],
        carry: None,
    ) -> tuple[None, list[Array]]:
        i = self.idx
        return None, [s[i][None] for s in stacked_states]


class ColdChainObserver(AbstractNRPTObserver):
    """Record the cold chain's state AND base energy each round.

    Powers the ground-state-search advisor (:func:`hamon.diagnose_search`):
    the per-round energy trace costs nothing extra to record — ``nrpt`` hands
    every observer the post-swap ``base_energies`` anyway.

    ``idx=None`` reads the static ``-1`` tail position (unpadded draws);
    an integer ``idx`` is traced like :class:`ColdIndexObserver` (the live
    cold chain at absolute position ``n_chains - 1`` of a padded ladder) and
    is masking-safe.

    Observation: ``{"states": [(1, ...) per free block], "energy": ()}``,
    scan-stacked to ``{"states": [(n_rounds, 1, ...)], "energy": (n_rounds,)}``.
    The energy is the UNSCALED base energy (``E_beta = beta * E_base``); in
    affine/``ebm_ref0`` mode it is the swap-path Δ = E₁ − E₀, not E₁.
    """

    idx: Array | None  # None => static -1; else traced live position

    def __init__(self, idx: int | Array | None = None):
        self.idx = None if idx is None else jnp.asarray(idx, dtype=jnp.int32)

    @property
    def masking_safe(self) -> bool:
        # The traced-index form reads only the caller-supplied live position;
        # the static -1 form would read the padding tail, so it is not safe.
        return self.idx is not None

    def __call__(
        self,
        stacked_states: list[Array],
        base_energies: Array,
        round_idx: Int[Array, ""],
        carry: None,
    ) -> tuple[None, dict]:
        i = -1 if self.idx is None else self.idx
        return None, {
            "states": [s[i][None] for s in stacked_states],
            "energy": base_energies[i],
        }


class NRPTEnergyObserver(AbstractNRPTObserver):
    r"""Accumulate per-chain mean base energy μ(β_i) = E[V^(β_i)] for
    thermodynamic integration of the log normalizing constant.

    Each round the NRPT loop hands every observer the post-swap
    ``base_energies`` (shape ``(n_chains,)``, aligned to chain/β positions).
    Under stationarity these are samples of ``V^(β_i)``, so a running mean
    estimates μ(β_i), which integrates to ``log Z`` via
    :func:`hamon.round_trips.thermodynamic_integration`.

    Accumulate-only: it returns ``None`` as the per-round observation, so it
    adds no per-round output stack — only a tiny carry ``(sum_E, count)``. Read
    the mean energies after a run from ``stats["observer_carry"]``::

        obs = NRPTEnergyObserver(n_chains)
        states, stats = tune_schedule(..., observer=obs)
        sum_E, count = stats["observer_carry"]
        mean_energies = sum_E / count

    or use the one-call
    :func:`hamon.round_trips.nrpt_log_normalizing_constant`.

    .. note::
       Attaching any observer (this one included) switches the NRPT round loop
       from the dynamic-trip-count ``lax.fori_loop`` fast path to ``lax.scan``,
       which compiles once per distinct ``n_rounds``. The default no-observer
       path is unaffected. In ``tune_schedule`` the observer is attached only to
       the production run, so accumulation is naturally post-tuning (no burn-in
       from the tuning phases is included).

    **Attributes:**

    - `n_chains`: number of chains in the ladder (sets the carry shape).
    - `_dtype`: accumulator dtype, fixed at construction.
    """

    n_chains: int
    _dtype: jnp.dtype

    def __init__(self, n_chains: int, dtype: jnp.dtype = jnp.float32):
        """Create an energy observer.

        **Arguments:**

        - `n_chains`: the number of chains (``len(betas)``).
        - `dtype`: accumulator dtype, fixed at construction. Defaults to
            ``jnp.float32``; use ``jnp.float64`` for double-precision models.
        """
        self.n_chains = n_chains
        self._dtype = jax.dtypes.canonicalize_dtype(dtype)

    def __call__(
        self,
        stacked_states: list[Array],
        base_energies: Array,
        round_idx: Int[Array, ""],
        carry: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        sum_E, count = carry
        return (sum_E + base_energies.astype(self._dtype), count + 1), None

    def init(self) -> tuple[Array, Array]:
        """Initialize the ``(sum_E, count)`` accumulator carry."""
        return (
            jnp.zeros(self.n_chains, dtype=self._dtype),
            jnp.zeros((), dtype=jnp.int32),
        )


def nrpt_node_samples(
    observations: list[Array],
    program: "BlockSamplingProgram",
    nodes: Sequence[AbstractNode],
    chain_index: int = 0,
) -> Array:
    """Reorder NRPT observer output into node order.

    ``stats["observations"]`` from [`hamon.nrpt`][] with an
    [`hamon.NRPTStateObserver`][] is a list of per-free-block arrays of shape
    ``(n_rounds, len(chain_indices), block_len, ...)`` — block-local layout,
    in free-block order. Assembling per-node samples from that requires
    concatenating blocks and inverting the block→node permutation, which is
    easy to get silently wrong (a forgotten inversion produces
    plausible-looking but scrambled samples). This helper does it once,
    correctly:

    ```python
    obs = NRPTStateObserver(chain_indices=(-1,))
    states, stats = nrpt(..., observer=obs)
    samples = nrpt_node_samples(stats["observations"], program, nodes)
    # samples[r, i] is the state of nodes[i] at round r — guaranteed.
    ```

    **Arguments:**

    - `observations`: ``stats["observations"]`` — one array per free block.
    - `program`: The sampling program the observations came from (any of the
      per-chain programs, or the template program in temperature-linear mode;
      only its ``gibbs_spec`` is used).
    - `nodes`: The nodes, in the order you want the output columns. All must
      share one node type and belong to free (not clamped) blocks.
    - `chain_index`: Which entry of the observer's ``chain_indices`` tuple to
      extract (default 0). Note this indexes the *recorded* chains, not the
      temperature ladder: for ``NRPTStateObserver(chain_indices=(0, -1))``,
      the cold chain is ``chain_index=1``.

    **Returns:**

    Array of shape ``(n_rounds, len(nodes), ...)`` with column ``i``
    holding the state of ``nodes[i]``.
    """
    spec = program.gibbs_spec
    free_blocks = spec.free_blocks
    if len(observations) != len(free_blocks):
        raise ValueError(
            f"Expected one observation array per free block ({len(free_blocks)}), got {len(observations)}."
        )
    if not nodes:
        raise ValueError("nodes must be non-empty.")

    node_type = type(nodes[0])
    for node in nodes:
        if type(node) is not node_type:
            raise ValueError(
                "All nodes must share one node type; mixed-type extraction "
                "is ambiguous because blocks of different types live in "
                "different state arrays."
            )

    # Concatenate matching free-block observations in free-block order and
    # index by each node's column, derived straight from free_blocks — correct
    # under any global-state layout, unlike node_global_location_map.
    same_type = [
        obs
        for obs, block in zip(observations, free_blocks)
        if block.node_type is node_type
    ]
    if not same_type:
        raise ValueError(f"No free blocks of node type {node_type.__name__}.")
    concat = jnp.concatenate(same_type, axis=2)

    node_to_column = {}
    offset = 0
    for block in free_blocks:
        if block.node_type is node_type:
            for k, n in enumerate(block.nodes):
                node_to_column[n] = offset + k
            offset += len(block.nodes)

    positions = []
    for node in nodes:
        column = node_to_column.get(node)
        if column is None:
            if spec.node_global_location_map.get(node) is None:
                raise ValueError(
                    "Node not found in the program's BlockSpec; samples can only "
                    "be extracted for nodes that belong to this program."
                )
            raise ValueError(
                "Node belongs to a clamped block; only free-block states are observed by NRPT observers."
            )
        positions.append(column)

    per_chain = concat[:, chain_index]
    return jnp.take(per_chain, jnp.array(positions), axis=1)
