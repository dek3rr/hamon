"""Gaussian Markov random fields: continuous-state EBMs with exact block Gibbs.

The continuous counterpart of the Ising/discrete stack. States live in ℝ
(:class:`~hamon.pgm.GaussianNode`, float32), the energy is quadratic,

$$\\mathcal{E}_\\beta(x) = \\beta \\left( \\tfrac{1}{2} \\sum_i d_i x_i^2
    - \\sum_i h_i x_i + \\sum_{(i,j)} c_{ij} x_i x_j \\right)$$

i.e. a multivariate normal with precision β·P (P_ii = d_i, P_ij = c_ij) and
mean P⁻¹h, and the single-site conditionals are themselves Gaussian —

$$x_i \\mid x_{\\setminus i} \\sim \\mathcal{N}\\!\\left(
    \\frac{h_i - \\sum_j c_{ij} x_j}{d_i},\\; \\frac{1}{\\beta d_i} \\right)$$

— so block Gibbs over a graph coloring is *exact*, just as for the discrete
models: within a color class the conditionals are independent scalar
Gaussians (no linear solve anywhere).

Positive definiteness of P is the caller's responsibility; strict diagonal
dominance (``d_i > Σ_j |c_ij|`` for every node) is a simple sufficient
condition. All interaction arrays are linear in β, so NRPT's temperature-linear
template mode applies bit-exactly. The one structural difference from the
discrete models: an unbounded state space has no proper β = 0 member (the
conditional variance 1/(β·d) diverges), so :class:`GaussianEBM` reports
``proper_at_beta_zero = False`` and NRPT rejects ladders that start at exactly
β = 0 — use ``beta_range=(β_min > 0, β_max)``.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree

from hamon.block_management import Block, BlockSpec, from_global_state
from hamon.block_sampling import BlockGibbsSpec, _State
from hamon.conditional_samplers import AbstractParametricConditionalSampler
from hamon.factor import ModelSamplingProgram
from hamon.interaction import InteractionGroup, interaction_float_dtype
from hamon.models.ebm import AbstractFactorizedEBM, EBMFactor
from hamon.pgm import AbstractNode, GaussianNode, _as_identity_seq, _fifo_cache

__all__ = [
    "QuadraticSelfInteraction",
    "QuadraticPairInteraction",
    "QuadraticSelfEBMFactor",
    "QuadraticPairEBMFactor",
    "GaussianGibbsConditional",
    "GaussianEBM",
    "GaussianSamplingProgram",
    "gaussian_init",
]


class QuadraticSelfInteraction(eqx.Module):
    """Per-node quadratic + linear energy terms ``½·diag·x² − lin·x``.

    **Attributes:**

    - `diag`: coefficient of ``½ x²`` per head node (the precision diagonal,
      already scaled by β). Must be positive for the conditional to be proper.
    - `lin`: coefficient of ``x`` per head node (the linear/natural term,
      already scaled by β).
    """

    diag: Array
    lin: Array


class QuadraticPairInteraction(eqx.Module):
    """Per-edge bilinear energy term ``coupling · x_head · x_tail``.

    **Attributes:**

    - `coupling`: energy coefficient per head node (already scaled by β).
    """

    coupling: Array


class QuadraticSelfEBMFactor(EBMFactor):
    """Batch of single-node energy terms ``½·diag·x² − lin·x``.

    ``diag`` and ``lin`` carry the *energy* coefficients (β folded in by the
    EBM, mirroring the Ising factors), one per node in ``node_group``.
    """

    diag: Array
    lin: Array

    def __init__(self, node_group: Block, diag: Array, lin: Array):
        super().__init__([node_group])
        n = len(node_group.nodes)
        if diag.shape[0] != n or lin.shape[0] != n:
            raise RuntimeError(
                "diag and lin must have one entry per node in node_group."
            )
        self.diag = diag
        self.lin = lin

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [
            InteractionGroup(
                QuadraticSelfInteraction(self.diag, self.lin),
                self.node_groups[0],
                [],
            )
        ]

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        (x,) = from_global_state(global_state, block_spec, self.node_groups)
        return jnp.sum(0.5 * self.diag * x * x - self.lin * x)


class QuadraticPairEBMFactor(EBMFactor):
    """Batch of pairwise bilinear energy terms ``coupling · x_i · x_j``.

    ``node_groups`` is ``[heads, tails]`` (one node pair per batch index);
    ``coupling`` carries the energy coefficient per pair (β folded in by the
    EBM). Sampling-side, the factor emits one merged interaction group covering
    both directions — the bilinear form is symmetric, the same head-merge the
    square discrete factors use.
    """

    coupling: Array

    def __init__(self, node_groups: list[Block], coupling: Array):
        super().__init__(node_groups)
        if len(node_groups) != 2:
            raise RuntimeError(
                "QuadraticPairEBMFactor needs exactly [heads, tails] node groups."
            )
        if coupling.shape[0] != len(node_groups[0].nodes):
            raise RuntimeError("coupling must have one entry per node pair.")
        self.coupling = coupling

    def to_interaction_groups(self) -> list[InteractionGroup]:
        heads, tails = self.node_groups
        merged_head = Block(list(heads.nodes) + list(tails.nodes))
        merged_tail = Block(list(tails.nodes) + list(heads.nodes))
        coupling2 = jnp.concatenate([self.coupling, self.coupling])
        return [
            InteractionGroup(
                QuadraticPairInteraction(coupling2), merged_head, [merged_tail]
            )
        ]

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        x_h, x_t = from_global_state(global_state, block_spec, self.node_groups)
        return jnp.sum(self.coupling * x_h * x_t)


class GaussianGibbsConditional(AbstractParametricConditionalSampler):
    r"""Exact Gaussian Gibbs update for continuous nodes.

    Accumulates the conditional's natural parameters from quadratic
    interactions —

    $$\text{prec}_i = \sum \text{diag}_i, \qquad
      \eta_i = \sum \text{lin}_i - \sum \text{coupling}_{ij}\, x_j$$

    — and draws ``x_i ~ N(η_i / prec_i, 1 / prec_i)``. Interaction arrays are
    premasked at program construction (padded entries zeroed), so plain sums
    over the multiplicity axis are correct, exactly as in the discrete
    conditionals. Every head node must carry at least one
    :class:`QuadraticSelfInteraction` with positive ``diag`` — otherwise its
    conditional precision is zero (an improper flat conditional) and the draw
    produces non-finite values.
    """

    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        dtype = interaction_float_dtype(interactions)
        prec = jnp.zeros(output_sd.shape, dtype=dtype)
        eta = jnp.zeros(output_sd.shape, dtype=dtype)
        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, QuadraticSelfInteraction):
                prec += jnp.sum(interaction.diag, axis=-1)
                eta += jnp.sum(interaction.lin, axis=-1)
            elif isinstance(interaction, QuadraticPairInteraction):
                (x_tail,) = state
                eta -= jnp.sum(interaction.coupling * x_tail.astype(dtype), axis=-1)
            else:
                raise RuntimeError("Unsupported interaction found")
        return (prec, eta), sampler_state

    def sample_given_parameters(
        self,
        key: Key,
        parameters: PyTree,
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, None]:
        prec, eta = parameters
        noise = jax.random.normal(key, output_sd.shape, dtype=output_sd.dtype)
        mean = eta / prec
        std = jax.lax.rsqrt(prec)
        return (mean + std * noise).astype(output_sd.dtype), sampler_state


# Head/tail Block construction depends only on node/edge identities, never on
# parameter values, but ``factors`` is recomputed per ``with_beta`` copy (the
# β·params products must stay inside the tracer flow for AD). Cache the Blocks
# so repeated factor builds — one per chain per nrpt call — skip the O(|graph|)
# Python work. Same pattern as the Ising factor blocks.
_GAUSSIAN_FACTOR_BLOCK_CACHE: dict = {}


def _gaussian_factor_blocks(
    nodes, edges
) -> tuple[Block, "Block | None", "Block | None"]:
    def build():
        return (
            nodes,  # pin the id()-keyed objects (see _fifo_cache)
            edges,
            Block(list(nodes)),
            # Edge-less models (e.g. a diagonal-Gaussian reference for
            # AnnealedEBM) have no pair factor and hence no head/tail blocks.
            Block([a for a, _ in edges]) if len(edges) else None,
            Block([b for _, b in edges]) if len(edges) else None,
        )

    hit = _fifo_cache(_GAUSSIAN_FACTOR_BLOCK_CACHE, 64, (id(nodes), id(edges)), build)
    return hit[2], hit[3], hit[4]


class GaussianEBM(AbstractFactorizedEBM):
    r"""A Gaussian Markov random field:

    $$\mathcal{E}(x) = \beta \left( \tfrac{1}{2} \sum_i d_i x_i^2
        - \sum_i h_i x_i + \sum_{(i,j)} c_{ij} x_i x_j \right)$$

    i.e. ``x ~ N(P⁻¹h, (βP)⁻¹)`` with ``P_ii = d_i`` and ``P_ij = c_ij``.

    **Attributes:**

    - `nodes`: the :class:`~hamon.pgm.GaussianNode`\ s.
    - `edges`: ``(node, node)`` pairs carrying the off-diagonal couplings; each
      undirected pair appears once.
    - `diag`: per-node precision diagonal ``d`` (must make P positive definite;
      strict diagonal dominance suffices).
    - `lin`: per-node linear term ``h``.
    - `couplings`: per-edge off-diagonal precision ``c``.
    - `beta`: scalar inverse temperature.

    ``nodes`` and ``edges`` are identity-hashed sequences shared across
    ``with_beta`` copies, keeping the jit caches hitting (same convention as
    :class:`~hamon.models.IsingEBM`). See the module docstring for why
    ``proper_at_beta_zero`` is ``False``.
    """

    nodes: Sequence[AbstractNode]
    edges: Sequence
    diag: Array
    lin: Array
    couplings: Array
    beta: Array

    def __init__(self, nodes, edges, diag: Array, lin: Array, couplings: Array, beta):
        super().__init__({GaussianNode: jax.ShapeDtypeStruct((), jnp.float32)})
        self.nodes = _as_identity_seq(nodes)
        self.edges = _as_identity_seq(edges)
        param_dtype = jnp.result_type(diag, lin, couplings)
        self.diag = diag
        self.lin = lin
        self.couplings = couplings
        # β follows the parameter dtype so an x64 host app cannot promote the
        # β·params interaction tensors (and the whole device loop) to float64.
        self.beta = jnp.asarray(beta, dtype=param_dtype)

    @property
    def proper_at_beta_zero(self) -> bool:
        return False

    def with_beta(self, beta: Array) -> "GaussianEBM":
        return GaussianEBM(
            self.nodes, self.edges, self.diag, self.lin, self.couplings, beta
        )

    @property
    def factors(self) -> list[EBMFactor]:
        # β·params recomputed on every call — caching breaks AD tracer flow.
        self_block, head_block, tail_block = _gaussian_factor_blocks(
            self.nodes, self.edges
        )
        fs: list[EBMFactor] = [
            QuadraticSelfEBMFactor(
                self_block, self.beta * self.diag, self.beta * self.lin
            )
        ]
        if head_block is not None and tail_block is not None:
            fs.append(
                QuadraticPairEBMFactor(
                    [head_block, tail_block], self.beta * self.couplings
                )
            )
        return fs


class GaussianSamplingProgram(ModelSamplingProgram):
    """Thin wrapper specializing :class:`ModelSamplingProgram` to a GMRF."""

    def __init__(
        self,
        ebm: GaussianEBM,
        free_blocks: list,
        clamped_blocks: list[Block],
        *,
        _gibbs_spec: BlockGibbsSpec | None = None,
    ):
        super().__init__(
            ebm,
            free_blocks,
            clamped_blocks,
            GaussianGibbsConditional(),
            _gibbs_spec=_gibbs_spec,
        )


def _per_block_init(key, model, blocks, batch_shape, site_draw) -> list[Array]:
    """Shared skeleton of the continuous per-site init functions: map each
    block's nodes to model indices and draw each block independently via
    ``site_draw(key, idx, shape) -> float array``."""
    pos = {id(n): i for i, n in enumerate(model.nodes)}
    keys = jax.random.split(key, max(len(blocks), 1))
    return [
        site_draw(
            k,
            jnp.asarray([pos[id(n)] for n in block.nodes], dtype=jnp.int32),
            (*batch_shape, len(block.nodes)),
        ).astype(jnp.float32)
        for k, block in zip(keys, blocks)
    ]


def gaussian_init(
    key: Key[Array, ""],
    model: GaussianEBM,
    blocks: list[Block],
    batch_shape: tuple[int, ...] = (),
) -> list[Array]:
    """Draw an initial state from the *site-independent* part of the model.

    Samples each node from ``N(h_i/d_i, 1/(β·d_i))`` — the model with couplings
    ignored. The continuous counterpart of ``hinton_init``: not the target
    distribution, just a sensibly-scaled starting point for Gibbs/NRPT.
    Requires ``β > 0`` (the β = 0 member is improper; see
    ``GaussianEBM.proper_at_beta_zero``).
    """
    dtype = model.diag.dtype

    def site_draw(k, idx, shape):
        mean = (model.lin[idx] / model.diag[idx]).astype(dtype)
        std = jax.lax.rsqrt(model.beta * model.diag[idx]).astype(dtype)
        return mean + std * jax.random.normal(k, shape, dtype=jnp.float32)

    return _per_block_init(key, model, blocks, batch_shape, site_draw)
