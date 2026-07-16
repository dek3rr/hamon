"""Double-well (φ⁴) lattice models: continuous *multimodal* EBMs via
slice-sampling-within-Gibbs.

The Gaussian MRF (:mod:`hamon.models.gaussian`) covers the conjugate continuous
case; this module covers the case tempering exists for — continuous targets
with **basins**. The local potential is a double well,

$$U(x) = a\\,(x^2 - 1)^2 \\;=\\; a x^4 - 2a x^2 + a,$$

and with pairwise bilinear couplings the model is the lattice φ⁴ field theory:
at cold β the sites order into the ±1 wells and the joint distribution is
bimodal, so a single chain mode-collapses and NRPT round trips are what carry
± flips.

The single-site conditional ``∝ exp(−β[a x⁴ − 2a x² + (Σⱼ c_{ij} x_j − h) x])``
is a non-standard 1-D density, so the update is **slice sampling** (Neal 2003)
within Gibbs: stepping-out with Neal's *bounded* budget (the m-limited variant,
which preserves detailed balance exactly — an unbounded cap-and-clip would not)
followed by shrinkage. Slice sampling leaves the conditional invariant, so the
block-Gibbs sweep remains a valid MCMC kernel — this is the "embed MCMC methods
within this sampler" case the
:class:`~hamon.conditional_samplers.AbstractConditionalSampler` contract
explicitly allows, and the first hamon sampler that is a Markov *transition*
rather than an independent conditional draw. The current value x₀ it needs is
delivered through a **self-anchored interaction**: the polynomial factor lists
its own node group as the tail block, so each site receives its own pre-sweep
value as a tail state (pairing is positional; the graph coloring adds no
self-conflict for it).

Randomness inside the shrinkage loop is keyed by ``fold_in(key, iteration)``,
never drawn sequentially from a shared stream — a site's draws depend only on
its own key, position, and iteration, so another lane needing more iterations
(under ``vmap``, e.g. chain masking's padding chains) cannot perturb them.

Like every unbounded-state-space model, the β = 0 member is improper
(``proper_at_beta_zero = False``): use ladders with β_min > 0.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree

from hamon.block_management import Block, BlockSpec, from_global_state
from hamon.block_sampling import BlockGibbsSpec, _State
from hamon.conditional_samplers import AbstractConditionalSampler
from hamon.factor import ModelSamplingProgram
from hamon.interaction import InteractionGroup, interaction_float_dtype
from hamon.models.ebm import AbstractFactorizedEBM, EBMFactor
from hamon.models.gaussian import (
    QuadraticPairEBMFactor,
    QuadraticPairInteraction,
    QuadraticSelfInteraction,
    _gaussian_factor_blocks,
    _per_block_init,
)
from hamon.pgm import AbstractNode, GaussianNode, _as_identity_seq

__all__ = [
    "PolynomialSelfInteraction",
    "PolynomialSelfEBMFactor",
    "SliceGibbsConditional",
    "DoubleWellEBM",
    "DoubleWellSamplingProgram",
    "double_well_init",
]


class PolynomialSelfInteraction(eqx.Module):
    """Per-node quartic polynomial energy ``quart·x⁴ + quad·x² − lin·x``.

    All coefficients are energy coefficients with β already folded in (the
    factor convention shared with the Gaussian stack). Its interaction group is
    self-anchored (see the module docstring), delivering each site's own
    pre-sweep value alongside its coefficients.
    """

    quart: Array
    quad: Array
    lin: Array


class PolynomialSelfEBMFactor(EBMFactor):
    """Batch of single-node quartic terms ``quart·x⁴ + quad·x² − lin·x``."""

    quart: Array
    quad: Array
    lin: Array

    def __init__(self, node_group: Block, quart: Array, quad: Array, lin: Array):
        super().__init__([node_group])
        n = len(node_group.nodes)
        if quart.shape[0] != n or quad.shape[0] != n or lin.shape[0] != n:
            raise RuntimeError(
                "quart, quad and lin must have one entry per node in node_group."
            )
        self.quart = quart
        self.quad = quad
        self.lin = lin

    def to_interaction_groups(self) -> list[InteractionGroup]:
        # Self-anchored: the tail is the head group itself, delivering each
        # site's pre-sweep value as states[i] (positional pairing).
        return [
            InteractionGroup(
                PolynomialSelfInteraction(self.quart, self.quad, self.lin),
                self.node_groups[0],
                [self.node_groups[0]],
            )
        ]

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        (x,) = from_global_state(global_state, block_spec, self.node_groups)
        x2 = x * x
        return jnp.sum((self.quart * x2 + self.quad) * x2 - self.lin * x)


class SliceGibbsConditional(AbstractConditionalSampler):
    """Slice-sampling update for sites with quartic-polynomial conditionals.

    Accumulates the 1-D conditional energy ``A x⁴ + B x² + C x`` per site from
    :class:`PolynomialSelfInteraction` (A, B, −C·x contributions and the x₀
    anchor), :class:`QuadraticSelfInteraction` (``½·diag`` into B, ``−lin``
    into C) and :class:`QuadraticPairInteraction` (``coupling·x_tail`` into C),
    then runs one slice-sampling transition per site (Neal 2003): a slice level
    under log p(x₀), bounded stepping-out (budget ``max_stepout`` split
    uniformly between the two directions — the m-limited variant, exact by
    construction), then shrinkage until acceptance. The transition leaves the
    conditional invariant; sites in a color class update independently.

    **Attributes:**

    - `width`: the stepping-out interval width, in state units. The double
      well's modes sit at x ≈ ±1, so the default 2.0 spans them.
    - `max_stepout`: total stepping-out budget m (Neal's limited variant).

    Stateless (``sampler_state`` is ``None`` throughout), so it works under
    NRPT's round loop, which resets sampler state each round.
    """

    width: float = 2.0
    max_stepout: int = 8

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, None]:
        dtype = interaction_float_dtype(interactions)
        shape = output_sd.shape
        coef_a = jnp.zeros(shape, dtype=dtype)
        coef_b = jnp.zeros(shape, dtype=dtype)
        coef_c = jnp.zeros(shape, dtype=dtype)
        x0 = None
        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, PolynomialSelfInteraction):
                coef_a += jnp.sum(interaction.quart, axis=-1)
                coef_b += jnp.sum(interaction.quad, axis=-1)
                coef_c -= jnp.sum(interaction.lin, axis=-1)
                if x0 is None:
                    (anchor,) = state
                    x0 = anchor[..., 0].astype(dtype)
            elif isinstance(interaction, QuadraticSelfInteraction):
                coef_b += 0.5 * jnp.sum(interaction.diag, axis=-1)
                coef_c -= jnp.sum(interaction.lin, axis=-1)
            elif isinstance(interaction, QuadraticPairInteraction):
                (x_tail,) = state
                coef_c += jnp.sum(interaction.coupling * x_tail.astype(dtype), axis=-1)
            else:
                raise RuntimeError("Unsupported interaction found")
        if x0 is None:
            raise RuntimeError(
                "SliceGibbsConditional needs a PolynomialSelfInteraction on "
                "every sampled node (it carries the x0 anchor)."
            )

        def logp(x):
            x2 = x * x
            return -((coef_a * x2 + coef_b) * x2 + coef_c * x)

        w = jnp.asarray(self.width, dtype=dtype)
        k_level, k_pos, k_budget, k_shrink = jax.random.split(key, 4)

        # Slice level: log u + log p(x0), u ~ U(0,1).
        y = logp(x0) - jax.random.exponential(k_level, shape, dtype=dtype)

        # Initial interval of width w placed uniformly around x0.
        u = jax.random.uniform(k_pos, shape, dtype=dtype)
        left = x0 - w * u
        right = left + w
        # Neal's limited stepping-out: total budget m split uniformly between
        # the directions (J left, m-1-J right) — this randomized split is what
        # keeps the bounded variant exactly reversible.
        j_budget = jnp.floor(
            self.max_stepout * jax.random.uniform(k_budget, shape, dtype=dtype)
        )
        k_budget_r = (self.max_stepout - 1) - j_budget

        def expand(_, carry):
            left, right, j_left, j_right = carry
            grow_l = (j_left > 0) & (logp(left) > y)
            left = jnp.where(grow_l, left - w, left)
            j_left = jnp.where(grow_l, j_left - 1, j_left)
            grow_r = (j_right > 0) & (logp(right) > y)
            right = jnp.where(grow_r, right + w, right)
            j_right = jnp.where(grow_r, j_right - 1, j_right)
            return left, right, j_left, j_right

        left, right, _, _ = lax.fori_loop(
            0, self.max_stepout, expand, (left, right, j_budget, k_budget_r)
        )

        # Shrinkage: propose uniformly on [left, right]; on rejection shrink the
        # side of x0 the proposal fell on. Terminates a.s. (the interval shrinks
        # toward x0, where log p >= y by construction). Draws are keyed by
        # (k_shrink, iteration): no site's stream depends on how many
        # iterations other sites — or other vmap lanes — need.
        def cond(carry):
            return jnp.any(~carry[3])

        def body(carry):
            left, right, x, accepted, it = carry
            u = jax.random.uniform(jax.random.fold_in(k_shrink, it), shape, dtype=dtype)
            proposal = left + (right - left) * u
            ok = logp(proposal) >= y
            x = jnp.where(~accepted & ok, proposal, x)
            shrink = ~accepted & ~ok
            left = jnp.where(shrink & (proposal < x0), proposal, left)
            right = jnp.where(shrink & (proposal >= x0), proposal, right)
            return left, right, x, accepted | ok, it + 1

        accepted0 = jnp.zeros(shape, dtype=bool)
        _, _, x_new, _, _ = lax.while_loop(cond, body, (left, right, x0, accepted0, 0))
        return x_new.astype(output_sd.dtype), sampler_state


class DoubleWellEBM(AbstractFactorizedEBM):
    r"""The lattice φ⁴ model:

    $$\mathcal{E}(x) = \beta \left( \sum_i \left[ a_i x_i^4 - 2 a_i x_i^2
        - h_i x_i \right] + \sum_{(i,j)} c_{ij} x_i x_j \right)$$

    — the double well ``a(x²−1)²`` with its constant dropped (constants cancel
    in every energy difference), a per-site tilt ``h``, and pairwise bilinear
    couplings. With ferromagnetic couplings (c < 0) and cold β the target is
    **bimodal** (the ±1 ordered wells): the continuous case NRPT exists for.

    **Attributes:**

    - `nodes` / `edges`: :class:`~hamon.pgm.GaussianNode`\ s and coupled pairs.
    - `barrier`: per-node well coefficient ``a > 0`` (barrier height at x = 0).
    - `lin`: per-node tilt ``h``.
    - `couplings`: per-edge coefficient ``c``.
    - `beta`: scalar inverse temperature.

    See the module docstring for why ``proper_at_beta_zero`` is ``False``.
    """

    nodes: Sequence[AbstractNode]
    edges: Sequence
    barrier: Array
    lin: Array
    couplings: Array
    beta: Array

    def __init__(self, nodes, edges, barrier: Array, lin: Array, couplings, beta):
        super().__init__({GaussianNode: jax.ShapeDtypeStruct((), jnp.float32)})
        self.nodes = _as_identity_seq(nodes)
        self.edges = _as_identity_seq(edges)
        param_dtype = jnp.result_type(barrier, lin, couplings)
        self.barrier = barrier
        self.lin = lin
        self.couplings = couplings
        self.beta = jnp.asarray(beta, dtype=param_dtype)

    @property
    def proper_at_beta_zero(self) -> bool:
        return False

    def with_beta(self, beta: Array) -> "DoubleWellEBM":
        return DoubleWellEBM(
            self.nodes, self.edges, self.barrier, self.lin, self.couplings, beta
        )

    @property
    def factors(self) -> list[EBMFactor]:
        # β·params recomputed on every call — caching breaks AD tracer flow.
        self_block, head_block, tail_block = _gaussian_factor_blocks(
            self.nodes, self.edges
        )
        fs: list[EBMFactor] = [
            PolynomialSelfEBMFactor(
                self_block,
                self.beta * self.barrier,
                self.beta * (-2.0 * self.barrier),
                self.beta * self.lin,
            )
        ]
        if head_block is not None and tail_block is not None:
            fs.append(
                QuadraticPairEBMFactor(
                    [head_block, tail_block], self.beta * self.couplings
                )
            )
        return fs


class DoubleWellSamplingProgram(ModelSamplingProgram):
    """Thin wrapper specializing :class:`ModelSamplingProgram` to the φ⁴ model."""

    def __init__(
        self,
        ebm: DoubleWellEBM,
        free_blocks: list,
        clamped_blocks: list[Block],
        *,
        width: float = 2.0,
        max_stepout: int = 8,
        _gibbs_spec: BlockGibbsSpec | None = None,
    ):
        super().__init__(
            ebm,
            free_blocks,
            clamped_blocks,
            SliceGibbsConditional(width=width, max_stepout=max_stepout),
            _gibbs_spec=_gibbs_spec,
        )

    def _with_ebm_kwargs(self) -> dict:
        samp = self.samplers[0]
        assert isinstance(samp, SliceGibbsConditional)
        return {"width": samp.width, "max_stepout": samp.max_stepout}


def double_well_init(
    key: Key[Array, ""],
    model: DoubleWellEBM,
    blocks: list[Block],
    batch_shape: tuple[int, ...] = (),
) -> list[Array]:
    """Draw an initial state from the *decoupled* wells.

    Each site picks a well at random (±1) and adds Gaussian jitter at the
    well's local curvature scale ``U''(±1) = 8a``: ``x = s + N(0, 1/(β·8a))``.
    Not the target distribution — a sensibly-scaled start, like
    ``gaussian_init`` / ``hinton_init``. Requires β > 0.
    """

    def site_draw(k, idx, shape):
        k_sign, k_noise = jax.random.split(k)
        sign = jnp.where(
            jax.random.bernoulli(k_sign, 0.5, shape),
            jnp.float32(1.0),
            jnp.float32(-1.0),
        )
        std = jax.lax.rsqrt(model.beta * 8.0 * model.barrier[idx])
        return sign + std * jax.random.normal(k_noise, shape, dtype=jnp.float32)

    return _per_block_init(key, model, blocks, batch_shape, site_draw)
