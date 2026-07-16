# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from hamon.block_management import Block, BlockSpec, block_state_to_global
from hamon.block_sampling import _SD, _State
from hamon.factor import AbstractFactor
from hamon.pgm import DEFAULT_NODE_SHAPE_DTYPES


class AbstractEBM(eqx.Module):
    """
    Something that has a well-defined energy function (map from a state to a scalar).
    """

    @property
    def proper_at_beta_zero(self) -> bool:
        """Whether the β = 0 member of this EBM's tempered family is a proper
        distribution.

        Finite state spaces always are (β = 0 is the uniform distribution), so
        the default is ``True``. Continuous/unbounded state spaces are not —
        there is no uniform distribution over ℝⁿ, and e.g. a Gaussian
        conditional's variance 1/(β·P) diverges as β → 0 — so continuous EBMs
        override this to ``False`` and NRPT refuses a ladder that starts at
        exactly β = 0 (use β_min > 0, or anneal from a proper reference with
        :class:`AnnealedEBM`).
        """
        return True

    @property
    def beta_affine(self) -> bool:
        """Whether this EBM's energy is *affine* rather than *linear* in β:
        ``E_β = E₀ + β·(E₁ − E₀)`` with ``E₀ ≠ 0``.

        ``False`` (the default) is the pure temperature-linear family
        ``E_β = β·E_base`` that NRPT's template mode scales by β. ``True``
        (:class:`AnnealedEBM`) selects the affine template path: the kernel
        interpolates interactions as ``offset + β·slope`` and swap energies use
        ``Δ = E₁ − E₀`` (the β-independent ``E₀`` cancels exactly in every swap
        ratio, so the DEO math is otherwise unchanged).
        """
        return False

    def with_beta(self, beta: Array) -> "AbstractEBM":
        """Return a copy of this EBM with a different inverse-temperature β.

        Subclasses that want to work with `tune_schedule(ebm=..., program=...)`
        must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_beta(). "
            "Either implement it or provide explicit factory callables to tune_schedule."
        )

    @abc.abstractmethod
    def energy(
        self, state: list[_State], blocks: "BlockSpec | list[Block]"
    ) -> Float[Array, ""]:
        """Evaluate the energy function of the EBM given some state information.

        **Arguments:**

        - `state`: The state for which to evaluate the energy function. Must be compatible with `blocks`.
        - `blocks`: Specifies how the information in `state` is organized. May be either a pre-built
            `BlockSpec` (fast path — avoids rebuilding the spec) or a plain `list[Block]` for
            convenience when calling from user code.

        **Returns:**

        A scalar representing the energy value associated with `state`.
        """
        raise NotImplementedError


class EBMFactor(AbstractFactor):
    """A factor that defines an energy function."""

    @abc.abstractmethod
    def energy(
        self, global_state: list[Array], block_spec: BlockSpec
    ) -> Float[Array, ""]:
        """Evaluate the energy function of the factor.

        **Arguments:**

        - `global_state`: The state information to use to evaluate the energy function.
            Is a global state of `block_spec`.
        - `block_spec`: The `BlockSpec` used to generate `global_state`.
        """
        raise NotImplementedError


class AbstractFactorizedEBM(AbstractEBM):
    r"""An EBM that is made up of Factors, i.e., an EBM with an energy function like,

    $$\mathcal{E}(x) = \sum_i \mathcal{E}^i(x)$$

    where the sum over $i$ is taken over factors.

    Child classes must define a property which returns a list of
    factors that substantiate the EBM.

    **Attributes:**

    - `node_shape_dtypes`: the shape/dtypes of the nodes involved in this EBM. Used to generate the BlockSpec that
        defines the global state that factors receive to compute energy.
    """

    node_shape_dtypes: _SD

    def __init__(self, node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES):
        self.node_shape_dtypes = node_shape_dtypes

    def energy(
        self, state: list[_State], blocks: "BlockSpec | list[Block]"
    ) -> Float[Array, ""]:
        """Evaluate the total energy as the sum of all factor energies."""
        if isinstance(blocks, BlockSpec):
            block_spec = blocks
        else:
            block_spec = BlockSpec(blocks, self.node_shape_dtypes)

        global_state = block_state_to_global(state, block_spec)

        # Accumulate in the factors' own dtype: seeding with jnp.array(0.0)
        # would promote float32 factor energies to float64 under x64.
        energy = None
        for factor in self.factors:
            factor_energy = factor.energy(global_state, block_spec)
            energy = factor_energy if energy is None else energy + factor_energy
        return jnp.array(0.0) if energy is None else energy

    @property
    @abc.abstractmethod
    def factors(self) -> list[EBMFactor]:
        """The factors that define this EBM."""
        raise NotImplementedError


class FactorizedEBM(AbstractFactorizedEBM):
    """An EBM that is defined by a concrete list of factors.

    **Attributes:**

    - `_factors`: the list of factors that defines this EBM.
    """

    _factors: list[EBMFactor]

    def __init__(
        self,
        factors: list[EBMFactor],
        node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES,
    ):
        super().__init__(node_shape_dtypes)
        self._factors = factors

    @property
    def factors(self):
        return self._factors


class AnnealedEBM(AbstractFactorizedEBM):
    r"""The reference-annealing path between two EBMs over the same nodes:

    $$\mathcal{E}_\beta(x) = (1-\beta)\,\mathcal{E}_{\text{ref}}(x)
        + \beta\,\mathcal{E}_{\text{target}}(x)
        = \mathcal{E}_{\text{ref}} + \beta\,(\mathcal{E}_{\text{target}}
        - \mathcal{E}_{\text{ref}})$$

    — the standard PT path from a *reference* distribution (β = 0) to the
    *target* (β = 1), rather than from the flat/uniform member of the target's
    own tempered family. Its point: an unbounded-state-space target has no
    proper β = 0 member (``proper_at_beta_zero = False``), but annealing from a
    proper reference — e.g. a diagonal Gaussian — makes **every** rung of the
    ladder proper, so β can start at exactly 0 and the ladder covers the full
    entropic path.

    Both EBMs must be defined over the same nodes and be temperature-linear
    themselves (``factors`` emit β-scaled coefficients, as all hamon models
    do): the annealed factors are simply the reference's at β' = 1−β plus the
    target's at β' = β. The sampling program must use a conditional that
    understands the union of both factor families (e.g.
    :class:`~hamon.models.SliceGibbsConditional` handles quartic + quadratic;
    an annealed pair of Gaussians is handled by
    :class:`~hamon.models.GaussianGibbsConditional`).

    ``beta_affine`` is ``True``: NRPT's template mode interpolates interactions
    affinely and computes swap energies as Δ = E_target − E_ref (the shared
    E_ref cancels in every swap ratio).
    """

    reference: "AbstractFactorizedEBM"
    target: "AbstractFactorizedEBM"
    beta: Array

    def __init__(
        self,
        reference: "AbstractFactorizedEBM",
        target: "AbstractFactorizedEBM",
        beta,
    ):
        ref_sd = getattr(reference, "node_shape_dtypes", {})
        tgt_sd = getattr(target, "node_shape_dtypes", {})
        super().__init__({**ref_sd, **tgt_sd})
        self.reference = reference
        self.target = target
        self.beta = jnp.asarray(beta)

    @property
    def proper_at_beta_zero(self) -> bool:
        # The annealed family's β=0 member is the reference at FULL weight —
        # proper whenever exp(−E_ref) is normalizable. Deliberately NOT
        # inherited from reference.proper_at_beta_zero, which asks about the
        # reference's own β→0 limit, a rung this ladder never visits.
        return True

    @property
    def beta_affine(self) -> bool:
        return True

    def with_beta(self, beta: Array) -> "AnnealedEBM":
        return AnnealedEBM(self.reference, self.target, beta)

    @property
    def factors(self) -> list[EBMFactor]:
        one = jnp.asarray(1.0, dtype=self.beta.dtype)
        ref = self.reference.with_beta(one - self.beta)
        tgt = self.target.with_beta(self.beta)
        assert isinstance(ref, AbstractFactorizedEBM)
        assert isinstance(tgt, AbstractFactorizedEBM)
        return list(ref.factors) + list(tgt.factors)
