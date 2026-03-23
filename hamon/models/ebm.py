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

    def with_beta(self, beta: Array) -> "AbstractEBM":
        """Return a copy of this EBM with a different inverse-temperature β.

        Subclasses that want to work with `nrpt_adaptive(ebm=..., program=...)`
        must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_beta(). "
            "Either implement it or provide explicit factory callables to nrpt_adaptive."
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

        energy = jnp.array(0.0)
        for factor in self.factors:
            energy += factor.energy(global_state, block_spec)
        return energy

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
