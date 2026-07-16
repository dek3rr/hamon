import abc
from collections.abc import Sequence
from typing import Self

import equinox as eqx
from jaxtyping import Array

from hamon.block_management import Block
from hamon.block_sampling import BlockGibbsSpec, BlockSamplingProgram
from hamon.conditional_samplers import AbstractConditionalSampler
from hamon.interaction import InteractionGroup


class AbstractFactor(eqx.Module):
    """A factor represents a batch of undirected interactions between sets of random variables.

    Concretely, this class implements a batch of factors defined over a bunch of parallel node groups. A single
    factor is defined over the nodes given by `node_groups[k][i]` for all `k` and a particular `i`. The defining trait of a
    factor is to produce InteractionGroups that affect each member of the factor in some way during the conditional
    updates of a block sampling program. As a user, you specify how this is done by implementing a
    concrete to_interaction_groups method for your child class.

    **Attributes:**

    - `node_groups`: the list of blocks that makes up this batch of factors.
    """

    node_groups: list[Block]

    def __init__(self, node_groups: list[Block]):
        """Create a batch of Factors.

        Practically, this just means writing down some parallel groups of nodes that the batch of Factors acts on.
        All of the functionality of the Factor is implemented by the method to_interaction_groups.

        **Arguments:**

        - `node_groups`: The node groups that this batch of factors acts on. A single Factor is defined
            over `node_groups[k][i]` for all values of `k` and a particular batch index `i`.
        """

        if not len(node_groups) > 0:
            raise RuntimeError("A factor should not be empty.")

        n_nodes = len(node_groups[0].nodes)

        for group in node_groups:
            if not len(group.nodes) == n_nodes:
                raise RuntimeError(
                    "Every block in node_groups must contain the same number of nodes."
                )

        self.node_groups = node_groups

    @abc.abstractmethod
    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Compile a factor to a set of directed interactions."""
        pass


class WeightedFactor(AbstractFactor):
    """A factor that is parameterized by a weight tensor.

    The leading dimension of the weights tensor must be the same length as the batch dimension of the factor (i.e
    the number of nodes in each of the node_groups).

    **Attributes:**

    - `weights`: the weight tensor.
    """

    weights: Array

    def __init__(self, weights: Array, node_groups: list[Block]):
        super().__init__(node_groups)
        if not weights.shape[0] == len(node_groups[0].nodes):
            raise RuntimeError(
                "The leading dimension of weights must have the same length as the number of nodes in each node group"
            )
        self.weights = weights


class FactorSamplingProgram(BlockSamplingProgram):
    """A sampling program built out of factors.

    This class simply breaks each factor passed to it down into interaction groups and uses them to build a
    BlockSamplingProgram.
    """

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        factors: Sequence[AbstractFactor],
        other_interaction_groups: list[InteractionGroup],
    ):
        """Create a FactorSamplingProgram. Thin wrapper over `BlockSamplingProgram`.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in gibbs_spec.
        - `factors`: The factors to use to build this sampling program.
        - `other_interaction_groups`: Other interaction groups to include in your program alongside what the
            factors produce.

        """

        all_interaction_groups = list(other_interaction_groups)
        for factor in factors:
            all_interaction_groups += factor.to_interaction_groups()

        super().__init__(gibbs_spec, samplers, all_interaction_groups)


class ModelSamplingProgram(FactorSamplingProgram):
    """Shared base for the per-model program wrappers (Ising, Gaussian, φ⁴…).

    Handles the boilerplate every model wrapper repeats: build the
    ``BlockGibbsSpec`` (or reuse a caller-supplied one — the spec is pure
    structure, no β or weights, so a rebuild over the same blocks would
    reproduce it node for node while paying an O(|nodes|) location-map
    construction), replicate one conditional per free block, and rebind new
    EBM weights via :meth:`with_ebm` without re-deriving the spec.
    """

    def __init__(
        self,
        ebm,
        free_blocks: list,
        clamped_blocks: list[Block],
        sampler: AbstractConditionalSampler,
        *,
        _gibbs_spec: BlockGibbsSpec | None = None,
    ):
        spec = (
            _gibbs_spec
            if _gibbs_spec is not None
            else BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)
        )
        super().__init__(spec, [sampler for _ in spec.free_blocks], ebm.factors, [])

    def _with_ebm_kwargs(self) -> dict:
        """Extra constructor kwargs :meth:`with_ebm` must forward (e.g. sampler
        configuration). Default: none."""
        return {}

    def with_ebm(self, ebm) -> Self:
        return type(self)(
            ebm,
            list(self.gibbs_spec.superblocks),
            self.gibbs_spec.clamped_blocks,
            _gibbs_spec=self.gibbs_spec,
            **self._with_ebm_kwargs(),
        )
