import abc
from collections.abc import Sequence
from dataclasses import dataclass, is_dataclass
from typing import ClassVar, TypeVar

import jax
from jax import numpy as jnp

_T = TypeVar("_T")


class _IdentitySeq(Sequence[_T]):
    """Immutable sequence that is a single opaque pytree leaf.

    Storing a large node or edge list directly on an ``eqx.Module`` makes
    every ``filter_jit`` call flatten it into one Python leaf per element and
    hash each element for the jit-cache lookup — O(|graph|) host work per
    call (measured: ~48K leaf visits and node ``__hash__`` calls per call on
    a 96×96 Ising model). This wrapper is not a registered pytree container,
    so the whole sequence is one leaf whose hash is computed once and cached.

    Equality/hash follow **element identity** — the same semantics the raw
    sequences had, since nodes are unique by construction (``_UniqueID``) and
    never compare equal across independently built models. Two wrappers over
    the same element objects (e.g. several ``IsingEBM``\\ s built from one
    ``nodes`` list, or ``with_beta`` passing the wrapper through) compare
    equal and share the jit cache; wrappers over different node objects
    differ, exactly as before.
    """

    __slots__ = ("_items", "_hash")

    def __init__(self, items):
        self._items = items if isinstance(items, tuple) else tuple(items)
        self._hash: int | None = None

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __hash__(self) -> int:
        h = self._hash
        if h is None:
            # id()-based: mirrors the unique-by-construction node semantics
            # without calling each element's Python-level __hash__. Edge
            # entries are (node, node) tuples, whose own ids are not stable
            # across rebuilds — canonicalize one level so freshly built
            # tuples over the same nodes hash (and compare) equal.
            h = self._hash = hash(
                tuple(
                    tuple(map(id, x)) if isinstance(x, tuple) else id(x)
                    for x in self._items
                )
            )
        return h

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, _IdentitySeq):
            return NotImplemented
        # Tuple comparison short-circuits on identical elements (C-level
        # pointer check), so the same-nodes case never runs Python __eq__.
        return self._items == other._items

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self._items)!r})"


class _CounterMeta(abc.ABCMeta):
    """Metaclass that automatically calls __post_init__ and provides unique ordering.

    Used internally by hamon for node identification and ordering.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not is_dataclass(cls):
            post_init = getattr(instance, "__post_init__", None)
            if callable(post_init):
                post_init()
        return instance

    def __lt__(cls, other):
        # Order node *types* by (module, qualname): a deterministic,
        # process-stable key (unlike id()), so any sort of node types is
        # reproducible across runs. It is unique for normally-defined classes;
        # two distinct classes sharing a module+qualname (e.g. dynamically
        # generated) would order ambiguously, but hamon never produces such.
        if not isinstance(other, type):
            raise NotImplementedError
        return (cls.__module__, cls.__qualname__) < (
            other.__module__,
            other.__qualname__,
        )


class _UniqueID(metaclass=_CounterMeta):
    """
    This is a way of ensuring that there is a unique identifier
    for subclasses, without them being required to call super().__init__().
    """

    __slots__ = ("_hash",)
    _counter: ClassVar[int] = 0
    _hash: int

    def __post_init__(self):
        self._hash = _UniqueID._counter
        _UniqueID._counter += 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _UniqueID):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other):
        if isinstance(other, _UniqueID):
            return self._hash < other._hash
        raise RuntimeError("less than only defined between _UniqueIDs")


@dataclass(eq=False)
class AbstractNode(_UniqueID):
    """
    A node in a PGM.

    Every node used in a PGM must inherit from this class. When compiling a program, each node is assigned a
    shape and datatype that are used to organize the state of the sampling program in a jax-friendly way.
    """

    def __new__(cls, *args, **kwargs):
        if cls is AbstractNode:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)


class SpinNode(AbstractNode):
    """A node that represents a random variable that takes on a state in {-1, 1}."""

    pass


class CategoricalNode(AbstractNode):
    """A node that represents a random variable that may take on any one of K possible discrete states,
    represented by an integer in [0, K).

    The default state dtype is ``uint8`` (see ``DEFAULT_NODE_SHAPE_DTYPES``),
    which caps K at 256; pass a custom ``node_shape_dtypes`` mapping with a
    wider integer dtype for larger category counts."""

    pass


class GaussianNode(AbstractNode):
    """A node that represents a continuous random variable with state in ℝ.

    The default state dtype is ``float32`` (see ``DEFAULT_NODE_SHAPE_DTYPES``).
    Continuous state spaces are unbounded, so — unlike the discrete nodes —
    there is no proper uniform distribution over them: models built on these
    nodes cannot be tempered to β = 0 (see
    ``AbstractEBM.proper_at_beta_zero``); use a β ladder with β_min > 0.
    """

    pass


DEFAULT_NODE_SHAPE_DTYPES = {
    SpinNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.bool_),
    CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.uint8),
    GaussianNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.float32),
}
