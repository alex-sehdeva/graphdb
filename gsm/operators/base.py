from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from gsm.core.graph import Graph


class GraphOperator(ABC):
    """
    Abstract base class for graph operators.

    A GraphOperator is a pure function Graph -> Graph that
    represents a structure-preserving or structured transformation
    (e.g., recolor, translate, rotate, etc.).

    Subclasses must implement __call__(self, graph: Graph) -> Graph.
    """

    @abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        raise NotImplementedError

    def then(self, other: "GraphOperator") -> "GraphOperator":
        """
        Return the composition (other ∘ self) as a new GraphOperator.

        Usage:
            f_then_g = f.then(g)
            result = f_then_g(G)  # same as g(f(G))
        """
        parent = self

        class Composed(GraphOperator):
            def __call__(self, g: Graph) -> Graph:
                return other(parent(g))

            def __repr__(self) -> str:
                return f"Composed({parent!r} → {other!r})"

        return Composed()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

