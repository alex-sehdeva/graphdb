from __future__ import annotations

from typing import Any, Tuple

from core.graph import Graph
from .base import GraphOperator

Coord = Tuple[int, int]


class Rotate90(GraphOperator):
    """
    Rotate grid coordinates 90 degrees counterclockwise around the origin.

    Node ids must be integer pairs (r, c). This does *not* automatically
    re-normalize into a bounding box; you may want to add a separate
    "recenter" operator if you care about keeping coords non-negative.
    """

    def _rotate_coord(self, coord: Any) -> Any:
        if (
            isinstance(coord, tuple)
            and len(coord) == 2
            and all(isinstance(x, int) for x in coord)
        ):
            r, c = coord
            # (r, c) -> (c, -r)
            return (c, -r)
        return coord

    def __call__(self, graph: Graph) -> Graph:
        new = Graph()
        id_map = {}
        for old_id, attrs in graph.nodes.items():
            new_id = self._rotate_coord(old_id)
            id_map[old_id] = new_id
            new.add_node(new_id, **attrs)

        for (u, v) in graph.edges:
            new.add_edge(id_map[u], id_map[v])

        return new

    def __repr__(self) -> str:
        return "Rotate90()"

