# gsm/operators/grid.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from gsm.core.graph import Graph, NodeId
from gsm.operators.base import GraphOperator


Coord = Tuple[int, int]


def _is_int_coord(nid: Any) -> bool:
    return (
        isinstance(nid, tuple)
        and len(nid) == 2
        and all(isinstance(x, int) for x in nid)
    )


class Translation(GraphOperator):
    """
    Translate nodes in a grid-like Graph.

    Two usage modes:

      1) Default (use_attr=None):
            - Treat node ids as integer coords: (row, col).
            - Build a new graph with translated node ids.
         Suitable for ARC-style GridGraphs.

      2) Attribute mode (use_attr="pos"):
            - Treat nodes as stable entities.
            - Only update the node attribute 'pos' = (row, col).
            - Does not change node ids.
         Suitable for domains that keep entity ids stable and store
         position in an attribute.
    """

    def __init__(self, dx: int, dy: int, use_attr: str | None = None) -> None:
        self.dx = dx
        self.dy = dy
        self.use_attr = use_attr

    def _translate_coord(self, coord: Any) -> Any:
        if (
            isinstance(coord, tuple)
            and len(coord) == 2
            and all(isinstance(x, int) for x in coord)
        ):
            r, c = coord
            return (r + self.dy, c + self.dx)
        return coord

    def __call__(self, graph: Graph) -> Graph:
        if self.use_attr is None:
            # Translate node ids directly (ARC / board games style).
            new_g = Graph()
            id_map: Dict[NodeId, NodeId] = {}

            for nid, attrs in graph.nodes.items():
                new_nid: NodeId = nid
                if _is_int_coord(nid):
                    new_nid = self._translate_coord(nid)
                id_map[nid] = new_nid
                new_g.add_node(new_nid, **attrs)

            for u, v in graph.edges:
                new_u = id_map.get(u, u)
                new_v = id_map.get(v, v)
                new_g.add_edge(new_u, new_v)

            return new_g
        else:
            # Translate an attribute (e.g., "pos") without changing node ids.
            new_g = Graph()
            for nid, attrs in graph.nodes.items():
                attrs_copy = dict(attrs)
                if self.use_attr in attrs_copy:
                    attrs_copy[self.use_attr] = self._translate_coord(
                        attrs_copy[self.use_attr]
                    )
                new_g.add_node(nid, **attrs_copy)

            for u, v in graph.edges:
                new_g.add_edge(u, v)

            return new_g

    def __repr__(self) -> str:
        mode = "ids" if self.use_attr is None else f"attr={self.use_attr}"
        return f"Translation(dx={self.dx}, dy={self.dy}, mode={mode})"


class RecenterToOrigin(GraphOperator):
    """
    Recenter a grid-like Graph so that the minimum (row, col) coordinate
    among all integer coordinate nodes becomes (0, 0).

    - If use_attr is None:
        - Shift node ids directly.
    - If use_attr is not None:
        - Shift an attribute (e.g. "pos") and leave node ids unchanged.

    This is useful after applying translations or rotations that move
    nodes away from the origin, while your grid encoder/decoder expects
    coords to start at (0, 0) for a tight bounding box.
    """

    def __init__(self, use_attr: str | None = None) -> None:
        self.use_attr = use_attr

    def __call__(self, graph: Graph) -> Graph:
        if self.use_attr is None:
            # Work directly on node ids.
            coords = [nid for nid in graph.nodes.keys() if _is_int_coord(nid)]
        else:
            # Work on an attribute, not the ids.
            coords = []
            for nid, attrs in graph.nodes.items():
                val = attrs.get(self.use_attr)
                if _is_int_coord(val):
                    coords.append(val)

        if not coords:
            return graph

        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)

        # Already at origin?
        if min_r == 0 and min_c == 0:
            return graph

        new_g = Graph()

        if self.use_attr is None:
            # Shift ids.
            id_map: Dict[NodeId, NodeId] = {}
            for nid, attrs in graph.nodes.items():
                new_nid: NodeId
                if _is_int_coord(nid):
                    r, c = nid
                    new_nid = (r - min_r, c - min_c)
                else:
                    new_nid = nid
                id_map[nid] = new_nid
                new_g.add_node(new_nid, **attrs)

            for u, v in graph.edges:
                new_u = id_map.get(u, u)
                new_v = id_map.get(v, v)
                new_g.add_edge(new_u, new_v)
        else:
            # Shift attribute only.
            for nid, attrs in graph.nodes.items():
                attrs_copy = dict(attrs)
                val = attrs_copy.get(self.use_attr)
                if _is_int_coord(val):
                    r, c = val
                    attrs_copy[self.use_attr] = (r - min_r, c - min_c)
                new_g.add_node(nid, **attrs_copy)

            for u, v in graph.edges:
                new_g.add_edge(u, v)

        return new_g

    def __repr__(self) -> str:
        mode = "ids" if self.use_attr is None else f"attr={self.use_attr}"
        return f"RecenterToOrigin(mode={mode})"

