# gsm/operators/recenter.py
from __future__ import annotations

from typing import Any, Iterable, Tuple

from gsm.core.graph import Graph, NodeId
from gsm.operators.base import GraphOperator

from gsm.operators.grid import RecenterToOrigin  as GridRecenterToOrigin

class RecenterToOrigin(GridRecenterToOrigin):
    """
    ARC-style recenter operator.

    This is a thin subclass of the generic grid.RecenterToOrigin, with the same
    default behavior (operate on node ids as (row, col) coordinates).

    Recenter a grid-like Graph so that the minimum (row, col) coordinate
    among all 2D integer node ids becomes (0, 0).

    This is useful after applying translations or rotations that move
    nodes away from the origin, while graph_to_grid expects coords to
    start at (0, 0) for a tightly packed bounding box.

    Suitable for ARC-style GridGraphs created via domains.arcagi.convert.
    """

    def __init__(self, use_attr: str | None = None) -> None:
        super().__init__(use_attr=None)
