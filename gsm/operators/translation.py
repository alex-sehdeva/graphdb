# gsm/operators/translation.py
from __future__ import annotations

from gsm.operators.grid import Translation as GridTranslation


class Translation(GridTranslation):
    """
    ARC-style translation operator.

    This is a thin subclass of the generic grid.Translation, with the same
    default behavior (operate on node ids as (row, col) coordinates).

    - Node ids are assumed to be 2D integer coordinates: (r, c).
    - Attributes are preserved.
    - Edges are lifted to connect the translated coordinates.

    Suitable for ARC-style GridGraphs created via domains.arcagi.convert.
    """

    def __init__(self, dx: int, dy: int) -> None:
        # Default: operate on node ids (use_attr=None)
        super().__init__(dx=dx, dy=dy, use_attr=None)

