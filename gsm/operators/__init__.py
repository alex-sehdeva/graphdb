"""
Operator algebra for the Graph Substrate Machine (gsm).

This package defines:
- GraphOperator: an abstract base for all operators
- Concrete operators such as:
    - Recolor: change node 'color' attributes
    - Translation: shift coordinate-based node ids

Most operators are designed to work with ARC-style grids
(nodes = (row, col), attrs include 'color'), but the base
class is domain-agnostic and can be reused in other domains.
"""

# gsm/operators/__init__.py
from gsm.operators.base import GraphOperator
from gsm.operators.attribute_map import AttributeMap
from gsm.operators.grid import Translation as GridTranslation, RecenterToOrigin as GridRecenterToOrigin
from gsm.operators.recolor import Recolor
from gsm.operators.translation import Translation  # ARC wrapper
from gsm.operators.recenter import RecenterToOrigin  # <-- add this

__all__ = [
    "GraphOperator",
    "AttributeMap",
    "Translation",
    "GridTranslation",
    "RecenterToOrigin",
    "GridRecenterToOrigin",
    "Recolor",
]
