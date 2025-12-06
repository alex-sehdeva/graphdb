# gsm/operators/recolor.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from gsm.core.graph import Graph
from gsm.operators.base import GraphOperator
from gsm.operators.attribute_map import AttributeMap

GraphPair = Tuple[Graph, Graph]


class Recolor(AttributeMap):
    """
    ARC-style recoloring operator: remap node 'color' attributes.

    This is now a thin specialization of the generic AttributeMap operator,
    fixing attr_name="color".
    """

    def __init__(self, color_map: Dict[int, int]) -> None:
        super().__init__(attr_name="color", mapping=color_map)

    def __repr__(self) -> str:
        return f"Recolor({self.mapping!r})"


def infer_recolor_map(train_pairs: List[GraphPair]) -> Optional[Dict[int, int]]:
    """
    Infer a simple recoloring map from training graph pairs.

    Assumes:
      - Node ids correspond between input and output graphs.
      - Each node has an integer 'color' attribute in both graphs.
      - For each input color c_in, all occurrences are mapped consistently
        to a single output color c_out (otherwise we return None).

    Returns:
      - dict mapping c_in -> c_out if a consistent mapping is found
      - None if no nontrivial recolor is detected or if there is a conflict.
    """
    mapping: Dict[int, int] = {}

    for g_in, g_out in train_pairs:
        for nid, attrs_in in g_in.nodes.items():
            if nid not in g_out.nodes:
                continue
            attrs_out = g_out.nodes[nid]
            c_in = int(attrs_in.get("color", 0))
            c_out = int(attrs_out.get("color", 0))

            if c_in == c_out:
                continue
            if c_in in mapping and mapping[c_in] != c_out:
                return None
            mapping[c_in] = c_out

    return mapping or None


def recolor_ops_from_training(train_pairs: List[GraphPair]) -> List[Recolor]:
    """
    Infer one or zero Recolor operators from training pairs.

    - If a consistent recolor map exists, returns [Recolor(mapping)].
    - If not, returns [].
    """
    color_map = infer_recolor_map(train_pairs)
    if not color_map:
        return []
    return [Recolor(color_map)]

