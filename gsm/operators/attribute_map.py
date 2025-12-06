# gsm/operators/attribute_map.py  (new)

from __future__ import annotations
from gsm.core.graph import Graph
from gsm.operators.base import GraphOperator
from typing import Any, Dict


class AttributeMap(GraphOperator):
    """
    Generic attribute remapping operator.

    - attr_name: which node attribute to remap (e.g., "color", "symbol", "token")
    - mapping:   dict old_value -> new_value

    This covers:
      - ARC recoloring (attr_name="color")
      - symbolic rewrite rules (attr_name="symbol")
      - category relabeling, etc.
    """

    def __init__(self, attr_name: str, mapping: Dict[Any, Any]) -> None:
        self.attr_name = attr_name
        self.mapping = dict(mapping)

    def __call__(self, graph: Graph) -> Graph:
        new_g = Graph()
        # copy nodes with remapped attribute
        for nid, attrs in graph.nodes.items():
            attrs_copy = dict(attrs)
            if self.attr_name in attrs_copy:
                old = attrs_copy[self.attr_name]
                attrs_copy[self.attr_name] = self.mapping.get(old, old)
            new_g.add_node(nid, **attrs_copy)

        # copy edges unchanged
        for u, v in graph.edges:
            new_g.add_edge(u, v)
        return new_g

    def __repr__(self) -> str:
        return f"AttributeMap(attr_name={self.attr_name!r}, mapping={self.mapping!r})"


