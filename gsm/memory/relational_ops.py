# gsm/memory/relational_ops.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from gsm.memory.core import MemoryGraph, NodeType, EdgeType, NodeId
from gsm.memory.operators import (
    Operator,
    OperatorKind,
    OperatorSemanticInfo,
    OperatorTypeSignature,
    RoleType,
    ValueKind,
    DomainKind,
    OperatorEffectKind,
)
from gsm.memory.cues import Cue  # if you prefer loose typing, you can drop this import


def default_above_type_signature() -> OperatorTypeSignature:
    """
    Type signature for the relational operator Above(x, y):

      - Works in any domain that has an interpretable vertical axis
        (here we explicitly list GRID_2D, MUSIC_GRAPH, TREE to start).
      - Takes two NODE roles: x, y.
      - Effect kind: RELATION (not a transform or policy).
    """
    x_role = RoleType(
        name="x",
        value_kind=ValueKind.NODE,
        allowed_domains={DomainKind.GRID_2D, DomainKind.MUSIC_GRAPH, DomainKind.TREE},
        constraints={"needs_axis_attr": True},
    )
    y_role = RoleType(
        name="y",
        value_kind=ValueKind.NODE,
        allowed_domains={DomainKind.GRID_2D, DomainKind.MUSIC_GRAPH, DomainKind.TREE},
        constraints={"needs_axis_attr": True},
    )

    return OperatorTypeSignature.for_relation(
        domain_kinds=[DomainKind.GRID_2D, DomainKind.MUSIC_GRAPH, DomainKind.TREE],
        input_roles=[x_role, y_role],
        notes="x is 'higher' than y along some domain-specific axis.",
    )


def default_above_semantic() -> OperatorSemanticInfo:
    """
    Semantic metadata for Above(x, y).

    This is used by language/interpretability layers and for cross-domain
    reuse; it does not directly affect execution.
    """
    return OperatorSemanticInfo(
        roles=["x", "y"],
        invariants=[
            "identity_preserved",         # x, y themselves are unchanged
        ],
        effects=[
            "constrains_vertical_order",  # x 'higher' than y
        ],
        polymorphic_axes=[
            "vertical_axis",              # row index in grids, pitch in music, depth/position in trees
        ],
        description=(
            "Relational operator 'Above(x, y)': x is higher than y along a "
            "domain-specific vertical axis (e.g., row index in a grid)."
        ),
    )


class AboveOperator(Operator):
    """
    Concept-level relational operator: Above(x, y).

    This is a polymorphic *relation* that can in principle be instantiated
    in multiple domains. For now, we give it a concrete ARC/GRID_2D-flavored
    stub implementation for propose_bindings / apply_to_concept_graph.

    Convention for ARC grids:
      - Entities have a 'pos' attribute = (row, col), with row 0 at the top.
      - x is 'above' y if row_x < row_y.
    """

    def __post_init__(self) -> None:
        # Call base dataclass post-init if defined
        # (currently Operator doesn't define __post_init__, so this is safe.)
        # super().__post_init__()  # not needed today, but safe if added later

        # Ensure we have a type signature, semantic info, and lexical labels.
        if self.type_signature is None:
            self.type_signature = default_above_type_signature()

        if self.semantic is None:
            self.semantic = default_above_semantic()

        if not self.lexical_labels:
            # Default lexical surface forms for language grounding.
            self.lexical_labels = [
                "above",
                "over",
                "higher_than",
            ]

    # --- ARC / GRID_2D flavored stub implementations ---

    def propose_bindings(self, cue: Any, graph: MemoryGraph) -> List[Dict[str, Any]]:
        """
        ARC-flavored stub: propose candidate bindings (x, y) where x is above y.

        Strategy (simple for now):
          - Look for ENTITY nodes with a 'pos' attribute = (row, col).
          - For each ordered pair (x, y), if row_x < row_y, propose a binding.

        This is deliberately naive and does *not* use Jungnickel-style
        homomorphism yet; it just demonstrates how a relational operator
        can hook into the MemoryGraph for ARC-style scenes.
        """
        bindings: List[Dict[str, Any]] = []

        # 1. Collect ENTITY nodes with positions
        entities_with_pos: List[Tuple[NodeId, Tuple[int, int]]] = []
        for nid in graph.nodes_of_type(NodeType.ENTITY):
            pos = graph.get_attr(nid, "pos", None)
            if pos is None:
                continue
            if not (isinstance(pos, (tuple, list)) and len(pos) == 2):
                continue
            row, col = pos
            if not (isinstance(row, int) and isinstance(col, int)):
                continue
            entities_with_pos.append((nid, (row, col)))

        # 2. Propose pairs where row_x < row_y
        for x_id, (row_x, col_x) in entities_with_pos:
            for y_id, (row_y, col_y) in entities_with_pos:
                if x_id == y_id:
                    continue
                if row_x < row_y:
                    bindings.append({"x": x_id, "y": y_id})

        return bindings

    def apply_to_concept_graph(
        self,
        graph: MemoryGraph,
        bindings: Dict[str, Any],
    ) -> Any:
        """
        ARC-flavored stub: record the Above(x, y) relation in the MemoryGraph.

        Expected bindings:
          - 'x': MemoryGraph node id for the 'higher' entity
          - 'y': MemoryGraph node id for the 'lower' entity

        This does *not* modify core.Graph or ARC grids directly; it simply
        adds a semantic SPATIAL_RELATION edge with attrs["relation"] = "above".
        """
        x_id = bindings.get("x")
        y_id = bindings.get("y")
        if x_id is None or y_id is None:
            # In a more robust system, you'd raise or log here.
            return None

        # Optionally, sanity-check that these are ENTITY nodes with a pos attribute.
        try:
            x_type = graph.node_type(x_id)
            y_type = graph.node_type(y_id)
        except KeyError:
            return None

        if x_type != NodeType.ENTITY or y_type != NodeType.ENTITY:
            # Only record Above between entities in this stub.
            return None

        # Optional: check that x is actually above y, using 'pos'.
        pos_x = graph.get_attr(x_id, "pos", None)
        pos_y = graph.get_attr(y_id, "pos", None)
        if (
            isinstance(pos_x, (tuple, list)) and len(pos_x) == 2
            and isinstance(pos_y, (tuple, list)) and len(pos_y) == 2
        ):
            row_x, _ = pos_x
            row_y, _ = pos_y
            if isinstance(row_x, int) and isinstance(row_y, int):
                if not (row_x < row_y):
                    # If the binding contradicts the geometric meaning,
                    # we could choose to skip or still record it; here we skip.
                    return None

        graph.add_edge(
            src=x_id,
            dst=y_id,
            type_=EdgeType.SPATIAL_RELATION,
            attrs={"relation": "above"},
        )
        return None


def build_above_operator(operator_id: str = "op:rel:above") -> AboveOperator:
    """
    Convenience constructor for an AboveOperator with reasonable defaults.

    This can be used in tests, demos, or promotion logic that wants to
    create a canonical Above(x, y) operator.
    """
    # Minimal patterns for now; you can refine these later to describe
    # how Above appears in the MemoryGraph (e.g., ENTITY nodes with pos).
    from gsm.memory.patterns import ConceptPattern

    pattern = ConceptPattern()
    pattern.add_node("x", type=NodeType.ENTITY)
    pattern.add_node("y", type=NodeType.ENTITY)

    op = AboveOperator(
        id=operator_id,
        name="Above(x, y)",
        kind=OperatorKind.STATIC_TRANSFORM,  # conceptually a relation; kind is legacy
        input_pattern=pattern,
        output_pattern=None,
        semantic=default_above_semantic(),
        type_signature=default_above_type_signature(),
        lexical_labels=["above", "over", "higher_than"],
    )
    return op

