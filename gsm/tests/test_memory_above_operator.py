# tests/test_memory_above_operator.py

from __future__ import annotations

from typing import Dict, Any

from gsm.memory.nx_backend import NxMemoryGraph
from gsm.memory.core import NodeType, EdgeType
from gsm.memory.relational_ops import build_above_operator, default_above_type_signature
from gsm.memory.operators import OperatorTypeSignature, DomainKind, ValueKind


class DummyCue:
    """Minimal stand-in for gsm.memory.cues.Cue for this test."""
    def __init__(self) -> None:
        self.extra: Dict[str, Any] = {}


def test_above_operator_signature_and_labels() -> None:
    op = build_above_operator(operator_id="op:test:above")

    # Type signature should be a RELATION with two node roles.
    sig: OperatorTypeSignature = op.type_signature  # type: ignore
    assert sig is not None
    assert len(sig.input_roles) == 2
    assert {r.name for r in sig.input_roles} == {"x", "y"}

    # Both roles should be NODE kind.
    assert all(r.value_kind == ValueKind.NODE for r in sig.input_roles)

    # Domain kinds should include GRID_2D.
    assert DomainKind.GRID_2D in sig.domain_kinds

    # Lexical labels should include "above".
    assert "above" in op.lexical_labels


def test_above_operator_propose_bindings_arc_like() -> None:
    mem = NxMemoryGraph()
    op = build_above_operator(operator_id="op:test:above")

    # Create two ENTITY nodes with 'pos' attributes:
    # x at row 0, y at row 2 -> x is above y.
    x_id = "entity:x"
    y_id = "entity:y"

    mem.add_node(
        node_id=x_id,
        type_=NodeType.ENTITY,
        embedding=[],
        attrs={"pos": (0, 1)},  # row 0, col 1
    )
    mem.add_node(
        node_id=y_id,
        type_=NodeType.ENTITY,
        embedding=[],
        attrs={"pos": (2, 1)},  # row 2, col 1
    )

    cue = DummyCue()
    bindings_list = op.propose_bindings(cue, mem)

    # We expect at least one binding where x is the upper entity and y is the lower.
    assert any(
        b.get("x") == x_id and b.get("y") == y_id
        for b in bindings_list
    ), f"Expected binding (x={x_id}, y={y_id}) in {bindings_list}"


def test_above_operator_apply_to_concept_graph_creates_edge() -> None:
    mem = NxMemoryGraph()
    op = build_above_operator(operator_id="op:test:above")

    x_id = "entity:x"
    y_id = "entity:y"

    mem.add_node(
        node_id=x_id,
        type_=NodeType.ENTITY,
        embedding=[],
        attrs={"pos": (0, 0)},
    )
    mem.add_node(
        node_id=y_id,
        type_=NodeType.ENTITY,
        embedding=[],
        attrs={"pos": (1, 0)},
    )

    bindings = {"x": x_id, "y": y_id}
    op.apply_to_concept_graph(mem, bindings)

    # Check that an edge x -> y with SPATIAL_RELATION and relation="above" exists.
    edges = list(mem.edges_from(x_id))
    spatial_edges = [
        (src, dst, etype, attrs)
        for (src, dst, etype), attrs in (
            ((src, dst, etype), mem._g[src][dst][k])  # type: ignore[attr-defined]
            for src, dst, k in mem._g.edges(keys=True)  # direct access to underlying MultiDiGraph
            if src == x_id
        )
    ]

    # Simpler / more robust version using MemoryGraph API if you don't want to touch mem._g:
    spatial_edges = []
    for src, dst, etype in mem.edges_from(x_id):
        if etype != EdgeType.SPATIAL_RELATION:
            continue
        attrs = mem._g[src][dst][0]  # or store attrs in MemoryGraph API if you prefer
        spatial_edges.append((src, dst, etype, attrs))

    assert any(
        dst == y_id and etype == EdgeType.SPATIAL_RELATION and attrs.get("relation") == "above"
        for (src, dst, etype, attrs) in spatial_edges
    ), f"Expected SPATIAL_RELATION edge x->y with relation='above'; got {spatial_edges}"

