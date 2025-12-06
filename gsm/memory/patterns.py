# gsm/memory/patterns.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .core import NodeType, EdgeType


@dataclass
class NodeConstraint:
    """
    Constraint on a MemoryGraph node used in a ConceptPattern.

    Examples:
      - type=NodeType.ENTITY, attrs={"task_id": "ARC-1234"}
      - type=NodeType.FEATURE, attrs={"feature_type": "color", "value": 2}
    """
    type: Optional[NodeType] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeConstraint:
    """
    Constraint on an edge in a ConceptPattern.

    The src_label and dst_label are variable names from the
    ConceptPattern.node_constraints dict (e.g. "obj", "schema").
    """
    type: Optional[EdgeType] = None
    src_label: Optional[str] = None
    dst_label: Optional[str] = None


@dataclass
class ConceptPattern:
    """
    A small pattern graph over labeled variables.

    This is deliberately lightweight: it doesn't know anything about
    gsm.core.Graph; it only speaks in terms of MemoryGraph nodes and
    semantic NodeType / EdgeType constraints.

    A matcher (to be implemented later) can use this to bind labels
    to concrete MemoryGraph node ids.
    """
    node_constraints: Dict[str, NodeConstraint] = field(default_factory=dict)
    edge_constraints: List[EdgeConstraint] = field(default_factory=list)

    def described_labels(self) -> Set[str]:
        """
        Return the set of variable labels mentioned in this pattern.
        """
        return set(self.node_constraints.keys())

    def add_node(
        self,
        label: str,
        type: Optional[NodeType] = None,
        **attrs: Any,
    ) -> None:
        """
        Convenience helper to add a node constraint.

        Example:
            pattern.add_node("obj", type=NodeType.ENTITY, role="player")
        """
        self.node_constraints[label] = NodeConstraint(type=type, attrs=dict(attrs))

    def add_edge(
        self,
        src_label: str,
        dst_label: str,
        type: Optional[EdgeType] = None,
    ) -> None:
        """
        Convenience helper to add an edge constraint.

        Example:
            pattern.add_edge("obj", "color", type=EdgeType.HAS_FEATURE)
        """
        self.edge_constraints.append(
            EdgeConstraint(type=type, src_label=src_label, dst_label=dst_label)
        )
