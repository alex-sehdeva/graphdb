# gsm/memory/operators.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gsm.memory.core import NodeId, NodeType, OperatorKind, MemoryGraph
from gsm.memory.patterns import ConceptPattern

@dataclass
class OperatorStats:
    """
    Simple usage / value statistics for a concept-level Operator.

    These stats live inside the Operator object, not the MemoryGraph
    node. You can also copy some of them into MemoryGraph.node_stats
    if you want retrieval to use them without pulling the Operator object.
    """
    # lifecycle
    created_at: int = 0
    last_used_at: int = 0

    # usage
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # value
    avg_reward: float = 0.0

    # immune-ish
    strength: float = 1.0     # clonal level

    @property
    def precision(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    def to_stats_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_reward": self.avg_reward,
            "strength": self.strength,
            "precision": self.precision,
        }

    def update_success(self, reward: float, now: int) -> None:
        self.use_count += 1
        self.success_count += 1
        self.last_used_at = now
        alpha = 0.1
        self.avg_reward = (1 - alpha) * self.avg_reward + alpha * reward
        self.strength *= 1.05   # clonal expansion

    def update_failure(self, now: int) -> None:
        self.use_count += 1
        self.failure_count += 1
        self.last_used_at = now
        self.strength *= 0.95   # negative selection


class Operator:
    """
    Concept-level operator living in the MemoryGraph.

    This is a higher-level object than gsm.operators.GraphOperator:

      - It knows *where* it tends to apply (input_pattern).
      - It knows *what kind of effect* it has (output_pattern).
      - It carries statistics about reward / usage.
      - It may wrap or refer to a GraphOperator / RuleProgram.

    For ARC:
      - kind = STATIC_TRANSFORM
      - underlying might be a GraphOperator or RuleProgram.

    For ARC-AGI 3:
      - kind = DYNAMICS / POLICY
      - associated_actions may list ACTION / OPTION node ids.
    """

    id: str
    name: str
    kind: OperatorKind

    input_pattern: ConceptPattern
    output_pattern: Optional[ConceptPattern] = None

    # For ARC-AGI 3 / video game style:
    associated_actions: List[NodeId] = field(default_factory=list)

    # Free-form parameters, e.g. {"dx": 1, "dy": 0}, {"color_map": {...}}
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Stats + last retrieval score
    stats: OperatorStats = field(default_factory=OperatorStats)
    last_applicability_score: float = 0.0

    # Optional: link back to a concrete GraphOperator or RuleProgram
    # We keep it typed as Any here to avoid import cycles; you can
    # specialize it later if you want tighter typing.
    underlying: Any = None

    # -------- scoring / matching hooks --------

    def score_applicability(self, cue: Any, graph: MemoryGraph) -> float:
        """
        Heuristic scalar score: how applicable is this operator
        to the current cue?

        This default implementation only uses recency and avg_reward,
        which is enough to get retrieval bootstrapped. Later you can
        plug in:
          - pattern match quality
          - context similarity (task_id, schemas)
          - embedding similarity, etc.
        """
        s = self.stats
        recency_term = 0.0
        timestep = getattr(cue, "extra", {}).get("timestep")
        if timestep is not None:
            delta = max(0, timestep - s.last_used_at)
            recency_term = 1.0 / (1.0 + delta)

        score = recency_term + s.avg_reward
        self.last_applicability_score = score
        return score

    def propose_bindings(self, cue: Any, graph: MemoryGraph) -> List[Dict[str, Any]]:
        """
        Try to bind the input_pattern into the current situation.

        Returns:
          - a list of 'bindings', each a dict mapping pattern labels
            to concrete MemoryGraph node ids, coordinates, etc.

        This is intentionally left as a stub; later we can connect it
        to your Jungnickel-style homomorphism / pattern-matching machinery.
        """
        # TODO: implement real pattern matching.
        return []

    def apply_to_concept_graph(
        self,
        graph: MemoryGraph,
        bindings: Dict[str, Any],
    ) -> Any:
        """
        Apply this operator at the concept level, given concrete bindings.

        For STATIC_TRANSFORM:
          - may call a gsm.operators.GraphOperator / RuleProgram
            via self.underlying, using bindings to pick the right objects.

        For DYNAMICS:
          - may predict next-state pattern.

        For POLICY:
          - may map the current state to an ACTION / OPTION choice.

        For now, this is a placeholder; you'll wire it into your
        actual operator algebra as you go.
        """
        raise NotImplementedError("Operator.apply_to_concept_graph is not implemented yet")

    def record_success(self, mem: MemoryGraph, node_id: NodeId, reward: float, now: int) -> None:
        self.stats.update_success(reward=reward, now=now)
        mem.update_stats(node_id, self.stats.to_stats_dict())

    def record_failure(self, mem: MemoryGraph, node_id: NodeId, now: int) -> None:
        self.stats.update_failure(now=now)
        mem.update_stats(node_id, self.stats.to_stats_dict())


def add_operator_node(
    mem: MemoryGraph,
    operator: Operator,
    embedding: Optional[List[float]] = None,
    signature: Any = None,
) -> NodeId:
    """
    Insert an OPERATOR node into the MemoryGraph and attach the Operator object.

    If signature is provided, it is stored in the node attrs under 'signature'
    so that promotion/merge logic can identify structurally equivalent operators.

    Returns:
        node_id: the MemoryGraph node id for this operator.
    """
    node_id: NodeId = f"op:{operator.id}"

    attrs: Dict[str, Any] = {
        "name": operator.name,
        "operator_obj": operator,
    }
    if signature is not None:
        attrs["signature"] = signature

    mem.add_node(
        node_id=node_id,
        type_=NodeType.OPERATOR,
        embedding=embedding or [],
        attrs=attrs,
    )

    # Mirror stats into node stats
    mem.update_stats(node_id, operator.stats.to_stats_dict())

    return node_id
