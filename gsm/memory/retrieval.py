# gsm/memory/retrieval.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, NamedTuple

from .core import (
    MemoryGraph,
    NodeId,
    NodeType,
    EdgeType,
    RetrievalGoal,
)
from .operators import Operator
from .cues import Cue


class RetrievalResult(NamedTuple):
    """
    Output of a RetrievalEngine.retrieve(...) call.

    - activated_nodes: list of node ids with nonzero activation.
    - node_activation: mapping node_id -> activation level (normalized to [0, 1]).
    - candidate_events: EVENT_INDEX nodes (episodes/examples) that are active.
    - candidate_operators: Operator objects surfaced for this cue/goal.
    - candidate_actions: ACTION / OPTION node ids (for ACTION_SUGGESTION).
    """
    activated_nodes: List[NodeId]
    node_activation: Dict[NodeId, float]
    candidate_events: List[NodeId]
    candidate_operators: List[Operator]
    candidate_actions: List[NodeId]


class RetrievalEngine:
    """
    Spreading-activation based retrieval over a MemoryGraph.

    High-level flow:

      1. Seed activation on candidate EVENT_INDEX nodes based on the cue
         (for now: a simple subset, later: embedding similarity, context).
      2. Run spreading activation for a fixed number of steps.
      3. Depending on the RetrievalGoal, extract:
         - operators (RULE_SUGGESTION, EXPLANATION)
         - actions/options (ACTION_SUGGESTION)
         - events/schemas (PATTERN_COMPLETION / EXPLANATION)
    """

    def __init__(self, graph: MemoryGraph) -> None:
        self.graph = graph

    # ---------- Public API ----------

    def retrieve(
        self,
        cue: Cue,
        goal: RetrievalGoal,
        k_results: int = 10,
        spread_steps: int = 3,
    ) -> RetrievalResult:
        """
        Main entry point.

        Args:
            cue: Cue describing the current question/situation.
            goal: RetrievalGoal specifying the intention.
            k_results: how many top operators / actions to return.
            spread_steps: number of spreading activation steps.

        Returns:
            RetrievalResult with activated region + candidate operators/actions.
        """
        seed_events = self._find_candidate_events(cue, k=k_results * 5)
        activation = self._spreading_activation(seed_events, steps=spread_steps)

        activated_nodes = [n for n, a in activation.items() if a > 0.0]
        candidate_events = [
            n for n in activated_nodes
            if self.graph.node_type(n) == NodeType.EVENT_INDEX
        ]

        candidate_operators = self._select_operators(
            cue=cue,
            activation=activation,
            goal=goal,
            k=k_results,
        )

        candidate_actions = self._select_actions(
            cue=cue,
            activation=activation,
            goal=goal,
            k=k_results,
        )

        return RetrievalResult(
            activated_nodes=activated_nodes,
            node_activation=activation,
            candidate_events=candidate_events,
            candidate_operators=candidate_operators,
            candidate_actions=candidate_actions,
        )

    # ---------- Internal helpers ----------

    def _find_candidate_events(self, cue: Cue, k: int) -> List[NodeId]:
        """
        Seed selection for EVENT_INDEX nodes.

        Current simple strategy:
          - Return up to k EVENT_INDEX nodes from the graph.

        Future improvements:
          - Filter by task_id in node attrs.
          - Use embeddings to find nearest neighbors to cue.embedding.
          - Use recency / context bias.
        """
        events: List[NodeId] = list(self.graph.nodes_of_type(NodeType.EVENT_INDEX))

        # Simple filtering: if cue.task_id is set, prefer matching context nodes.
        if cue.task_id is not None:
            filtered: List[NodeId] = []
            for ev in events:
                task_id = self.graph.get_attr(ev, "task_id")
                if task_id == cue.task_id:
                    filtered.append(ev)
            if filtered:
                events = filtered

        return events[:k]

    def _spreading_activation(
        self,
        seed_nodes: Iterable[NodeId],
        steps: int = 3,
        leak: float = 0.2,
        spread_factor: float = 0.5,
    ) -> Dict[NodeId, float]:
        """
        Simple synchronous spreading activation with leakage.

        Args:
            seed_nodes: initial set of nodes with activation 1.0
            steps: number of propagation iterations
            leak: fraction of activation lost each step
            spread_factor: multiplicative factor for propagation along edges

        Returns:
            activation: node_id -> activation level (normalized to [0, 1])
        """
        activation: Dict[NodeId, float] = {n: 1.0 for n in seed_nodes}

        for _ in range(steps):
            new_act: Dict[NodeId, float] = defaultdict(float)

            for n, a in activation.items():
                # Retain some activation on the node itself
                new_act[n] += a * (1.0 - leak)

                # Propagate along selected edge types
                for _, dst, etype in self.graph.edges_from(n):
                    if self._edge_type_propagates(etype):
                        new_act[dst] += a * spread_factor

            activation = self._normalize(new_act)

        return activation

    def _edge_type_propagates(self, etype: EdgeType) -> bool:
        """
        Decide which edge types propagate activation.

        You can tune this over time; for now we propagate across structural
        and semantic relations that are likely to carry relevance.
        """
        return etype in {
            EdgeType.HAS_FEATURE,
            EdgeType.IN_CONTEXT,
            EdgeType.PART_OF,
            EdgeType.BEFORE,
            EdgeType.APPLIES_TO,
            EdgeType.LEADS_TO,
            EdgeType.REFINES,
            EdgeType.INSTANCE_OF,
            EdgeType.CAUSES,
            EdgeType.SIMILAR_TO,
        }

    def _normalize(self, activation: Dict[NodeId, float]) -> Dict[NodeId, float]:
        """
        Normalize activation dict so max value is 1.0 (if non-empty).
        """
        if not activation:
            return activation
        max_val = max(activation.values())
        if max_val <= 0.0:
            # All zero or negative; just return as is.
            return activation
        return {n: a / max_val for n, a in activation.items()}

    def _select_operators(
        self,
        cue: Cue,
        activation: Dict[NodeId, float],
        goal: RetrievalGoal,
        k: int,
    ) -> List[Operator]:
        """
        Select a ranked list of Operators from the activated region.

        Ranking heuristic:
          - base activation of the operator node
          - plus operator.score_applicability(cue, graph)

        Filtering by goal:
          - RULE_SUGGESTION: STATIC_TRANSFORM only
          - ACTION_SUGGESTION: DYNAMICS / POLICY only
          - other goals: no filtering (for now)
        """
        operator_nodes: List[NodeId] = [
            n for n in activation.keys()
            if self.graph.node_type(n) == NodeType.OPERATOR
        ]

        ops: List[Operator] = []
        for n in operator_nodes:
            op_obj = self.graph.get_attr(n, "operator_obj")
            if op_obj is None:
                continue
            if not isinstance(op_obj, Operator):
                # You may choose to log or warn here.
                continue

            base_act = activation.get(n, 0.0)
            applicability = op_obj.score_applicability(cue, self.graph)
            score = base_act + applicability
            op_obj.last_applicability_score = score
            ops.append(op_obj)

        # Filter by goal
        from .core import OperatorKind  # local import to avoid cycles in some setups

        if goal == RetrievalGoal.RULE_SUGGESTION:
            ops = [op for op in ops if op.kind == OperatorKind.STATIC_TRANSFORM]
        elif goal == RetrievalGoal.ACTION_SUGGESTION:
            ops = [op for op in ops if op.kind in {OperatorKind.DYNAMICS, OperatorKind.POLICY}]
        # For PATTERN_COMPLETION / EXPLANATION we leave them all.

        ops.sort(key=lambda op: op.last_applicability_score, reverse=True)
        return ops[:k]

    def _select_actions(
        self,
        cue: Cue,
        activation: Dict[NodeId, float],
        goal: RetrievalGoal,
        k: int,
    ) -> List[NodeId]:
        """
        Select ACTION / OPTION nodes from the activated region.

        For ARC-AGI 1:
          - this will usually return [].

        For ARC-AGI 3 / game-like environments:
          - action nodes connected to high-reward episodes will be ranked.
        """
        if goal != RetrievalGoal.ACTION_SUGGESTION:
            return []

        action_nodes: List[NodeId] = [
            n for n in activation.keys()
            if self.graph.node_type(n) in {NodeType.ACTION, NodeType.OPTION}
        ]

        scored: List[tuple[float, NodeId]] = []
        for n in action_nodes:
            base_act = activation.get(n, 0.0)
            stats = self.graph.node_stats(n)
            avg_reward = stats.get("avg_reward", 0.0)
            score = base_act + avg_reward
            scored.append((score, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:k]]

