# gsm/memory/operators.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Set

from gsm.memory.core import NodeId, NodeType, OperatorKind, MemoryGraph
from gsm.memory.patterns import ConceptPattern

class DomainKind(Enum):
    """
    High-level domain categories an Operator can be instantiated in.

    This is intentionally coarse and *not* tied to concrete classes;
    adapters will map from concrete domain objects (grids, note graphs,
    proof states) into one of these kinds.

    Examples:
      - GRID_2D: ARC-like grids, pixel maps, spatial boards.
      - OBJECT_GRAPH: entity/relationship graphs (games, scenes).
      - SEQUENCE: linear sequences (token streams, time series).
      - TREE: syntactic trees, proof trees, expression ASTs.
      - MUSIC_GRAPH: notes, chords, timing relations.
      - GENERIC: works over any domain (purely conceptual).
    """
    GENERIC = auto()
    GRID_2D = auto()
    OBJECT_GRAPH = auto()
    SEQUENCE = auto()
    TREE = auto()
    MUSIC_GRAPH = auto()

class ValueKind(Enum):
    """
    What 'kind' of value a role binds to.

    This is more abstract than a Python type; it's meant to describe
    how the role participates in the operator:

      - NODE: single entity / node in a graph-like structure.
      - NODE_SET: a set/list of nodes.
      - EDGE: a relation/edge.
      - GRAPH: an entire core.Graph or domain-state graph.
      - ATTRIBUTE: attribute or feature value (e.g. 'color', 'pitch').
      - SCALAR: number / enum / small literal (e.g. dx, dy, n_steps).
      - PREDICATE: boolean condition.
    """
    NODE = auto()
    NODE_SET = auto()
    EDGE = auto()
    GRAPH = auto()
    ATTRIBUTE = auto()
    SCALAR = auto()
    PREDICATE = auto()

@dataclass
class RoleType:
    """
    Type information for a single role/argument of an Operator.

    Examples:
      - "x": NODE,   allowed_domains={GRID_2D, MUSIC_GRAPH}
      - "y": NODE,   allowed_domains={GRID_2D, MUSIC_GRAPH}
      - "grid": GRAPH, allowed_domains={GRID_2D}
      - "amount": SCALAR, allowed_domains={GENERIC}
    """
    name: str
    value_kind: ValueKind
    allowed_domains: Set[DomainKind] = field(default_factory=set)

    # Optional constraints, free-form for now:
    #   e.g. {"must_have_attr": "color"}, {"entity_role": "note"}
    constraints: Dict[str, Any] = field(default_factory=dict)

class OperatorEffectKind(Enum):
    """
    High-level category of what the operator *does*.

      - TRANSFORM: Graph/structure -> Graph/structure (e.g., Translation).
      - RELATION:   Establish or query a relation (e.g., Above(x, y)).
      - PREDICATE:  Boolean-valued check (e.g., IsAbove(x, y)).
      - POLICY:     Maps state -> action/option.
    """
    TRANSFORM = auto()
    RELATION = auto()
    PREDICATE = auto()
    POLICY = auto()

@dataclass
class OperatorTypeSignature:
    """
    Minimal type/role signature for a concept-level Operator.

    This does *not* execute anything. It tells us:

      - which DomainKind(s) this operator can live in,
      - what roles it expects as inputs,
      - what roles it produces or constrains as outputs,
      - what kind of effect it has (transform vs relation vs predicate vs policy).

    Examples:
      - Translation on grids:
          domain_kinds = {GRID_2D}
          effect_kind = TRANSFORM
          input_roles  = [ RoleType("grid", GRAPH, {GRID_2D}) ]
          output_roles = [ RoleType("grid", GRAPH, {GRID_2D}) ]

      - Above relation:
          domain_kinds = {GRID_2D, MUSIC_GRAPH, TREE}
          effect_kind = RELATION
          input_roles  = [ RoleType("x", NODE, {...}),
                           RoleType("y", NODE, {...}) ]
          output_roles = []  # relation recorded in graph, or used as constraint

      - IsAbove predicate:
          domain_kinds = {GRID_2D, MUSIC_GRAPH, TREE}
          effect_kind = PREDICATE
          input_roles  = [x, y]
          output_roles = [ RoleType("result", PREDICATE, {GENERIC}) ]
    """
    domain_kinds: Set[DomainKind] = field(default_factory=set)
    effect_kind: OperatorEffectKind = OperatorEffectKind.TRANSFORM

    input_roles: List[RoleType] = field(default_factory=list)
    output_roles: List[RoleType] = field(default_factory=list)

    # Optional: ad hoc flags for later
    is_total: bool = True    # total vs partial function
    is_pure: bool = True     # side-effect-free?
    notes: str | None = None

    @classmethod
    def for_transform(
        cls,
        *,
        domain_kinds: Iterable[DomainKind],
        input_roles: List[RoleType],
        output_roles: List[RoleType],
        notes: str | None = None,
    ) -> "OperatorTypeSignature":
        return cls(
            domain_kinds=set(domain_kinds),
            effect_kind=OperatorEffectKind.TRANSFORM,
            input_roles=input_roles,
            output_roles=output_roles,
            notes=notes,
        )

    @classmethod
    def for_relation(
        cls,
        *,
        domain_kinds: Iterable[DomainKind],
        input_roles: List[RoleType],
        notes: str | None = None,
    ) -> "OperatorTypeSignature":
        return cls(
            domain_kinds=set(domain_kinds),
            effect_kind=OperatorEffectKind.RELATION,
            input_roles=input_roles,
            output_roles=[],
            notes=notes,
        )

    @classmethod
    def for_predicate(
        cls,
        *,
        domain_kinds: Iterable[DomainKind],
        input_roles: List[RoleType],
        notes: str | None = None,
    ) -> "OperatorTypeSignature":
        return cls(
            domain_kinds=set(domain_kinds),
            effect_kind=OperatorEffectKind.PREDICATE,
            input_roles=input_roles,
            output_roles=[
                RoleType(
                    name="result",
                    value_kind=ValueKind.PREDICATE,
                    allowed_domains={DomainKind.GENERIC},
                )
            ],
            notes=notes,
        )

    @classmethod
    def for_policy(
        cls,
        *,
        domain_kinds: Iterable[DomainKind],
        input_roles: List[RoleType],
        output_roles: List[RoleType],
        notes: str | None = None,
    ) -> "OperatorTypeSignature":
        return cls(
            domain_kinds=set(domain_kinds),
            effect_kind=OperatorEffectKind.POLICY,
            input_roles=input_roles,
            output_roles=output_roles,
            notes=notes,
        )


@dataclass
class OperatorSemanticInfo:
    """
    Optional, domain-agnostic semantic description of an Operator.

    This is *not* executable code; it is structured metadata that:

      - Describes roles (arguments) conceptually
      - Lists invariants (things preserved)
      - Lists effects (what changes / what this does)
      - Notes polymorphic axes (how this can be instantiated in different domains)

    Examples:
      - Translation(dx=1, dy=0):
          roles = ["object"]
          invariants = ["shape_preserved", "color_multiset_preserved"]
          effects = ["translate_along_x"]
          polymorphic_axes = ["spatial_x_axis"]

      - A future 'Above' relation operator:
          roles = ["x", "y"]
          invariants = ["underlying_identity_preserved"]
          effects = ["constrains_order"]
          polymorphic_axes = ["vertical_axis"]
    """
    roles: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    polymorphic_axes: List[str] = field(default_factory=list)

    # Optional: free-form notes for debugging / explanation
    description: str | None = None

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

    # 🔹 NEW: semantic description for language-grounding
    semantic: Optional[OperatorSemanticInfo] = None

    # 🔹 NEW: lexical labels (words / phrases pointing to this operator)
    lexical_labels: List[str] = field(default_factory=list)

    # 🔹 NEW: minimal type/role signature
    type_signature: Optional[OperatorTypeSignature] = None

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

    def add_label(self, label: str) -> None:
        """
        Add a lexical label (if not already present).

        This does not automatically update the MemoryGraph node; use
        gsm.memory.operators.add_operator_node(...) or a dedicated helper
        to sync to the graph if needed.
        """
        if label not in self.lexical_labels:
            self.lexical_labels.append(label)

    def add_labels(self, labels: List[str]) -> None:
        for lab in labels:
            self.add_label(lab)

    def supports_domain(self, domain: DomainKind) -> bool:
        """
        Return True if this operator's type_signature says it can
        be instantiated in the given DomainKind.

        If no type_signature is set, we conservatively return True
        (treat as GENERIC) to avoid breaking old code.
        """
        if self.type_signature is None:
            return True
        if not self.type_signature.domain_kinds:
            return True
        return domain in self.type_signature.domain_kinds


def add_operator_node(
    mem: MemoryGraph,
    operator: Operator,
    embedding: Optional[List[float]] = None,
    signature: Any = None,
    labels: Optional[List[str]] = None,
) -> NodeId:
    """
    Insert an OPERATOR node into the MemoryGraph and attach the Operator object.

    If signature is provided, it is stored in the node attrs under 'signature'
    so that promotion/merge logic can identify structurally equivalent operators.

    Returns:
        node_id: the MemoryGraph node id for this operator.
    """
    node_id: NodeId = f"op:{operator.id}"

    # Combine labels from the Operator and the explicit argument.
    combined_labels: List[str] = []
    if operator.lexical_labels:
        combined_labels.extend(operator.lexical_labels)
    if labels:
        for lab in labels:
            if lab not in combined_labels:
                combined_labels.append(lab)

    attrs: Dict[str, Any] = {
        "name": operator.name,
        "operator_obj": operator,
    }
    if signature is not None:
        attrs["signature"] = signature
    if combined_labels:
        attrs["lexical_labels"] = combined_labels
    if operator.semantic is not None:
        # Store semantic info in a JSON-serializable way if you care about export,
        # or just inline the object for now.
        attrs["semantic"] = operator.semantic

    mem.add_node(
        node_id=node_id,
        type_=NodeType.OPERATOR,
        embedding=embedding or [],
        attrs=attrs,
    )

    # Mirror stats into node stats
    mem.update_stats(node_id, operator.stats.to_stats_dict())

    return node_id
