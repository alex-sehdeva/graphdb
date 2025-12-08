# gsm/memory/core.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Protocol, Tuple, Hashable


# In gsm/core/graph.py you already use hashable node ids.
# We reuse that convention here.
NodeId = Hashable


class NodeType(Enum):
    """
    High-level node roles in the global MemoryGraph.

    This sits above the low-level gsm.core.Graph abstraction.
    A single MemoryGraph node can *refer to* or *summarize* many
    core.Graph objects, operators, or episodes.
    """
    ENTITY = auto()       # Objects / entities (ARC objects, game entities, etc.)
    FEATURE = auto()      # Color, shape, role, size, etc.
    CONTEXT = auto()      # Task, level, environment, goal
    TIME = auto()         # Timepoints / temporal buckets
    SCHEMA = auto()       # Abstract patterns / prototypes
    EVENT_INDEX = auto()  # Episodic index node (one experience / example)
    OPERATOR = auto()     # Concept-level operator (rule / skill)
    ACTION = auto()       # Atomic action (ARC-AGI 3 / game control)
    REWARD = auto()       # Reward markers
    OPTION = auto()       # Higher-level skills / options


class EdgeType(Enum):
    """
    High-level relation types between MemoryGraph nodes.

    These are *semantic* relations. They do not replace low-level edges
    inside gsm.core.Graph; they talk about how episodes, entities,
    schemas, and operators relate to each other across time and tasks.
    """
    HAS_FEATURE = auto()
    IN_CONTEXT = auto()
    BEFORE = auto()
    CAUSES = auto()
    PART_OF = auto()
    SIMILAR_TO = auto()
    INSTANCE_OF = auto()
    APPLIES_TO = auto()
    PERFORMS = auto()
    LEADS_TO = auto()
    YIELDS = auto()
    REFINES = auto()
    SPATIAL_RELATION = auto()
    """
    Generic spatial relation between entities.

    The specific relation is stored in edge attrs["relation"], e.g.:
      - "above"
      - "below"
      - "left_of"
      - "right_of"
    """


class OperatorKind(Enum):
    """
    Kinds of concept-level operators that live as nodes in MemoryGraph.

    These can wrap or refer to gsm.operators.GraphOperator instances,
    but also extend beyond ARC-style static transforms.
    """
    STATIC_TRANSFORM = auto()   # ARC I/O style: core.Graph -> core.Graph
    DYNAMICS = auto()           # state_t + action -> state_{t+1}
    POLICY = auto()             # state -> action / option
    META = auto()               # operators on operators / schemas (meta-reasoning)


class RetrievalGoal(Enum):
    """
    High-level intention passed to the RetrievalEngine.

    - PATTERN_COMPLETION: "What else belongs with this pattern?"
    - RULE_SUGGESTION: "Which operators/rules might explain this ARC task?"
    - ACTION_SUGGESTION: "What action/option should I take next?"
    - EXPLANATION: "What schemas/operators could explain this situation?"
    """
    PATTERN_COMPLETION = auto()
    RULE_SUGGESTION = auto()
    ACTION_SUGGESTION = auto()
    EXPLANATION = auto()


class MemoryGraph(Protocol):
    """
    Abstract interface for the long-term memory graph.

    This is intentionally separate from gsm.core.Graph:

    - gsm.core.Graph  : minimal axiomatic graph for a single structure
                        (ARC grid, object graph, concept graph, etc.).
    - MemoryGraph     : large, evolving graph over *experiences*,
                        *schemas*, *operators*, *actions*, etc.

    Different backends can implement this Protocol. The first one is
    NxMemoryGraph (networkx.MultiDiGraph) in gsm.memory.nx_backend.
    """

    # ---------- Core query methods ----------

    def neighbors(self, node_id: NodeId) -> Iterable[NodeId]:
        """
        Immediate successors of node_id in the MemoryGraph.
        """
        ...

    def node_type(self, node_id: NodeId) -> NodeType:
        """
        Return the semantic NodeType of a node.

        Implementations should raise KeyError if node_id is unknown.
        """
        ...

    def node_embedding(self, node_id: NodeId) -> List[float]:
        """
        Return a vector embedding for this node, or [] if not set.

        Embeddings allow similarity search and seed selection in retrieval.
        """
        ...

    def node_stats(self, node_id: NodeId) -> Dict[str, Any]:
        """
        Return a mutable dict of statistics for this node.

        Typical keys:
          - created_at
          - last_used_at
          - access_count
          - avg_reward
        """
        ...

    def edges_from(self, node_id: NodeId) -> Iterable[Tuple[NodeId, NodeId, EdgeType]]:
        """
        Yield (src, dst, edge_type) for all outgoing edges from node_id.
        """
        ...

    def nodes_of_type(self, t: NodeType) -> Iterable[NodeId]:
        """
        Iterate over all nodes with the given NodeType.
        """
        ...

    def get_attr(self, node_id: NodeId, key: str, default: Any = None) -> Any:
        """
        Get an arbitrary attribute attached to a node, e.g.:

            mem.get_attr(node, "operator_obj")
            mem.get_attr(node, "task_id")
        """
        ...

    # ---------- Convenience mutation methods ----------

    def add_node(
        self,
        node_id: NodeId,
        type_: NodeType,
        embedding: List[float] | None = None,
        attrs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure a node exists and set its NodeType, embedding, and attributes.

        Implementations should *merge* attrs into any existing node dict
        rather than overwriting everything.
        """
        ...

    def add_edge(
        self,
        src: NodeId,
        dst: NodeId,
        type_: EdgeType,
        attrs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Add a directed edge src -> dst with the given EdgeType and attrs.
        """
        ...

    def update_stats(self, node_id: NodeId, updates: Dict[str, Any]) -> None:
        """
        Update the stats sub-dict for a node with the given key/values.
        """
        ...

