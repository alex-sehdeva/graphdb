# gsm/memory/nx_backend.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

from .core import MemoryGraph, NodeId, NodeType, EdgeType


class NxMemoryGraph(MemoryGraph):
    """
    NetworkX-backed implementation of MemoryGraph.

    This is a *global*, long-lived graph storing:

      - EVENT_INDEX nodes (episodes / examples)
      - SCHEMA nodes (abstract patterns / prototypes)
      - OPERATOR nodes (concept-level operators / skills)
      - ACTION / OPTION / REWARD nodes (for ARC-AGI 3 style tasks)
      - FEATURE / ENTITY / CONTEXT / TIME nodes

    Each node stores:

      - "type": NodeType
      - "embedding": optional List[float]
      - "stats": Dict[str, Any] (mutable statistics)
      - plus any extra attributes (e.g. "task_id", "operator_obj", etc.)

    Each edge stores:

      - "type": EdgeType
      - plus any extra attributes
    """

    def __init__(self) -> None:
        # MultiDiGraph lets us have multiple labeled edges between a pair.
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ---------- internal helpers ----------

    def _ensure_node(self, node_id: NodeId) -> None:
        if node_id not in self.g:
            self.g.add_node(
                node_id,
                type=None,
                embedding=None,
                stats={},
            )

    # ---------- MemoryGraph interface implementation ----------

    def neighbors(self, node_id: NodeId) -> Iterable[NodeId]:
        """
        Immediate successors of node_id.
        """
        return self.g.successors(node_id)

    def node_type(self, node_id: NodeId) -> NodeType:
        data = self.g.nodes[node_id]
        t = data.get("type")
        if t is None:
            raise KeyError(f"Node {node_id!r} has no NodeType set")
        return t

    def node_embedding(self, node_id: NodeId) -> List[float]:
        emb = self.g.nodes[node_id].get("embedding")
        return list(emb) if emb is not None else []

    def node_stats(self, node_id: NodeId) -> Dict[str, Any]:
        data = self.g.nodes[node_id]
        stats = data.get("stats")
        if stats is None:
            stats = {}
            data["stats"] = stats
        return stats

    def edges_from(self, node_id: NodeId) -> Iterable[Tuple[NodeId, NodeId, EdgeType]]:
        """
        Yield (src, dst, edge_type) for all outgoing edges.
        """
        for _, dst, key, data in self.g.out_edges(node_id, data=True, keys=True):
            etype = data.get("type")
            if etype is None:
                # Optionally skip unlabeled edges
                continue
            yield node_id, dst, etype

    def nodes_of_type(self, t: NodeType) -> Iterable[NodeId]:
        for nid, data in self.g.nodes(data=True):
            if data.get("type") == t:
                yield nid

    def get_attr(self, node_id: NodeId, key: str, default: Any = None) -> Any:
        return self.g.nodes[node_id].get(key, default)

    # ---------- mutation helpers ----------

    def add_node(
        self,
        node_id: NodeId,
        type_: NodeType,
        embedding: List[float] | None = None,
        attrs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure node exists and merge attributes.

        This does not erase existing attributes; it only sets or overwrites
        the ones explicitly given (type_, embedding, attrs).
        """
        self._ensure_node(node_id)
        node_data = self.g.nodes[node_id]
        node_data["type"] = type_
        if embedding is not None:
            node_data["embedding"] = list(embedding)
        if attrs:
            node_data.update(attrs)

    def add_edge(
        self,
        src: NodeId,
        dst: NodeId,
        type_: EdgeType,
        attrs: Dict[str, Any] | None = None,
    ) -> None:
        """
        Add a directed edge src -> dst with EdgeType type_.

        attrs are merged into the edge attribute dict.
        """
        self._ensure_node(src)
        self._ensure_node(dst)
        data: Dict[str, Any] = {"type": type_}
        if attrs:
            data.update(attrs)
        self.g.add_edge(src, dst, **data)

    def update_stats(self, node_id: NodeId, updates: Dict[str, Any]) -> None:
        """
        Update stats[node_id] with the given key/value pairs.
        """
        stats = self.node_stats(node_id)
        stats.update(updates)

