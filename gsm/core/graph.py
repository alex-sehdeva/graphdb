from dataclasses import dataclass, field
from typing import Dict, Any, Set, Tuple, Iterable, Hashable

NodeId = Hashable
Edge = Tuple[NodeId, NodeId]


@dataclass
class Graph:
    """
    Minimal undirected graph with attribute-bearing nodes.

    - nodes: mapping node_id -> attribute dict
    - edges: set of (u, v) pairs, stored symmetrically for convenience

    This is intentionally small and generic enough for:
    - grids (nodes = (r, c), attrs include 'color', etc.)
    - object graphs
    - feature / concept graphs
    """

    nodes: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    edges: Set[Edge] = field(default_factory=set)

    # --- basic construction ---

    def add_node(self, nid: NodeId, **attrs: Any) -> None:
        """Add or update a node with attributes."""
        self.nodes[nid] = dict(attrs)

    def add_edge(self, u: NodeId, v: NodeId) -> None:
        """Add an undirected edge between u and v, if both exist."""
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"Both endpoints must exist as nodes: {u}, {v}")
        self.edges.add((u, v))
        self.edges.add((v, u))

    # --- introspection ---

    def neighbors(self, nid: NodeId) -> Set[NodeId]:
        """Return the neighbor set of nid."""
        return {v for (u, v) in self.edges if u == nid}

    def degree(self, nid: NodeId) -> int:
        return len(self.neighbors(nid))

    # --- induced subgraph ---

    def induced_subgraph(self, node_ids: Iterable[NodeId]) -> "Graph":
        """Return the induced subgraph on the given node set."""
        node_set = set(node_ids)
        g = Graph()
        for n in node_set:
            if n not in self.nodes:
                raise KeyError(f"Unknown node id in induced_subgraph: {n}")
            g.add_node(n, **self.nodes[n])
        for (u, v) in self.edges:
            if u in node_set and v in node_set:
                g.add_edge(u, v)
        return g

