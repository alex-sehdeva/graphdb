from typing import Any, Dict, Tuple, List
from itertools import permutations, product
from collections import defaultdict

from .graph import Graph, NodeId


def _node_signature(attrs: Dict[str, Any], degree: int) -> Tuple:
    """
    Signature used to bucket nodes during canonicalization.

    Currently: (degree, sorted(attrs.items())).
    You can extend this if needed.
    """
    attr_items = tuple(sorted(attrs.items()))
    return (degree, attr_items)


def graph_code(G: Graph) -> Tuple:
    """
    Produce a canonical-ish code for a *labeled* graph G.

    - Node order: sorted by node id
    - Node signatures: (degree, attrs)
    - Adjacency bits: upper-triangular pattern

    If two graphs are isomorphic via relabeling that preserves
    signatures, and are both in canonical label, they will share this code.
    """
    if not G.nodes:
        return ((), ())

    nodes_sorted = sorted(G.nodes.keys())
    deg = {n: G.degree(n) for n in nodes_sorted}
    node_sigs = [_node_signature(G.nodes[n], deg[n]) for n in nodes_sorted]

    # adjacency upper triangle
    adj_bits: List[int] = []
    for i, u in enumerate(nodes_sorted):
        for j, v in enumerate(nodes_sorted):
            if i < j:
                adj_bits.append(1 if (u, v) in G.edges else 0)

    return (tuple(node_sigs), tuple(adj_bits))


def relabel_graph(G: Graph, perm: Dict[NodeId, NodeId]) -> Graph:
    """Apply permutation old_id -> new_id to nodes and edges."""
    new = Graph()
    for old_id, attrs in G.nodes.items():
        new_id = perm[old_id]
        new.add_node(new_id, **attrs)
    for (u, v) in G.edges:
        new.add_edge(perm[u], perm[v])
    return new


def _node_partition_by_signature(G: Graph) -> Dict[Tuple, List[NodeId]]:
    """
    Partition nodes into signature buckets. Each bucket can only permute internally.
    """
    buckets: Dict[Tuple, List[NodeId]] = defaultdict(list)
    for n, attrs in G.nodes.items():
        sig = _node_signature(attrs, G.degree(n))
        buckets[sig].append(n)
    return buckets


def canonical_form(G: Graph) -> Tuple[Graph, Dict[NodeId, NodeId]]:
    """
    Compute a canonical relabeling of G.

    Returns:
        canon_G: Graph with canonical node ids
        perm:    mapping old_id -> new_id used for relabeling

    Algorithm:
      - Partition nodes by (degree, attrs).
      - For each bucket, enumerate permutations.
      - Combine bucket permutations and pick the relabeling with
        lexicographically smallest graph_code.
    This is exponential in worst case but acceptable for small ARC-like objects.
    """
    if not G.nodes:
        return G, {}

    buckets = _node_partition_by_signature(G)
    bucket_keys = list(buckets.keys())

    # per-bucket permutations
    per_bucket_perms: List[List[Tuple[NodeId, ...]]] = []
    for key in bucket_keys:
        nodes = buckets[key]
        if len(nodes) == 1:
            per_bucket_perms.append([tuple(nodes)])
        else:
            per_bucket_perms.append(list(permutations(nodes)))

    all_nodes_sorted = sorted(G.nodes.keys())

    best_code = None
    best_perm: Dict[NodeId, NodeId] | None = None

    for choice in product(*per_bucket_perms):
        # flatten: choice is a tuple of tuples
        flat_old: List[NodeId] = []
        for tup in choice:
            flat_old.extend(list(tup))

        # map flat_old -> sorted target ids
        perm: Dict[NodeId, NodeId] = {
            old: new for old, new in zip(flat_old, all_nodes_sorted)
        }

        Gp = relabel_graph(G, perm)
        code = graph_code(Gp)

        if best_code is None or code < best_code:
            best_code = code
            best_perm = perm

    assert best_perm is not None
    canon_G = relabel_graph(G, best_perm)
    return canon_G, best_perm

