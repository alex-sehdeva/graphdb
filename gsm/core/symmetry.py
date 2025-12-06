from typing import Dict, Any

from .graph import Graph, NodeId


def is_automorphism(G: Graph, perm: Dict[NodeId, NodeId]) -> bool:
    """
    Check whether 'perm' is an automorphism of G.

    Conditions:
      - perm is a bijection on nodes(G)
      - attributes are preserved exactly
      - edge set is preserved
    """
    # bijection on domain
    if set(perm.keys()) != set(G.nodes.keys()):
        return False
    if len(set(perm.values())) != len(perm.values()):
        return False

    # attribute preservation
    for u, attrs in G.nodes.items():
        v = perm[u]
        if v not in G.nodes:
            return False
        if G.nodes[v] != attrs:
            return False

    # edge preservation
    for (u, v) in G.edges:
        u2 = perm[u]
        v2 = perm[v]
        if (u2, v2) not in G.edges:
            return False

    return True

