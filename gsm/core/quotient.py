from typing import Dict, Hashable

from .graph import Graph, NodeId


def quotient_graph(G: Graph, partition: Dict[NodeId, Hashable]) -> Graph:
    """
    Build the quotient graph G/~ from a partition mapping node -> block_id.

    - Each block_id in partition.values() becomes a node in the quotient.
    - There is an edge between block_ids B1 and B2 iff some u in B1,
      v in B2 had an edge (u, v) in G.

    This is the main abstraction tool for collapsing fibers into roles.
    """
    H = Graph()

    # create quotient nodes (blocks)
    blocks = set(partition.values())
    for b in blocks:
        H.add_node(b)

    # create quotient edges
    for (u, v) in G.edges:
        bu = partition[u]
        bv = partition[v]
        if bu != bv:
            H.add_edge(bu, bv)

    return H

