from dataclasses import dataclass
from typing import Dict, Hashable

from .graph import Graph, NodeId


@dataclass
class GraphHomomorphism:
    """
    A graph homomorphism φ : G -> H.

    Requirements:
    - mapping: node in G -> node in H
    - edge preservation: (u,v) in E(G) ⇒ (φ(u), φ(v)) in E(H)

    This class does NOT try to compute homomorphisms; it just
    represents and validates a mapping.
    """

    mapping: Dict[NodeId, NodeId]

    # --- validation ---

    def check(self, G: Graph, H: Graph) -> bool:
        """Return True iff this mapping is a graph homomorphism G -> H."""
        # all keys must be nodes of G
        if not set(self.mapping.keys()) <= set(G.nodes.keys()):
            return False
        # images must be nodes of H
        if not set(self.mapping.values()) <= set(H.nodes.keys()):
            return False

        for (u, v) in G.edges:
            u_img = self.mapping[u]
            v_img = self.mapping[v]
            if (u_img, v_img) not in H.edges:
                return False
        return True

    # --- categorical flavors ---

    def is_injective(self) -> bool:
        """Injective on vertices."""
        return len(set(self.mapping.values())) == len(self.mapping.values())

    def is_surjective(self, H: Graph) -> bool:
        """Surjective onto all nodes of H."""
        return set(self.mapping.values()) == set(H.nodes.keys())

