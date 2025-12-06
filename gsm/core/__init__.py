"""
core: Axiomatic graph substrate for gsm.

This module exposes the main core primitives:
- Graph                (nodes + undirected edges)
- GraphHomomorphism    (structure-preserving maps)
- quotient_graph       (partition → quotient)
- canonical_form       (canonical labeling)
- graph_code           (hashable code for equality)

Higher-level systems (memory, agents, domains) should build on this,
not reimplement graph logic.
"""

from .graph import Graph
from .homomorphism import GraphHomomorphism
from .quotient import quotient_graph
from .canon import canonical_form, graph_code
from .symmetry import is_automorphism

