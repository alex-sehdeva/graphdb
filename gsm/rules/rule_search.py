# gsm/rules/rule_search.py

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, List, Optional, Tuple

from gsm.core.graph import Graph
from gsm.core.canon import canonical_form, graph_code
from gsm.rules.rule_program import RuleProgram
from gsm.operators.base import GraphOperator


GraphPair = Tuple[Graph, Graph]


@dataclass
class RuleSearchConfig:
    """
    Configuration for the rule search space.

    - max_program_length: maximum number of operators in a RuleProgram.
    - operator_generators: list of callables that each return a finite set
      of GraphOperators to use in the search space.
    """
    max_program_length: int = 2
    operator_generators: List[Callable[[], Iterable[GraphOperator]]] = field(
        default_factory=list
    )


def generate_operator_candidates(config: RuleSearchConfig) -> Iterable[List[GraphOperator]]:
    """
    Generate sequences of operators up to max_program_length.

    Current simple strategy:
      - collect all operators from all operator_generators into one pool
      - take Cartesian products (with repetition) to form sequences
        of length 1..max_program_length
    """
    # collect operators from all generators
    merged_pool: List[GraphOperator] = []
    for gen in config.operator_generators:
        merged_pool.extend(list(gen()))

    if not merged_pool:
        return []

    # sequences of length 1..max_program_length
    for length in range(1, config.max_program_length + 1):
        for combo in product(merged_pool, repeat=length):
            yield list(combo)


@dataclass
class RuleSearcher:
    """
    Brute-force rule search engine over a space of GraphOperator sequences.

    Given training pairs (G_in, G_out), it:
      - canonicalizes each graph
      - enumerates candidate RulePrograms
      - returns the first program that matches all training pairs
        (up to canonical graph equality).
    """

    config: RuleSearchConfig

    # ---------- internal helpers ----------

    @staticmethod
    def _graphs_equal(g1: Graph, g2: Graph) -> bool:
        """
        Equality comparison via canonical graph_code.
        """
        return graph_code(g1) == graph_code(g2)

    def _canonicalize_pairs(self, train_pairs: List[GraphPair]) -> List[GraphPair]:
        """
        Map each (G_in, G_out) to their canonical forms.
        """
        canon_pairs: List[GraphPair] = []
        for g_in, g_out in train_pairs:
            cin, _ = canonical_form(g_in)
            cout, _ = canonical_form(g_out)
            canon_pairs.append((cin, cout))
        return canon_pairs

    def _program_matches_all(
        self,
        program: RuleProgram,
        canon_pairs: List[GraphPair],
    ) -> bool:
        """
        Check if program transforms every canonical input into the matching canonical output.
        """
        for cin, cout in canon_pairs:
            predicted = program(cin)
            p_can, _ = canonical_form(predicted)
            c_can, _ = canonical_form(cout)
            if not self._graphs_equal(p_can, c_can):
                return False
        return True

    # ---------- public API ----------

    def search(
        self,
        train_pairs: List[GraphPair],
        name_prefix: str = "rule",
    ) -> Optional[RuleProgram]:
        """
        Search for a RuleProgram that fits all training pairs.

        Returns the first matching RuleProgram found, or None if no match exists
        in the configured search space.
        """
        canon_pairs = self._canonicalize_pairs(train_pairs)

        for idx, op_seq in enumerate(generate_operator_candidates(self.config)):
            prog = RuleProgram.from_ops(name=f"{name_prefix}_{idx}", ops=op_seq)
            if self._program_matches_all(prog, canon_pairs):
                return prog

        return None

