# gsm/memory/lexicon.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from gsm.memory.core import MemoryGraph, NodeId, NodeType
from gsm.memory.operators import Operator


@dataclass
class LexiconEntry:
    """
    A single lexicon entry tying a phrase to one or more OPERATOR nodes.

    For now this is extremely simple and string-based. In the future, a
    semantic parser / LM can normalize phrases and map them here.

    Example:
      phrase = "move right"
      operator_ids = ["op:translation:+1:0"]
    """
    phrase: str
    operator_ids: List[NodeId]


def index_lexicon_from_memory(mem: MemoryGraph) -> Dict[str, LexiconEntry]:
    """
    Build a simple lexicon index from the MemoryGraph.

    We scan all OPERATOR nodes and look for a 'lexical_labels' attribute,
    which is expected to be a list of strings.

    Returns:
        A dict phrase -> LexiconEntry.
    """
    index: Dict[str, LexiconEntry] = {}

    for op_id in mem.nodes_of_type(NodeType.OPERATOR):
        labels = mem.get_attr(op_id, "lexical_labels", []) or []
        for phrase in labels:
            entry = index.get(phrase)
            if entry is None:
                index[phrase] = LexiconEntry(phrase=phrase, operator_ids=[op_id])
            else:
                if op_id not in entry.operator_ids:
                    entry.operator_ids.append(op_id)

    return index


def find_operators_by_phrase(
    mem: MemoryGraph,
    phrase: str,
    *,
    exact: bool = True,
    max_results: int = 10,
) -> List[Tuple[NodeId, Operator]]:
    """
    Look up operators that are labeled with the given phrase.

    For now this is very naive:

      - If exact=True:
          require phrase to be in lexical_labels exactly.
      - If exact=False:
          allow substring matches (phrase in label.lower()).

    Returns:
        A list of (operator_node_id, Operator) tuples, up to max_results.
    """
    phrase_norm = phrase.strip().lower()
    candidates: List[Tuple[NodeId, Operator]] = []

    for op_id in mem.nodes_of_type(NodeType.OPERATOR):
        labels = mem.get_attr(op_id, "lexical_labels", []) or []
        labels_norm = [lab.lower() for lab in labels]

        match = False
        if exact:
            match = phrase_norm in labels_norm
        else:
            match = any(phrase_norm in lab for lab in labels_norm)

        if not match:
            continue

        op_obj = mem.get_attr(op_id, "operator_obj", None)
        if isinstance(op_obj, Operator):
            candidates.append((op_id, op_obj))

        if len(candidates) >= max_results:
            break

    return candidates

