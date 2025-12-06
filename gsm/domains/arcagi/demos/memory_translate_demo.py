# gsm/domains/arcagi/demos/memory_translate_demo.py
from __future__ import annotations

from typing import Any, List

import numpy as np

from gsm.memory.nx_backend import NxMemoryGraph
from gsm.memory.retrieval import RetrievalEngine, RetrievalGoal
from gsm.memory.cues import Cue
from gsm.memory.operators import Operator, OperatorStats, add_operator_node
from gsm.memory.patterns import ConceptPattern
from gsm.memory.core import OperatorKind, EdgeType

from gsm.domains.arcagi.memory_ingest import ingest_arc_task, load_arc_task_json
from gsm.domains.arcagi.convert import grid_to_graph, graph_to_grid

from gsm.operators.grid import Translation, RecenterToOrigin

def build_demo_operator() -> Operator:
    """
    Build a memory-level Operator that wraps a composed GraphOperator:
        Translation(dx=1, dy=0) then RecenterToOrigin().
    """
    graph_op = Translation(dx=1, dy=0).then(RecenterToOrigin())

    input_pattern = ConceptPattern()

    op = Operator(
        id="demo_translate_right_1",
        name="DemoTranslateRightBy1",
        kind=OperatorKind.STATIC_TRANSFORM,
        input_pattern=input_pattern,
        output_pattern=None,
        parameters={"dx": 1, "dy": 0, "recenter": True},
        stats=OperatorStats(created_at=0, last_used_at=0),
        underlying=graph_op,
    )
    return op

def apply_underlying_translation(grid: List[List[int]], op: Operator) -> List[List[int]]:
    """
    Apply the underlying GraphOperator (Translation) to an ARC grid.
    """
    if op.underlying is None:
        raise ValueError("Operator has no underlying GraphOperator")

    # Convert grid -> core.Graph
    g_in = grid_to_graph(grid)
    g_out = op.underlying(g_in)  # Translation is a GraphOperator: Graph -> Graph

    # Convert back to grid
    grid_out = graph_to_grid(g_out)
    return grid_out


def run_demo() -> None:
    task_id = "demo_translate_right"

    # 1. Build MemoryGraph + RetrievalEngine
    mem = NxMemoryGraph()
    retrieval = RetrievalEngine(mem)

    # 2. Ingest the entire task from gsm/domains/arcagi/arc_tasks/demo_translate_right.json
    ingest_arc_task(mem, task_id=task_id, start_timestep=0)

    # 3. Create a memory-level Operator wrapping the Translation GraphOperator
    demo_op = build_demo_operator()
    op_node_id = add_operator_node(mem, demo_op)

    print(f"Added OPERATOR node: {op_node_id} ({demo_op.name})")

    context_node = f"arc:ctx:task:{task_id}"
    mem.add_edge(
        src=context_node,
        dst=op_node_id,
        type_=EdgeType.APPLIES_TO,
        attrs={"source": "demo"},
    )

    # 4. Load the task JSON so we can grab the test input grid
    task = load_arc_task_json(task_id)
    test_grid = task["test"][0]["input"]

    print("Test input grid:")
    print(np.array(test_grid))

    # 5. Build a Cue for the test input
    cue = Cue.from_arc_example(
        grid_in=test_grid,
        grid_out=None,
        task_id=task_id,
        timestep=999,  # some large logical time
    )

    # 6. Run retrieval with goal=RULE_SUGGESTION
    result = retrieval.retrieve(
        cue=cue,
        goal=RetrievalGoal.RULE_SUGGESTION,
        k_results=5,
    )

    print("\nActivated EVENT_INDEX nodes:")
    for ev in result.candidate_events:
        print("  ", ev)

    print("\nCandidate operators (in order):")
    for op in result.candidate_operators:
        print(f"  {op.name} (score={op.last_applicability_score:.3f})")

    if not result.candidate_operators:
        print("\nNo operators returned by retrieval. (This is expected if you later add more operators and change scoring.)")
        # For the demo, we still show applying our demo_op explicitly:
        top_op = demo_op
    else:
        top_op = result.candidate_operators[0]

    # 7. Apply the underlying GraphOperator to the test grid
    predicted = apply_underlying_translation(test_grid, top_op)

    print("\nPredicted output grid (by applying underlying Translation):")
    print(np.array(predicted))

    # 8. For comparison, print the training output grid (the "correct" pattern)
    train_grid_out = task["train"][0]["output"]
    print("\nTraining output grid (target pattern):")
    print(np.array(train_grid_out))


if __name__ == "__main__":
    run_demo()

