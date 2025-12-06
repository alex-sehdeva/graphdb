# gsm/domains/arcagi/demos/multi_task_memory_demo.py
from __future__ import annotations

from typing import List

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


def build_shared_translate_operator() -> Operator:
    """
    Build a memory-level Operator that represents:
        Translation(dx=1, dy=0) followed by RecenterToOrigin().

    This is domain-agnostic over any 2D grid representation that uses
    coordinates as node ids.
    """
    graph_op = Translation(dx=1, dy=0).then(RecenterToOrigin())

    input_pattern = ConceptPattern()
    stats = OperatorStats(created_at=0, last_used_at=0)

    op = Operator(
        id="shared_translate_right_1",
        name="SharedTranslateRightBy1",
        kind=OperatorKind.STATIC_TRANSFORM,
        input_pattern=input_pattern,
        output_pattern=None,
        associated_actions=[],
        parameters={"dx": 1, "dy": 0, "recenter": True},
        stats=stats,
        underlying=graph_op,
    )
    return op


def apply_underlying_op_to_grid(grid, op: Operator):
    if op.underlying is None:
        raise ValueError("Operator has no underlying GraphOperator")
    g_in = grid_to_graph(grid)
    g_out = op.underlying(g_in)
    return graph_to_grid(g_out)


def run_multi_task_demo() -> None:
    # Two tasks that share the same abstract rule
    task_ids = ["demo_translate_right", "demo_translate_right_2"]

    mem = NxMemoryGraph()
    retrieval = RetrievalEngine(mem)

    # 1. Ingest both tasks into the same MemoryGraph
    timestep = 0
    for tid in task_ids:
        ingest_arc_task(mem, task_id=tid, start_timestep=timestep)
        task = load_arc_task_json(tid)
        timestep += len(task.get("train", [])) + len(task.get("test", []))

    # 2. Create ONE shared operator and link it to BOTH task contexts
    shared_op = build_shared_translate_operator()
    op_node_id = add_operator_node(mem, shared_op)
    print(f"Shared OPERATOR node: {op_node_id} ({shared_op.name})")

    for tid in task_ids:
        context_node = f"arc:ctx:task:{tid}"
        mem.add_edge(
            src=context_node,
            dst=op_node_id,
            type_=EdgeType.APPLIES_TO,
            attrs={"source": "multi_task_demo"},
        )

    # 3. For each task, build a Cue from its test input and retrieve
    for tid in task_ids:
        print("\n============================")
        print(f"Task: {tid}")
        task = load_arc_task_json(tid)
        test_grid = task["test"][0]["input"]
        train_out = task["train"][0]["output"]

        print("Test input grid:")
        print(np.array(test_grid))

        cue = Cue.from_arc_example(
            grid_in=test_grid,
            grid_out=None,
            task_id=tid,
            timestep=999,
        )

        result = retrieval.retrieve(
            cue=cue,
            goal=RetrievalGoal.RULE_SUGGESTION,
            k_results=5,
        )

        print("Activated EVENT_INDEX nodes:")
        for ev in result.candidate_events:
            print("  ", ev)

        print("Candidate operators (in order):")
        for op in result.candidate_operators:
            print(f"  {op.name} (score={op.last_applicability_score:.3f})")

        # Use the top candidate if available, else fall back to shared_op
        if result.candidate_operators:
            top_op = result.candidate_operators[0]
        else:
            print("  (No operators returned; using shared_op directly.)")
            top_op = shared_op

        predicted = apply_underlying_op_to_grid(test_grid, top_op)

        print("\nPredicted output grid:")
        print(np.array(predicted))

        print("Training output grid (target):")
        print(np.array(train_out))


if __name__ == "__main__":
    run_multi_task_demo()

