# gsm/domains/arcagi/demos/multi_task_promotion_demo.py
from __future__ import annotations

from typing import List

import numpy as np

from gsm.memory.nx_backend import NxMemoryGraph
from gsm.memory.retrieval import RetrievalEngine, RetrievalGoal
from gsm.memory.cues import Cue
from gsm.memory.core import EdgeType

from gsm.domains.arcagi.memory_ingest import ingest_arc_task, load_arc_task_json
from gsm.domains.arcagi.convert import grid_to_graph, graph_to_grid

from gsm.operators.grid import Translation, RecenterToOrigin
from gsm.rules.rule_program import RuleProgram
from gsm.rules.promotion import promote_rule_to_memory, PromotionConfig


def build_translate_program(name: str) -> RuleProgram:
    """
    Build a RuleProgram that applies Translation(dx=1, dy=0) then RecenterToOrigin().
    In a real setup, RuleSearcher would discover this; here we construct it
    directly to exercise the promotion / merge path.
    """
    ops = [
        Translation(dx=1, dy=0),
        RecenterToOrigin(),
    ]
    return RuleProgram.from_ops(name=name, ops=ops)


def apply_rule_program_to_grid(program: RuleProgram, grid) -> List[List[int]]:
    """
    Apply a RuleProgram (sequence of GraphOperators) to an ARC grid.
    """
    g = grid_to_graph(grid)
    g_out = program(g)
    return graph_to_grid(g_out)


def run_demo() -> None:
    task_ids = ["demo_translate_right", "demo_translate_right_2"]

    mem = NxMemoryGraph()
    retrieval = RetrievalEngine(mem)

    timestep = 0
    for tid in task_ids:
        ingest_arc_task(mem, task_id=tid, start_timestep=timestep)
        task_json = load_arc_task_json(tid)
        timestep += len(task_json.get("train", [])) + len(task_json.get("test", []))

    # Build a RuleProgram per task (pretending they were learned independently)
    programs = {
        tid: build_translate_program(name=f"{tid}_translate_right")
        for tid in task_ids
    }

    # Promote each RuleProgram and see what operator node we get
    op_nodes: dict[str, str] = {}

    now = 0
    for tid in task_ids:
        task_json = load_arc_task_json(tid)

        # EVENT_INDEX nodes solved by this rule: all train_* examples
        event_nodes_solved = [
            f"arc:event:{tid}:train_{i}" for i in range(len(task_json["train"]))
        ]

        context_node = f"arc:ctx:task:{tid}"

        op_node_id = promote_rule_to_memory(
            mem,
            task_id=tid,
            program=programs[tid],
            event_nodes_solved=event_nodes_solved,
            context_node=context_node,
            config=PromotionConfig(now=now, reward=1.0),
        )
        op_nodes[tid] = op_node_id
        print(f"Promoted rule for task {tid} -> operator node {op_node_id}")

        now += 10

    print("\nOperator nodes per task:")
    for tid, nid in op_nodes.items():
        print(f"  {tid}: {nid}")

    # Check if they are the same node
    unique_nodes = set(op_nodes.values())
    print("\nUnique operator nodes:", unique_nodes)
    if len(unique_nodes) == 1:
        print("✅ Both tasks share the SAME promoted operator node.")
    else:
        print("⚠ Tasks ended up with different operator nodes (signatures differed).")

    # Now, test retrieval + application for each task
    for tid in task_ids:
        print("\n============================")
        print(f"Task: {tid}")
        task_json = load_arc_task_json(tid)
        test_grid = task_json["test"][0]["input"]
        train_out = task_json["train"][0]["output"]

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

        print("Candidate operators (in order):")
        for op in result.candidate_operators:
            print(f"  {op.name} (score={op.last_applicability_score:.3f})")

        if result.candidate_operators:
            top_op = result.candidate_operators[0]
            print(f"\nUsing top candidate: {top_op.name}")
            # top_op.underlying is the RuleProgram we wrapped
            program = top_op.underlying
            predicted = apply_rule_program_to_grid(program, test_grid)
        else:
            print("\nNo operators returned by retrieval; using per-task program directly.")
            predicted = apply_rule_program_to_grid(programs[tid], test_grid)

        print("\nPredicted output grid:")
        print(np.array(predicted))

        print("Training output grid (target):")
        print(np.array(train_out))


if __name__ == "__main__":
    run_demo()

