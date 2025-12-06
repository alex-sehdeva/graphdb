# gsm/domains/arcagi/memory_ingest.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gsm.memory.core import NodeId, NodeType, EdgeType, MemoryGraph
from gsm.domains.arcagi.convert import grid_to_graph

# Directory containing ARC task JSON files:
#   gsm/domains/arcagi/arc_tasks/<task_id>.json
ARC_TASKS_DIR = Path(__file__).resolve().parent / "arc_tasks"


def load_arc_task_json(task_id: str) -> dict:
    """
    Load an ARC task JSON from gsm/domains/arcagi/arc_tasks/<task_id>.json.

    The expected structure is the usual ARC format:
        {
          "train": [
            {"input": [[...]], "output": [[...]]},
            ...
          ],
          "test": [
            {"input": [[...]]},
            ...
          ]
        }
    """
    path = ARC_TASKS_DIR / f"{task_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ingest_arc_task(
    mem: MemoryGraph,
    task_id: str,
    start_timestep: int = 0,
) -> None:
    """
    Ingest *all* examples from a single ARC task JSON into the MemoryGraph.

    This will:
      - Load gsm/domains/arcagi/arc_tasks/<task_id>.json
      - For each train pair:
          - create an EVENT_INDEX node
          - attach cells as ENTITY nodes and colors as FEATURE nodes
      - For each test input:
          - same, but with grid_out=None and split="test"

    Args:
        mem:
            MemoryGraph backend (e.g., NxMemoryGraph).

        task_id:
            Name of the JSON file without extension, e.g. "00506d8b".

        start_timestep:
            Logical starting timestep for stats; each example increments
            this by 1. This is just a monotonically increasing counter;
            you can ignore it if you don't care about temporal ordering.

    Returns:
        None (but mutates `mem` in-place).
    """
    task = load_arc_task_json(task_id)

    t = start_timestep

    # Ingest training examples
    for i, pair in enumerate(task.get("train", [])):
        grid_in = pair["input"]
        grid_out = pair.get("output")
        example_id = f"train_{i}"

        ingest_arc_example(
            mem=mem,
            task_id=task_id,
            example_id=example_id,
            grid_in=grid_in,
            grid_out=grid_out,
            split="train",
            timestep=t,
        )
        t += 1

    # Ingest test examples
    for i, pair in enumerate(task.get("test", [])):
        grid_in = pair["input"]
        grid_out = pair.get("output")  # usually absent for test
        example_id = f"test_{i}"

        ingest_arc_example(
            mem=mem,
            task_id=task_id,
            example_id=example_id,
            grid_in=grid_in,
            grid_out=grid_out,
            split="test",
            timestep=t,
        )
        t += 1


def ingest_arc_example(
    mem: MemoryGraph,
    *,
    task_id: str,
    example_id: str,
    grid_in: Any,
    grid_out: Any | None = None,
    split: str = "train",
    timestep: int = 0,
) -> NodeId:
    """
    Ingest a single ARC example (train pair or test input) into MemoryGraph.

    Args:
        mem:
            MemoryGraph backend (e.g., NxMemoryGraph).

        task_id:
            ARC task identifier (e.g., "00506d8b").

        example_id:
            Unique id for this example within the task
            (e.g. "train_0", "train_1", "test_0").

        grid_in:
            Input grid for this example (2D list or numpy array of ints).

        grid_out:
            Output grid for this example (2D list or numpy array), or None
            if this is a test input without known output yet.

        split:
            "train" or "test" (or any label you prefer).

        timestep:
            Integer logical time; used to initialize stats like created_at.

    Returns:
        event_node:
            NodeId of the EVENT_INDEX node representing this example.
    """
    # 1. Convert input grid to a core.Graph using your existing converter
    g_in = grid_to_graph(grid_in)  # gsm.core.graph.Graph

    # 2. Create an EVENT_INDEX node for this example
    event_node: NodeId = f"arc:event:{task_id}:{example_id}"

    mem.add_node(
        node_id=event_node,
        type_=NodeType.EVENT_INDEX,
        embedding=[],  # later: pooled embedding over cells / objects
        attrs={
            "task_id": task_id,
            "example_id": example_id,
            "split": split,
            # Stash raw grids so higher layers can still access them.
            "grid_in": grid_in,
            "grid_out": grid_out,
        },
    )
    mem.update_stats(
        event_node,
        {
            "created_at": timestep,
            "last_used_at": timestep,
            "access_count": 0,
        },
    )

    # 3. Create or reuse a CONTEXT node for this task
    context_node: NodeId = f"arc:ctx:task:{task_id}"
    mem.add_node(
        node_id=context_node,
        type_=NodeType.CONTEXT,
        embedding=[],
        attrs={
            "task_id": task_id,
        },
    )
    mem.add_edge(
        src=event_node,
        dst=context_node,
        type_=EdgeType.IN_CONTEXT,
    )

    # 4. Optional TIME node (explicit temporal structure)
    time_node: NodeId = f"time:{timestep}"
    mem.add_node(
        node_id=time_node,
        type_=NodeType.TIME,
        embedding=[],
        attrs={"timestep": timestep},
    )
    mem.add_edge(
        src=event_node,
        dst=time_node,
        type_=EdgeType.BEFORE,  # event -> its timepoint
    )

    # 5. Ingest each cell of the input grid as an ENTITY + color FEATURE
    for (coord, attrs) in g_in.nodes.items():
        # coord is expected to be (row, col)
        r, c = coord
        color = int(attrs.get("color", 0))

        # ENTITY node for this cell
        cell_node: NodeId = f"arc:cell:{task_id}:{example_id}:{r},{c}"
        mem.add_node(
            node_id=cell_node,
            type_=NodeType.ENTITY,
            embedding=[],  # later: cell / local patch embedding
            attrs={
                "task_id": task_id,
                "example_id": example_id,
                "row": int(r),
                "col": int(c),
                "color": color,
            },
        )

        # link cell to event
        mem.add_edge(
            src=event_node,
            dst=cell_node,
            type_=EdgeType.PART_OF,
        )

        # FEATURE node for this color (reused across tasks/examples)
        color_node: NodeId = f"arc:feat:color:{color}"
        mem.add_node(
            node_id=color_node,
            type_=NodeType.FEATURE,
            embedding=[],
            attrs={
                "feature_type": "color",
                "value": color,
            },
        )

        # link cell -> color
        mem.add_edge(
            src=cell_node,
            dst=color_node,
            type_=EdgeType.HAS_FEATURE,
        )

    return event_node

