# gsm/memory/cues.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core import NodeId


@dataclass
class Cue:
    """
    A Cue is the 'question' you ask the MemoryGraph.

    It bundles:
      - a task identifier (ARC task id, game level id, etc.)
      - an optional reference to a local state graph (if stored)
      - an optional embedding (for similarity-based seed selection)
      - a generic 'extra' dict for domain-specific fields.

    Examples of extra fields:
      - ARC:  {"grid_in": grid, "grid_out": grid_or_none, "timestep": 42}
      - Game: {"obs_t": obs, "action_space": action_space, "timestep": 137}
    """
    task_id: Optional[str] = None
    local_graph_id: Optional[NodeId] = None
    embedding: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # --------- ARC-specific constructors ---------

    @classmethod
    def from_arc_example(
        cls,
        grid_in: Any,
        grid_out: Any | None = None,
        task_id: Optional[str] = None,
        timestep: Optional[int] = None,
    ) -> "Cue":
        """
        Build a Cue from an ARC training or test example.

        grid_in / grid_out can be numpy arrays or nested lists; they are
        passed through unchanged and only used by downstream code that
        knows how to handle them.
        """
        extra: Dict[str, Any] = {
            "grid_in": grid_in,
            "grid_out": grid_out,
        }
        if timestep is not None:
            extra["timestep"] = timestep

        return cls(
            task_id=task_id,
            local_graph_id=None,   # can be set later if you store it in MemoryGraph
            embedding=None,        # can be filled by an encoder/embedding model
            extra=extra,
        )

    # --------- Env / video-game-style constructors ---------

    @classmethod
    def from_env_step(
        cls,
        obs_t: Any,
        action_space: Any,
        task_id: Optional[str] = None,
        timestep: Optional[int] = None,
    ) -> "Cue":
        """
        Build a Cue from a single environment step (ARC-AGI 3 style).

        obs_t:
          - any observation object (grid, image, dict, etc.)
        action_space:
          - description of available actions (e.g., gym-like space, list of moves)
        """
        extra: Dict[str, Any] = {
            "obs_t": obs_t,
            "action_space": action_space,
        }
        if timestep is not None:
            extra["timestep"] = timestep

        return cls(
            task_id=task_id,
            local_graph_id=None,
            embedding=None,
            extra=extra,
        )

