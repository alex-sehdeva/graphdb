# memory/immune.py
from gsm.memory.core import MemoryGraph, NodeType

class OperatorImmuneSystem:
    """
    Immune-style homeostasis over OPERATOR nodes:

      - decays strength over time
      - marks low-precision operators as inactive (soft delete)
    """

    def __init__(
        self,
        mem: MemoryGraph,
        *,
        min_precision: float = 0.6,
        decay_per_step: float = 0.001,
        min_uses_for_judgment: int = 10,
    ) -> None:
        self.mem = mem
        self.min_precision = min_precision
        self.decay_per_step = decay_per_step
        self.min_uses_for_judgment = min_uses_for_judgment
        self.step_count = 0

    def step(self) -> None:
        """Call this once per training step / env step / outer loop iteration."""
        self.step_count += 1

        for node_id in self.mem.nodes_of_type(NodeType.OPERATOR):
            stats = self.mem.node_stats(node_id)

            use_count = stats.get("use_count", 0)
            success_count = stats.get("success_count", 0)
            strength = stats.get("strength", 1.0)

            # decay strength
            strength *= (1.0 - self.decay_per_step)

            precision = success_count / use_count if use_count > 0 else 0.0

            updates = {
                "strength": strength,
                "precision": precision,
            }

            # negative selection: deactivate consistently bad operators
            if (
                use_count >= self.min_uses_for_judgment
                and precision < self.min_precision
            ):
                updates["active"] = False

            self.mem.update_stats(node_id, updates)

