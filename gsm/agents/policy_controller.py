"""
unimplemented, will depend on:

gsm.memory.RetrievalEngine
gsm.domains.arcagi (or other domains)

but not on low-level core internals directly

"""
# in agents/policy_controller.py or in a demo solver

from gsm.memory.operators import Operator
from gsm.memory.core import MemoryGraph, NodeId

def record_operator_outcome(
    mem: MemoryGraph,
    op_node_id: NodeId,
    *,
    success: bool,
    reward: float,
    now: int,
) -> None:
    op: Operator = mem.get_attr(op_node_id, "operator_obj")

    if success:
        op.stats.update_success(reward=reward, now=now)
    else:
        op.stats.update_failure(now=now)

    mem.update_stats(op_node_id, op.stats.to_stats_dict())

