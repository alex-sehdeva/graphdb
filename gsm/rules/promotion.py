# gsm/rules/promotion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, List, Optional, Tuple

from gsm.memory.core import MemoryGraph, NodeId, EdgeType, OperatorKind, NodeType
from gsm.memory.patterns import ConceptPattern
from gsm.memory.operators import Operator, OperatorStats, add_operator_node

from gsm.rules.rule_program import RuleProgram, RuleStep

from gsm.operators.attribute_map import AttributeMap
from gsm.operators.grid import Translation, RecenterToOrigin

def attribute_map_signature(attr_name: str, mapping: dict) -> tuple:
    """
    Structural signature for an AttributeMap (and thus Recolor).

    We ignore *which* concrete values are used and only capture the structure:

      - identity:
          every key maps to itself (or mapping is empty)
          -> ("AttributeMap", attr_name, "identity")

      - permutation / renaming:
          mapping is one-to-one on its changed keys (no collisions),
          even if it only acts on a subset of the alphabet
          -> ("AttributeMap", attr_name, "perm")

      - merge / partition:
          mapping collapses multiple old values to the same new value
          -> ("AttributeMap", attr_name, "merge", (sizes...))

        where (sizes...) is the sorted tuple of preimage sizes per target.

    This is invariant under:
      - renaming of the alphabet (color ids, token ids)
      - order of keys in the dict
    """
    if not mapping:
        return ("AttributeMap", attr_name, "identity")

    # Keep only keys that actually change
    changed = {k: v for k, v in mapping.items() if k != v}
    if not changed:
        return ("AttributeMap", attr_name, "identity")

    # Build target -> [sources...] preimage groups
    preimages: dict = {}
    for src, tgt in changed.items():
        preimages.setdefault(tgt, []).append(src)

    # Check if this is a pure permutation on the changed subset:
    #   - all targets distinct (each changed src has its own tgt)
    #   - (values can overlap unchanged keys; we don't care)
    all_targets = list(preimages.keys())
    if all(len(srcs) == 1 for srcs in preimages.values()):
        # One-to-one on the changed subset → treat as generic permutation
        return ("AttributeMap", attr_name, "perm")

    # Otherwise, we have merges / many-to-one mapping.
    # Capture only the sorted list of preimage sizes.
    sizes = sorted(len(srcs) for srcs in preimages.values())
    return ("AttributeMap", attr_name, "merge", tuple(sizes))


@dataclass
class PromotionConfig:
    """
    Configuration for promoting a RuleProgram into a memory.Operator.
    """
    now: int = 0
    reward: float = 1.0
    operator_id: Optional[str] = None
    label_suffix: Optional[str] = None


# --------- Signature logic ---------

def normalize_program_ops(program: RuleProgram) -> list:
    """
    Return a normalized list of operators for signature computation.

    In particular:
      - Consecutive Translation ops (with same use_attr) are collapsed
        into a single Translation with summed (dx, dy).
      - Zero translations (dx=0, dy=0) are dropped.
      - Other ops (RecenterToOrigin, AttributeMap, etc.) are left as-is
        and break translation chains.

    This makes programs equivalent up to how they compose translations:
      Translation(1,0) ∘ Translation(1,0)  and  Translation(2,0)
    yield the same signature.
    """
    normalized: list = []
    pending_dx = 0
    pending_dy = 0
    pending_use_attr = None

    def flush_pending_translation():
        nonlocal pending_dx, pending_dy, pending_use_attr, normalized
        if pending_dx == 0 and pending_dy == 0:
            # identity; nothing to add
            return
        normalized.append(Translation(dx=pending_dx, dy=pending_dy, use_attr=pending_use_attr))
        pending_dx = 0
        pending_dy = 0
        pending_use_attr = None

    for step in program.steps:
        op = step.operator

        if isinstance(op, Translation):
            # If we are chaining translations with the same mode (use_attr),
            # we can accumulate (dx, dy) before flushing.
            if pending_use_attr is None or pending_use_attr == op.use_attr:
                pending_dx += op.dx
                pending_dy += op.dy
                pending_use_attr = op.use_attr
            else:
                # Different mode: flush previous chain, start new one
                flush_pending_translation()
                pending_dx = op.dx
                pending_dy = op.dy
                pending_use_attr = op.use_attr

        else:
            # Any non-Translation breaks the chain
            flush_pending_translation()
            normalized.append(op)

    # Flush trailing translation chain
    flush_pending_translation()

    return normalized

def rule_program_signature(program: RuleProgram) -> Hashable:
    """
    Compute a structural signature for a RuleProgram.

    Two RulePrograms should have the same signature if they represent
    the *same kind* of transformation up to nuisance details like:

      - which concrete colors / symbols are used
      - which exact ids are in a permutation
      - how many individual Translation steps were used to get the
        same net displacement

    We achieve this by:
      - normalizing chains of Translation into a single net Translation
      - classifying AttributeMap by structure (identity / perm / merge)
      - keeping RecenterToOrigin as a step
    """
    sig_steps: List[Tuple[Any, ...]] = []

    # Normalize operator sequence first (collapsing Translation chains)
    ops = normalize_program_ops(program)

    for op in ops:
        # Grid translation: keep net dx, dy (direction and distance)
        if isinstance(op, Translation):
            sig_steps.append(("Translation", op.dx, op.dy))

        # Grid recenter: just the fact that we recenter
        elif isinstance(op, RecenterToOrigin):
            sig_steps.append(("RecenterToOrigin",))

        # Generic attribute map (including Recolor)
        elif isinstance(op, AttributeMap):
            sig_steps.append(attribute_map_signature(op.attr_name, op.mapping))

        else:
            # Fallback: use class name + repr
            sig_steps.append((op.__class__.__name__, repr(op)))

    return tuple(sig_steps)


def find_operator_by_signature(
    mem: MemoryGraph,
    signature: Hashable,
) -> Optional[NodeId]:
    """
    Search the MemoryGraph for an OPERATOR node with a matching signature.
    """
    for n in mem.nodes_of_type(NodeType.OPERATOR):
        sig = mem.get_attr(n, "signature")
        if sig == signature:
            return n
    return None


# --------- Promotion / merge core ---------


def promote_rule_to_memory(
    mem: MemoryGraph,
    *,
    task_id: Optional[str],
    program: RuleProgram,
    event_nodes_solved: Iterable[NodeId],
    context_node: Optional[NodeId] = None,
    config: PromotionConfig | None = None,
) -> NodeId:
    """
    Promote a learned RuleProgram into a concept-level Operator node
    in the MemoryGraph, merging with an existing one if the structural
    signature matches.

    Args:
        mem:
            MemoryGraph backend.

        task_id:
            Domain-specific task identifier (e.g., ARC task id), or None.

        program:
            RuleProgram that fits training pairs for this task.

        event_nodes_solved:
            EVENT_INDEX node ids that this program explains / solves.

        context_node:
            CONTEXT node id for the task (e.g., "arc:ctx:task:{task_id}").

        config:
            PromotionConfig with timestamps / reward / ids.

    Returns:
        operator_node_id:
            OPERATOR node id (possibly reused from a previous promotion).
    """
    if config is None:
        config = PromotionConfig()

    now = config.now
    reward = config.reward
    event_nodes_solved = list(event_nodes_solved)

    # 1. Compute structural signature
    signature = rule_program_signature(program)

    # 2. See if an operator with this signature already exists
    existing_node = find_operator_by_signature(mem, signature)
    if existing_node is not None:
        # Merge into existing operator node
        _merge_rule_into_existing_operator(
            mem=mem,
            op_node_id=existing_node,
            event_nodes_solved=event_nodes_solved,
            context_node=context_node,
            now=now,
            reward=reward,
        )
        return existing_node

    # 3. Otherwise, create a new Operator and node

    # Derive operator id & name
    if config.operator_id is not None:
        op_id = config.operator_id
    else:
        if task_id is not None:
            op_id = f"rule:{task_id}:{program.name}"
        else:
            op_id = f"rule:{program.name}"

    if task_id is not None:
        base_name = f"Rule[{task_id}]::{program.name}"
    else:
        base_name = f"Rule::{program.name}"

    if config.label_suffix:
        name = f"{base_name}::{config.label_suffix}"
    else:
        name = base_name

    # Stats: assume one successful 'use' per solved event
    success_count = len(event_nodes_solved)
    use_count = success_count

    stats = OperatorStats(
        created_at=now,
        last_used_at=now,
        use_count=use_count,
        success_count=success_count,
        avg_reward=reward,
    )

    input_pattern = ConceptPattern()

    memory_op = Operator(
        id=op_id,
        name=name,
        kind=OperatorKind.STATIC_TRANSFORM,
        input_pattern=input_pattern,
        output_pattern=None,
        associated_actions=[],
        parameters={
            "source": "RuleProgram",
            "num_steps": len(program.steps),
        },
        stats=stats,
        underlying=program,
    )

    op_node_id = add_operator_node(mem, memory_op, signature=signature)

    # Wire context -> operator
    if context_node is not None:
        mem.add_edge(
            src=context_node,
            dst=op_node_id,
            type_=EdgeType.APPLIES_TO,
            attrs={"source": "promotion"},
        )

    # Wire operator -> solved events
    for ev in event_nodes_solved:
        mem.add_edge(
            src=op_node_id,
            dst=ev,
            type_=EdgeType.LEADS_TO,
            attrs={"source": "promotion"},
        )

    return op_node_id


def _merge_rule_into_existing_operator(
    mem: MemoryGraph,
    *,
    op_node_id: NodeId,
    event_nodes_solved: List[NodeId],
    context_node: Optional[NodeId],
    now: int,
    reward: float,
) -> None:
    """
    Update stats and edges when a new task's RuleProgram matches an
    existing operator's signature.
    """
    # Fetch the existing Operator object
    op_obj = mem.get_attr(op_node_id, "operator_obj")
    if not isinstance(op_obj, Operator):
        # If something went wrong, just return; you may want logging here.
        return

    # Update stats: treat each solved event as another successful 'use'
    additional_uses = len(event_nodes_solved)
    s = op_obj.stats

    old_use = s.use_count
    old_avg = s.avg_reward

    s.use_count += additional_uses
    s.success_count += additional_uses
    s.last_used_at = now

    # Update avg_reward as a weighted average
    if s.use_count > 0:
        total_reward = old_avg * old_use + reward * additional_uses
        s.avg_reward = total_reward / s.use_count

    # Mirror updated stats into MemoryGraph node stats
    mem.update_stats(op_node_id, {
        "last_used_at": s.last_used_at,
        "use_count": s.use_count,
        "success_count": s.success_count,
        "avg_reward": s.avg_reward,
    })

    # Wire context -> operator (if new)
    if context_node is not None:
        mem.add_edge(
            src=context_node,
            dst=op_node_id,
            type_=EdgeType.APPLIES_TO,
            attrs={"source": "promotion_merge"},
        )

    # Wire operator -> each new solved event
    for ev in event_nodes_solved:
        mem.add_edge(
            src=op_node_id,
            dst=ev,
            type_=EdgeType.LEADS_TO,
            attrs={"source": "promotion_merge"},
        )

