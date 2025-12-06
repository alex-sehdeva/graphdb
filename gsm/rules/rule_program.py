# gsm/rules/rule_program.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

from gsm.core.graph import Graph
from gsm.operators.base import GraphOperator


@dataclass
class RuleStep:
    """
    One step in a rule program: a single operator plus optional metadata.
    """
    name: str
    operator: GraphOperator
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleProgram:
    """
    A compositional rule represented as an ordered list of RuleSteps.

    - Acts like a function Graph -> Graph (by applying each operator in turn).
    - Is introspectable (describe()) and serializable (to_dict()).
    """
    name: str
    steps: List[RuleStep]

    def __call__(self, graph: Graph) -> Graph:
        g = graph
        for step in self.steps:
            g = step.operator(g)
        return g

    # ---------- convenience constructors ----------

    @classmethod
    def from_ops(cls, name: str, ops: List[GraphOperator]) -> "RuleProgram":
        """
        Build a RuleProgram from a bare list of operators, giving default step names.
        """
        steps = [
            RuleStep(name=f"op_{i}", operator=op, metadata={})
            for i, op in enumerate(ops)
        ]
        return cls(name=name, steps=steps)

    # ---------- introspection / serialization ----------

    def describe(self) -> str:
        """
        Human-readable description for debugging / interpretability.
        """
        lines = [f"RuleProgram: {self.name}"]
        for i, step in enumerate(self.steps):
            meta_str = ", ".join(f"{k}={v}" for k, v in step.metadata.items())
            if meta_str:
                lines.append(f"  [{i}] {step.name} :: {type(step.operator).__name__} ({meta_str})")
            else:
                lines.append(f"  [{i}] {step.name} :: {type(step.operator).__name__}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a simple JSON-like dict (without encoding the full operator internals).
        """
        return {
            "name": self.name,
            "steps": [
                {
                    "name": s.name,
                    "operator_type": type(s.operator).__name__,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
        }

    def __repr__(self) -> str:
        return f"RuleProgram(name={self.name!r}, steps={len(self.steps)})"

