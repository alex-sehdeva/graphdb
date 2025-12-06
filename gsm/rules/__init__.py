"""
Rule system for the Graph Substrate Machine (gsm).

This package defines:
- RuleProgram: a compositional program over GraphOperators
- RuleSearcher: a basic search engine over a space of RulePrograms

These tools operate on the core.Graph abstraction and the operator
algebra defined in gsm/operators.
"""

from .rule_program import RuleProgram, RuleStep
from .rule_search import RuleSearcher, RuleSearchConfig
