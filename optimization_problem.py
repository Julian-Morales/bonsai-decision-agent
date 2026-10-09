"""Minimal optimization-problem definition for the Bonsai decision agent.

The definition owns parameter metadata and validation. It does not implement any
optimization mathematics; the existing Bonsai engine remains the computation layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Tuple
import re


class OptimizationProblemError(ValueError):
    """Raised when an optimization parameter override is invalid."""




UnsupportedRequirementMatcher = Callable[[str], Mapping[str, object] | None]


@dataclass(frozen=True)
class UnsupportedRequirementRule:
    """Private request-level capability rule owned by one executable problem.

    The matcher is deliberately deterministic and narrow.  It returns compact diagnostic
    metadata when the known unsupported requirement is present, otherwise ``None``.
    These rules are application safety metadata; they are not optimizer parameters and are
    not exposed as public LLM tools.
    """

    capability_id: str
    category: str
    description: str
    reason_code: str
    safe_explanation: str
    affected_action_types: Tuple[str, ...]
    matcher: UnsupportedRequirementMatcher = field(repr=False, compare=False)

    def match(self, text: str) -> Mapping[str, object] | None:
        if not isinstance(text, str) or not text.strip():
            return None
        matched = self.matcher(text)
        return dict(matched) if isinstance(matched, Mapping) else None


def _match_bonsai_max_box_types(text: str) -> Mapping[str, object] | None:
    """Recognize conservative formulations of a required maximum box-type count.

    This is intentionally *not* a generic constraint parser.  The patterns are bounded around
    the known business concept and extract the numeric bound only for diagnostics.
    """

    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    patterns = (
        r"\bat most\s+(?P<bound>\d+)\s+(?:different\s+)?box\s+types?\b",
        r"\bno more than\s+(?P<bound>\d+)\s+(?:different\s+)?box\s+types?\b",
        r"\bmaximum(?:\s+of)?\s+(?P<bound>\d+)\s+(?:different\s+)?box\s+types?\b",
        r"\blimit\s+(?:the\s+number\s+of\s+)?box\s+types?\s+to\s+(?P<bound>\d+)\b",
        r"\blimit\s+(?:the\s+)?solution\s+to\s+(?P<bound>\d+)\s+(?:different\s+)?box\s+types?\b",
        r"\buse\s+(?P<bound>\d+)\s+or\s+fewer\s+(?:different\s+)?box\s+types?\b",
        r"\buse\s+at most\s+(?P<bound>\d+)\s+(?:different\s+)?box\s+types?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is not None:
            return {"max_box_types": int(match.group("bound"))}
    return None


def bonsai_max_box_types_rule() -> UnsupportedRequirementRule:
    return UnsupportedRequirementRule(
        capability_id="max_box_types",
        category="constraint",
        description=(
            "Require an upper bound on the number of distinct box types in the optimized solution."
        ),
        reason_code="unsupported_required_capability",
        safe_explanation=(
            "The current Bonsai optimization problem cannot enforce a maximum number of distinct box types."
        ),
        affected_action_types=("run_optimization", "run_scenario", "create_scenario"),
        matcher=_match_bonsai_max_box_types,
    )

@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    category: str  # business | solver
    default: Any
    value_type: type
    configurable: bool = True
    allowed_values: Tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def validate(self, value: Any) -> Any:
        if type(value) is not self.value_type:
            raise OptimizationProblemError(
                f"Parameter '{self.name}' has invalid type; expected {self.value_type.__name__}."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise OptimizationProblemError(f"Parameter '{self.name}' must be finite.")
        if self.allowed_values is not None and value not in self.allowed_values:
            raise OptimizationProblemError(
                f"Parameter '{self.name}' has unsupported value {value!r}; "
                f"allowed values are {self.allowed_values}."
            )
        if self.minimum is not None and value < self.minimum:
            raise OptimizationProblemError(
                f"Parameter '{self.name}' must be >= {self.minimum}."
            )
        if self.maximum is not None and value > self.maximum:
            raise OptimizationProblemError(
                f"Parameter '{self.name}' must be <= {self.maximum}."
            )
        return value


@dataclass(frozen=True)
class OptimizationProblem:
    name: str
    parameters: Mapping[str, ParameterDefinition]
    values: Mapping[str, Any]
    data_requirements: Tuple[str, ...]
    capabilities: Mapping[str, Tuple[str, ...]]
    unsupported_requirement_rules: Tuple[UnsupportedRequirementRule, ...] = ()

    def with_overrides(self, overrides: Mapping[str, Any]) -> "OptimizationProblem":
        values = dict(self.values)
        for name, value in overrides.items():
            definition = self.parameters.get(name)
            if definition is None:
                raise OptimizationProblemError(f"Unknown optimization parameter: {name}")
            if not definition.configurable:
                raise OptimizationProblemError(f"Parameter '{name}' is not configurable.")
            values[name] = definition.validate(value)
        return OptimizationProblem(
            name=self.name,
            parameters=self.parameters,
            values=values,
            data_requirements=self.data_requirements,
            capabilities=self.capabilities,
            unsupported_requirement_rules=self.unsupported_requirement_rules,
        )

    @property
    def business_parameters(self) -> Dict[str, Any]:
        return {
            name: self.values[name]
            for name, definition in self.parameters.items()
            if definition.category == "business"
        }

    @property
    def solver_parameters(self) -> Dict[str, Any]:
        return {
            name: self.values[name]
            for name, definition in self.parameters.items()
            if definition.category == "solver"
        }


def default_bonsai_packaging_problem() -> OptimizationProblem:
    """Return the supported Bonsai packaging problem definition."""
    parameters = {
        "thickness": ParameterDefinition(
            "thickness", "business", 3.0, float, allowed_values=(3.0, 4.5, 5.0),
            description="Uniform box thickness supported by the existing Bonsai model.",
        ),
        "pallet_cost": ParameterDefinition(
            "pallet_cost", "business", 150.0, float, minimum=1e-9,
            description="Uniform pallet cost applied by the existing freight model.",
        ),
        "milp_time": ParameterDefinition(
            "milp_time", "solver", 30.0, float, minimum=1e-9,
            description="MILP time limit in seconds.",
        ),
        "mip_gap": ParameterDefinition(
            "mip_gap", "solver", 5e-4, float, minimum=0.0, maximum=1.0,
            description="Relative MILP gap passed to the solver.",
        ),
        "extra_candidates": ParameterDefinition(
            "extra_candidates", "solver", 40, int, minimum=0,
        ),
        "lns_time": ParameterDefinition(
            "lns_time", "solver", 0.2, float, minimum=1e-9,
        ),
        "lns_rounds": ParameterDefinition(
            "lns_rounds", "solver", 2, int, minimum=0,
        ),
        "target_count": ParameterDefinition(
            "target_count", "solver", 40, int, minimum=0,
        ),
        "target_time": ParameterDefinition(
            "target_time", "solver", 0.05, float, minimum=1e-9,
        ),
    }
    values = {name: definition.default for name, definition in parameters.items()}
    return OptimizationProblem(
        name="bonsai_packaging",
        parameters=parameters,
        values=values,
        data_requirements=("catalog", "specs", "operations", "procurement"),
        capabilities={
            "configurable": tuple(parameters),
            "unsupported": ("demand", "freight_model", "max_box_types", "arbitrary_constraints"),
        },
        unsupported_requirement_rules=(bonsai_max_box_types_rule(),),
    )


class BonsaiPackagingProblem(OptimizationProblem):
    """Named concrete Bonsai problem instance with default parameters."""

    def __init__(self) -> None:
        default = default_bonsai_packaging_problem()
        super().__init__(
            name=default.name,
            parameters=default.parameters,
            values=default.values,
            data_requirements=default.data_requirements,
            capabilities=default.capabilities,
            unsupported_requirement_rules=default.unsupported_requirement_rules,
        )
