"""Immutable machine-readable optimization-problem metadata.

The formal specification is a projection of the executable ``OptimizationProblem``.
Parameter defaults, categories, ranges, allowed values, and configurability are never
redeclared here; they are derived from the Step 11 parameter model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from optimization_problem import OptimizationProblem, UnsupportedRequirementRule, default_bonsai_packaging_problem


BONSAI_PROBLEM_ID = "bonsai_packaging"


@dataclass(frozen=True)
class ProblemInputSpec:
    """One named data input required by an optimization problem."""

    name: str
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class ProblemParameterSpec:
    """Read-only projection of one executable ``ParameterDefinition``."""

    name: str
    category: str
    default: Any
    value_type: str
    configurable: bool
    allowed_values: Tuple[Any, ...] | None
    minimum: float | None
    maximum: float | None
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "default": self.default,
            "value_type": self.value_type,
            "configurable": self.configurable,
            "allowed_values": list(self.allowed_values) if self.allowed_values is not None else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "description": self.description,
        }


@dataclass(frozen=True)
class OptimizationProblemSpec:
    """Small immutable contract used for application/LLM grounding.

    This object is descriptive metadata, not an executable optimization language.
    """

    problem_id: str
    display_name: str
    description: str
    inputs: Tuple[ProblemInputSpec, ...]
    parameters: Tuple[ProblemParameterSpec, ...]
    objective_summary: str
    decision_summary: str
    supported_capabilities: Tuple[str, ...]
    unsupported_change_categories: Tuple[str, ...]
    request_capability_rules: Tuple[UnsupportedRequirementRule, ...] = ()

    @property
    def parameter_map(self) -> Dict[str, ProblemParameterSpec]:
        return {parameter.name: parameter for parameter in self.parameters}

    def to_dict(self) -> Dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "display_name": self.display_name,
            "description": self.description,
            "inputs": [item.to_dict() for item in self.inputs],
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "objective_summary": self.objective_summary,
            "decision_summary": self.decision_summary,
            "supported_capabilities": list(self.supported_capabilities),
            "unsupported_change_categories": list(self.unsupported_change_categories),
        }


def project_problem_spec(
    problem: OptimizationProblem,
    *,
    problem_id: str,
    display_name: str,
    description: str,
    input_descriptions: Mapping[str, str] | None = None,
    objective_summary: str,
    decision_summary: str,
) -> OptimizationProblemSpec:
    """Project an executable problem into immutable descriptive metadata.

    Crucially, parameter metadata is read directly from ``problem.parameters`` so the
    formal contract cannot drift from executable validation rules through duplication.
    """
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be non-empty text")

    descriptions = dict(input_descriptions or {})
    inputs = tuple(
        ProblemInputSpec(name=name, description=descriptions.get(name, name))
        for name in problem.data_requirements
    )
    parameters = tuple(
        ProblemParameterSpec(
            name=name,
            category=definition.category,
            default=definition.default,
            value_type=definition.value_type.__name__,
            configurable=definition.configurable,
            allowed_values=definition.allowed_values,
            minimum=definition.minimum,
            maximum=definition.maximum,
            description=definition.description,
        )
        for name, definition in problem.parameters.items()
    )

    configurable = tuple(problem.capabilities.get("configurable", ()))
    supported_capabilities = tuple(
        f"parameter_override:{name}" for name in configurable
    )
    unsupported = tuple(problem.capabilities.get("unsupported", ()))

    return OptimizationProblemSpec(
        problem_id=problem_id.strip(),
        display_name=display_name,
        description=description,
        inputs=inputs,
        parameters=parameters,
        objective_summary=objective_summary,
        decision_summary=decision_summary,
        supported_capabilities=supported_capabilities,
        unsupported_change_categories=unsupported,
        request_capability_rules=tuple(problem.unsupported_requirement_rules),
    )


def bonsai_problem_spec(
    problem: OptimizationProblem | None = None,
) -> OptimizationProblemSpec:
    """Return the formal Bonsai contract projected from the executable problem."""
    executable = problem or default_bonsai_packaging_problem()
    return project_problem_spec(
        executable,
        problem_id=BONSAI_PROBLEM_ID,
        display_name="Bonsai Packaging Portfolio Optimization",
        description=(
            "Deterministic packaging-portfolio optimization using the existing Bonsai "
            "data, optimizer, validator, and comparison logic."
        ),
        input_descriptions={
            "catalog": "Product catalog input.",
            "specs": "Box specification input.",
            "operations": "Plant operations and baseline input.",
            "procurement": "Procurement pricing input used by the data audit.",
        },
        objective_summary="Minimize the encoded Bonsai packaging-plus-freight cost objective.",
        decision_summary=(
            "Choose and assign packaging options under the existing deterministic Bonsai formulation."
        ),
    )


__all__ = [
    "BONSAI_PROBLEM_ID",
    "OptimizationProblemSpec",
    "ProblemInputSpec",
    "ProblemParameterSpec",
    "bonsai_problem_spec",
    "project_problem_spec",
]
