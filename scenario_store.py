"""Session-local named scenario definitions for one optimization problem.

A scenario is only a validated set of ``OptimizationProblem`` parameter overrides plus
an optional reference to a validated solution owned by ``SolutionStore``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Dict, Mapping

from optimization_problem import OptimizationProblem, OptimizationProblemError, default_bonsai_packaging_problem
from problem_definition import BONSAI_PROBLEM_ID


class ScenarioStoreError(Exception):
    """Base class for predictable scenario-registry failures."""


class ScenarioNotFoundError(ScenarioStoreError):
    """Raised when a scenario handle is not registered."""


class ScenarioExistsError(ScenarioStoreError):
    """Raised when a scenario handle/name already exists."""


class InvalidScenarioNameError(ScenarioStoreError):
    """Raised when a user-visible scenario name cannot be represented safely."""


@dataclass(frozen=True)
class ScenarioDefinition:
    handle: str
    name: str
    problem_id: str
    parameter_overrides: Mapping[str, object]
    _base_problem: OptimizationProblem = field(repr=False, compare=False)

    def resolve_problem(self) -> OptimizationProblem:
        return self._base_problem.with_overrides(self.parameter_overrides)


@dataclass(frozen=True)
class ScenarioRecord:
    definition: ScenarioDefinition
    problem_id: str
    solution_handle: str | None = None


_SAFE_HANDLE = re.compile(r"^[a-z0-9_]+$")


def scenario_handle_from_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise InvalidScenarioNameError("Scenario name must be non-empty text.")
    slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    if not slug or not _SAFE_HANDLE.fullmatch(slug):
        raise InvalidScenarioNameError("Scenario name does not produce a safe handle.")
    return f"scenario_{slug}"


def scenario_solution_handle(scenario_handle: str) -> str:
    if not isinstance(scenario_handle, str) or not scenario_handle.startswith("scenario_"):
        raise InvalidScenarioNameError("Invalid scenario handle.")
    if not _SAFE_HANDLE.fullmatch(scenario_handle):
        raise InvalidScenarioNameError("Invalid scenario handle.")
    return f"scenario_solution_{scenario_handle[len('scenario_') :]}"


class ScenarioStore:
    """In-memory scenario registry for one agent/application session.

    The executable problem is injected so the store is reusable by another adapter.
    Bonsai defaults remain only as a backward-compatible construction path.
    """

    def __init__(
        self,
        *,
        problem: OptimizationProblem | None = None,
        problem_id: str = BONSAI_PROBLEM_ID,
    ) -> None:
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("ScenarioStore problem_id must be non-empty text.")
        self._problem = problem or default_bonsai_packaging_problem()
        self._problem_id = problem_id.strip()
        self._records: Dict[str, ScenarioRecord] = {}
        self._name_to_handle: Dict[str, str] = {}

    @property
    def problem_id(self) -> str:
        return self._problem_id

    def create(self, name: str, parameter_overrides: Mapping[str, object]) -> ScenarioRecord:
        handle = scenario_handle_from_name(name)
        key = name.strip().casefold()
        if handle in self._records or key in self._name_to_handle:
            raise ScenarioExistsError(f"Scenario already exists: {name}")

        # Delegate all supported-parameter/type/range rules to OptimizationProblem.
        problem = self._problem.with_overrides(dict(parameter_overrides))
        normalized = {
            parameter: problem.values[parameter]
            for parameter in parameter_overrides
        }
        definition = ScenarioDefinition(
            handle=handle,
            name=name.strip(),
            problem_id=self._problem_id,
            parameter_overrides=MappingProxyType(dict(normalized)),
            _base_problem=self._problem,
        )
        record = ScenarioRecord(
            definition=definition,
            problem_id=self._problem_id,
            solution_handle=None,
        )
        self._records[handle] = record
        self._name_to_handle[key] = handle
        return record

    def get(self, handle_or_name: str) -> ScenarioRecord:
        if not isinstance(handle_or_name, str):
            raise ScenarioNotFoundError("Scenario identifier must be text.")
        key = handle_or_name.strip()
        handle = key if key in self._records else self._name_to_handle.get(key.casefold())
        if handle is None or handle not in self._records:
            raise ScenarioNotFoundError(f"Scenario is not registered: {handle_or_name}")
        return self._records[handle]

    def set_solution(self, scenario_handle: str, solution_handle: str) -> ScenarioRecord:
        record = self.get(scenario_handle)
        updated = replace(record, solution_handle=solution_handle)
        self._records[record.definition.handle] = updated
        return updated

    def list(self) -> tuple[ScenarioRecord, ...]:
        return tuple(self._records.values())

    def __contains__(self, handle_or_name: object) -> bool:
        if not isinstance(handle_or_name, str):
            return False
        try:
            self.get(handle_or_name)
        except ScenarioNotFoundError:
            return False
        return True


__all__ = [
    "InvalidScenarioNameError",
    "ScenarioDefinition",
    "ScenarioExistsError",
    "ScenarioNotFoundError",
    "ScenarioRecord",
    "ScenarioStore",
    "ScenarioStoreError",
    "scenario_handle_from_name",
    "scenario_solution_handle",
]
