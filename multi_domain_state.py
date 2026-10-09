"""Session-local state scoping for explicitly registered optimization problems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from problem_adapter import OptimizationProblemAdapter
from problem_registry import ProblemNotRegisteredError, ProblemRegistry
from run_history import RunHistoryStore
from scenario_store import ScenarioStore
from solution_store import SolutionStore


@dataclass
class ProblemSessionState:
    """Existing application stores scoped to exactly one registered problem."""

    adapter: OptimizationProblemAdapter
    solution_store: SolutionStore
    scenario_store: ScenarioStore
    run_history_store: RunHistoryStore


class MultiDomainSessionState:
    """One session containing isolated per-problem state and one active problem id."""

    def __init__(
        self,
        registry: ProblemRegistry,
        *,
        active_problem_id: str,
        solution_stores: Mapping[str, SolutionStore] | None = None,
        scenario_stores: Mapping[str, ScenarioStore] | None = None,
        run_history_stores: Mapping[str, RunHistoryStore] | None = None,
    ) -> None:
        self.registry = registry
        self._states: Dict[str, ProblemSessionState] = {}
        supplied_solutions = dict(solution_stores or {})
        supplied_scenarios = dict(scenario_stores or {})
        supplied_history = dict(run_history_stores or {})
        unknown = (set(supplied_solutions) | set(supplied_scenarios) | set(supplied_history)) - set(registry.problem_ids)
        if unknown:
            raise ProblemNotRegisteredError(
                f"State supplied for unregistered problem(s): {sorted(unknown)}"
            )
        for problem_id in registry.problem_ids:
            adapter = registry.get(problem_id)
            scenario_store = supplied_scenarios.get(problem_id) or ScenarioStore(
                problem=adapter.problem,
                problem_id=problem_id,
            )
            history_store = supplied_history.get(problem_id) or RunHistoryStore(
                default_problem_id=problem_id,
            )
            solution_store = supplied_solutions.get(problem_id) or SolutionStore()
            if scenario_store.problem_id != problem_id:
                raise ValueError("ScenarioStore problem_id does not match registry entry.")
            self._states[problem_id] = ProblemSessionState(
                adapter=adapter,
                solution_store=solution_store,
                scenario_store=scenario_store,
                run_history_store=history_store,
            )
        self._active_problem_id = ""
        self.select(active_problem_id)

    @property
    def active_problem_id(self) -> str:
        return self._active_problem_id

    @property
    def active(self) -> ProblemSessionState:
        return self._states[self._active_problem_id]

    def state_for(self, problem_id: str) -> ProblemSessionState:
        if problem_id not in self.registry:
            raise ProblemNotRegisteredError(f"Problem is not registered: {problem_id}")
        return self._states[problem_id]

    def select(self, problem_id: str) -> ProblemSessionState:
        if problem_id not in self.registry:
            raise ProblemNotRegisteredError(f"Problem is not registered: {problem_id}")
        self._active_problem_id = problem_id
        return self.active


__all__ = ["MultiDomainSessionState", "ProblemSessionState"]
