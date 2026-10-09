"""Structured session-local planning records for Step 17.

This module contains orchestration metadata only. It does not import deterministic
optimizers, domain adapters, solution rows, or source datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple
import re


PLAN_STEP_STATUSES = (
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
    "pending_approval",
    "recoverable_failed",
    "rejected",
    "outcome_unknown",
    "blocked_unsupported_requirement",
)
ARGUMENT_BINDING_TYPES = ("literal", "deferred", "session")
SESSION_ENTITY_TYPES = ("scenario", "solution", "run")
DEFERRED_SELECTOR_TYPES = ("direct_field", "collection_match", "unique_collection_item")
MAX_PLAN_STEPS = 12
_ARGUMENT_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_SAFE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# Typed, allowlisted compact-output contracts. Planner-authored JSON paths/expressions are
# intentionally not supported. These names describe stable compact application outputs only.
DEFERRED_OUTPUT_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "create_scenario": {
        "direct_fields": ("scenario_handle",),
        "collections": {},
    },
    "run_scenario": {
        "direct_fields": ("scenario_handle", "solution_handle", "historical_solution_handle", "run_id"),
        "collections": {},
    },
    "run_optimization": {
        "direct_fields": ("solution_handle", "historical_solution_handle", "run_id"),
        "collections": {},
    },
    "select_problem": {
        "direct_fields": ("active_problem_id",),
        "collections": {},
    },
    "list_scenarios": {
        "direct_fields": (),
        "collections": {
            "scenarios": {
                "match_fields": ("handle", "name", "optimized", "solution_handle"),
                "extract_fields": ("handle", "name", "solution_handle"),
            }
        },
    },
    "list_runs": {
        "direct_fields": (),
        "collections": {
            "runs": {
                "match_fields": (
                    "run_id", "run_kind", "status", "scenario_handle",
                    "solution_handle", "current_alias", "error_code",
                ),
                "extract_fields": ("run_id", "scenario_handle", "solution_handle", "current_alias"),
            }
        },
    },
}




@dataclass(frozen=True)
class SessionEntityRecord:
    """Compact session-local entity provenance used only for cross-turn reference grounding."""

    reference_id: str
    entity_type: str
    handle: str
    display_name: str | None
    problem_id: str
    source_plan_id: str
    source_step_id: str
    focus_generation: int

    def __post_init__(self) -> None:
        if self.entity_type not in SESSION_ENTITY_TYPES:
            raise ValueError(f"Unsupported session entity type: {self.entity_type!r}")
        for label, value in (
            ("reference_id", self.reference_id),
            ("handle", self.handle),
            ("problem_id", self.problem_id),
            ("source_plan_id", self.source_plan_id),
            ("source_step_id", self.source_step_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Session entity {label} must be non-empty text.")
        if self.display_name is not None and (not isinstance(self.display_name, str) or not self.display_name.strip()):
            raise ValueError("Session entity display_name must be non-empty text or null.")
        if self.focus_generation < 1:
            raise ValueError("Session entity focus_generation must be positive.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "entity_type": self.entity_type,
            "handle": self.handle,
            "display_name": self.display_name,
            "problem_id": self.problem_id,
            "source_plan_id": self.source_plan_id,
            "source_step_id": self.source_step_id,
        }


class SessionEntityContext:
    """Session-local compact entity focus with deterministic references, never solution rows."""

    def __init__(self) -> None:
        self._next_sequence = 1
        self._generation = 0
        self._records: Dict[str, SessionEntityRecord] = {}
        self._focus: Dict[tuple[str, str], tuple[str, ...]] = {}
        self._pending: list[SessionEntityRecord] = []

    def begin_turn(self) -> int:
        self._generation += 1
        self._pending = []
        return self._generation

    @property
    def generation(self) -> int:
        return self._generation

    def record(
        self,
        *,
        entity_type: str,
        handle: str,
        display_name: str | None,
        problem_id: str,
        source_plan_id: str,
        source_step_id: str,
    ) -> SessionEntityRecord:
        if self._generation < 1:
            self.begin_turn()
        reference_id = f"session_entity_{self._next_sequence:06d}"
        self._next_sequence += 1
        record = SessionEntityRecord(
            reference_id=reference_id,
            entity_type=entity_type,
            handle=handle,
            display_name=display_name,
            problem_id=problem_id,
            source_plan_id=source_plan_id,
            source_step_id=source_step_id,
            focus_generation=self._generation,
        )
        self._records[reference_id] = record
        self._pending.append(record)
        return record

    def finalize_turn(self) -> None:
        grouped: Dict[tuple[str, str], list[SessionEntityRecord]] = {}
        for record in self._pending:
            grouped.setdefault((record.problem_id, record.entity_type), []).append(record)
        for key, records in grouped.items():
            # Deduplicate repeated touches of the same entity within one turn while retaining
            # the latest source metadata. Multiple distinct handles remain intentionally ambiguous.
            by_handle: Dict[str, SessionEntityRecord] = {}
            for record in records:
                by_handle[record.handle] = record
            self._focus[key] = tuple(item.reference_id for item in by_handle.values())
        self._pending = []

    def current(self, *, problem_id: str, entity_type: str | None = None) -> tuple[SessionEntityRecord, ...]:
        if entity_type is not None:
            refs = self._focus.get((problem_id, entity_type), ())
            return tuple(self._records[ref] for ref in refs)
        records: list[SessionEntityRecord] = []
        for (stored_problem, _stored_type), refs in self._focus.items():
            if stored_problem == problem_id:
                records.extend(self._records[ref] for ref in refs)
        return tuple(sorted(records, key=lambda item: item.reference_id))

    def resolve(
        self,
        reference_id: str,
        *,
        expected_entity_type: str,
        expected_problem_id: str,
    ) -> SessionEntityRecord:
        try:
            record = self._records[reference_id]
        except KeyError as exc:
            raise PlanValidationError(
                "unknown_session_entity",
                f"Session entity reference is not registered: {reference_id}",
            ) from exc
        if record.entity_type != expected_entity_type:
            raise PlanValidationError(
                "incompatible_session_entity_type",
                f"Session entity {reference_id} is {record.entity_type}, expected {expected_entity_type}.",
            )
        if record.problem_id != expected_problem_id:
            raise PlanValidationError(
                "incompatible_session_entity_problem",
                f"Session entity {reference_id} belongs to {record.problem_id}, expected {expected_problem_id}.",
            )
        current_refs = self._focus.get((record.problem_id, record.entity_type), ())
        if reference_id not in current_refs:
            raise PlanValidationError(
                "inactive_session_entity",
                f"Session entity {reference_id} is not an active recent {record.entity_type} reference.",
            )
        return record

    def planner_entities(self, *, problem_id: str) -> tuple[Dict[str, object], ...]:
        records = self.current(problem_id=problem_id)
        counts: Dict[str, int] = {}
        for record in records:
            counts[record.entity_type] = counts.get(record.entity_type, 0) + 1
        return tuple(
            {**record.to_dict(), "unambiguous_for_type": counts[record.entity_type] == 1}
            for record in records
        )

    def snapshot(self) -> Mapping[str, SessionEntityRecord]:
        return dict(self._records)


class PlanValidationError(ValueError):
    """Raised when a planner-produced plan violates deterministic protocol rules."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DeferredSelector:
    """Typed selector over one declared prior step's compact output.

    ``direct_field`` extracts one allowlisted scalar field. ``collection_match`` selects
    exactly one record by one allowlisted equality predicate and extracts one allowlisted
    scalar field. ``unique_collection_item`` extracts from a collection only when it contains
    exactly one record, which is useful for anaphoric session references without guessing.
    """

    selector_type: str
    extract_field: str
    collection: str | None = None
    match_field: str | None = None
    match_value: object | None = None

    def __post_init__(self) -> None:
        if self.selector_type not in DEFERRED_SELECTOR_TYPES:
            raise ValueError(f"Unknown deferred selector type: {self.selector_type!r}")
        if not isinstance(self.extract_field, str) or _SAFE_FIELD_RE.fullmatch(self.extract_field) is None:
            raise ValueError("Deferred selector extract_field must be a safe field name.")
        if self.selector_type == "direct_field":
            if self.collection is not None or self.match_field is not None or self.match_value is not None:
                raise ValueError("direct_field selectors cannot declare collection-match metadata.")
            return
        if not isinstance(self.collection, str) or _SAFE_FIELD_RE.fullmatch(self.collection) is None:
            raise ValueError("Collection selectors require a safe collection name.")
        if self.selector_type == "collection_match":
            if not isinstance(self.match_field, str) or _SAFE_FIELD_RE.fullmatch(self.match_field) is None:
                raise ValueError("collection_match selectors require a safe match_field.")
            if not isinstance(self.match_value, (str, int, float, bool)) and self.match_value is not None:
                raise ValueError("collection_match match_value must be a JSON scalar.")
        else:
            if self.match_field is not None or self.match_value is not None:
                raise ValueError("unique_collection_item selectors cannot declare a match predicate.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "selector_type": self.selector_type,
            "collection": self.collection,
            "match_field": self.match_field,
            "match_value": self.match_value,
            "extract_field": self.extract_field,
        }


@dataclass(frozen=True)
class ArgumentBinding:
    """Planner-declared provenance for one eventual tool argument path.

    ``literal`` preserves a user-stated scalar exactly. ``deferred`` allows the executor
    to use a scalar discovered only from one declared earlier step through a typed selector.
    Application defaults are intentionally not planner-authored bindings.
    """

    argument_path: str
    binding_type: str
    literal_value: object | None = None
    source_step_id: str | None = None
    source_tool_name: str | None = None
    selector: DeferredSelector | None = None
    session_entity_type: str | None = None
    session_entity_ref: str | None = None
    session_problem_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.argument_path, str) or _ARGUMENT_PATH_RE.fullmatch(self.argument_path) is None:
            raise ValueError(f"Invalid argument binding path: {self.argument_path!r}")
        if self.binding_type not in ARGUMENT_BINDING_TYPES:
            raise ValueError(f"Unknown argument binding type: {self.binding_type!r}")
        if self.binding_type == "literal":
            if any(value is not None for value in (
                self.source_step_id, self.source_tool_name, self.selector,
                self.session_entity_type, self.session_entity_ref, self.session_problem_id,
            )):
                raise ValueError("Literal bindings cannot declare deferred/session-source metadata.")
            if not isinstance(self.literal_value, (str, int, float, bool)) and self.literal_value is not None:
                raise ValueError("Literal binding values must be JSON scalar values.")
        elif self.binding_type == "deferred":
            if not isinstance(self.source_step_id, str) or not self.source_step_id.strip():
                raise ValueError("Deferred bindings require source_step_id.")
            if not isinstance(self.source_tool_name, str) or self.source_tool_name not in DEFERRED_OUTPUT_CONTRACTS:
                raise ValueError("Deferred bindings require an allowlisted source_tool_name with a compact-output contract.")
            if not isinstance(self.selector, DeferredSelector):
                raise ValueError("Deferred bindings require a typed selector.")
            if self.literal_value is not None:
                raise ValueError("Deferred bindings cannot carry a literal value.")
            if any(value is not None for value in (
                self.session_entity_type, self.session_entity_ref, self.session_problem_id,
            )):
                raise ValueError("Deferred bindings cannot declare session-source metadata.")
        else:
            if self.literal_value is not None or self.source_step_id is not None or self.source_tool_name is not None or self.selector is not None:
                raise ValueError("Session bindings cannot carry literal/deferred-source metadata.")
            if self.session_entity_type not in SESSION_ENTITY_TYPES:
                raise ValueError("Session bindings require a supported session_entity_type.")
            if not isinstance(self.session_entity_ref, str) or not self.session_entity_ref.strip():
                raise ValueError("Session bindings require session_entity_ref.")
            if not isinstance(self.session_problem_id, str) or not self.session_problem_id.strip():
                raise ValueError("Session bindings require session_problem_id.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "argument_path": self.argument_path,
            "binding_type": self.binding_type,
            "literal_value": self.literal_value,
            "source_step_id": self.source_step_id,
            "source_tool_name": self.source_tool_name,
            "selector": self.selector.to_dict() if self.selector is not None else None,
            "session_entity_type": self.session_entity_type,
            "session_entity_ref": self.session_entity_ref,
            "session_problem_id": self.session_problem_id,
        }


@dataclass
class PlanStep:
    """One ordered application action in an explicit execution plan."""

    step_id: str
    objective: str
    expected_application_tool: str | None
    target_problem_id: str | None = None
    argument_bindings: Tuple[ArgumentBinding, ...] = ()
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.status not in PLAN_STEP_STATUSES:
            raise ValueError(f"Unknown plan step status: {self.status}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "step_id": self.step_id,
            "objective": self.objective,
            "expected_application_tool": self.expected_application_tool,
            "target_problem_id": self.target_problem_id,
            "argument_bindings": [binding.to_dict() for binding in self.argument_bindings],
            "status": self.status,
        }


@dataclass
class ExecutionPlan:
    """Application-owned explicit plan with provenance constraints, not fabricated tool payloads."""

    plan_id: str
    user_goal: str
    initial_active_problem_id: str
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def planned_tool_trace(self) -> Tuple[str, ...]:
        return tuple(
            step.expected_application_tool
            for step in self.steps
            if step.expected_application_tool is not None
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "user_goal": self.user_goal,
            "initial_active_problem_id": self.initial_active_problem_id,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PlanStepExecutionSummary:
    """Compact metadata for one finished/skipped plan step; never solution rows."""

    step_id: str
    expected_application_tool: str | None
    status: str
    actual_tool: str | None = None
    error_code: str | None = None
    outcome_metadata: Mapping[str, object] | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "step_id": self.step_id,
            "expected_application_tool": self.expected_application_tool,
            "status": self.status,
            "actual_tool": self.actual_tool,
            "error_code": self.error_code,
            "outcome_metadata": dict(self.outcome_metadata or {}),
        }


@dataclass(frozen=True)
class PlanExecutionRecord:
    """Session-local plan adherence record containing metadata only."""

    plan_id: str
    initial_problem_id: str
    final_problem_id: str
    planned_tool_trace: Tuple[str, ...]
    actual_tool_trace: Tuple[str, ...]
    steps: Tuple[PlanStepExecutionSummary, ...]
    validation_error: str | None = None
    final_synthesis_error: str | None = None
    plan_status: str = "completed"

    @property
    def succeeded(self) -> bool:
        return (
            self.plan_status == "completed"
            and self.validation_error is None
            and all(step.status in {"completed", "skipped"} for step in self.steps)
            and not any(
                step.status in {"failed", "recoverable_failed", "rejected", "outcome_unknown", "pending_approval", "blocked_unsupported_requirement"}
                for step in self.steps
            )
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "initial_problem_id": self.initial_problem_id,
            "final_problem_id": self.final_problem_id,
            "planned_tool_trace": list(self.planned_tool_trace),
            "actual_tool_trace": list(self.actual_tool_trace),
            "steps": [step.to_dict() for step in self.steps],
            "validation_error": self.validation_error,
            "final_synthesis_error": self.final_synthesis_error,
            "plan_status": self.plan_status,
            "succeeded": self.succeeded,
        }


def _selector_contract_error(source_tool: str | None, selector: DeferredSelector) -> str | None:
    contract = DEFERRED_OUTPUT_CONTRACTS.get(str(source_tool))
    if contract is None:
        return f"Tool {source_tool!r} does not expose deferred-reference output selectors."
    if selector.selector_type == "direct_field":
        allowed = tuple(contract.get("direct_fields", ()))
        if selector.extract_field not in allowed:
            return (
                f"Tool {source_tool!r} does not expose direct deferred field "
                f"{selector.extract_field!r}."
            )
        return None

    collections = contract.get("collections")
    if not isinstance(collections, Mapping):
        return f"Tool {source_tool!r} does not expose collection selectors."
    collection_contract = collections.get(selector.collection)
    if not isinstance(collection_contract, Mapping):
        return (
            f"Tool {source_tool!r} does not expose deferred collection "
            f"{selector.collection!r}."
        )
    extract_fields = tuple(collection_contract.get("extract_fields", ()))
    if selector.extract_field not in extract_fields:
        return (
            f"Collection {selector.collection!r} does not expose extract field "
            f"{selector.extract_field!r}."
        )
    if selector.selector_type == "collection_match":
        match_fields = tuple(collection_contract.get("match_fields", ()))
        if selector.match_field not in match_fields:
            return (
                f"Collection {selector.collection!r} does not expose match field "
                f"{selector.match_field!r}."
            )
    return None


class PlanValidator:
    """Deterministic allowlist validator for planner-produced execution plans."""

    def __init__(
        self,
        *,
        application_tool_names: Iterable[str],
        registered_problem_ids: Iterable[str],
        session_entities: Mapping[str, SessionEntityRecord] | None = None,
        historical_reference_anchor: str | None = None,
        max_steps: int = MAX_PLAN_STEPS,
    ) -> None:
        self._tool_names = frozenset(str(name) for name in application_tool_names)
        self._problem_ids = frozenset(str(problem_id) for problem_id in registered_problem_ids)
        self._session_entities = dict(session_entities or {})
        self._historical_reference_anchor = historical_reference_anchor
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self._historical_reference_anchor is not None:
            if (
                not isinstance(self._historical_reference_anchor, str)
                or not self._historical_reference_anchor.strip()
            ):
                raise ValueError("historical_reference_anchor must be non-empty text or None")

    @property
    def application_tool_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tool_names))

    @property
    def registered_problem_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._problem_ids))

    def validate(self, plan: ExecutionPlan) -> None:
        if not isinstance(plan.user_goal, str) or not plan.user_goal.strip():
            raise PlanValidationError("empty_goal", "Plan user goal must be non-empty text.")
        if not plan.steps:
            raise PlanValidationError("empty_plan", "Execution plan must contain at least one step.")
        if len(plan.steps) > self.max_steps:
            raise PlanValidationError(
                "plan_too_long",
                f"Execution plan exceeds the maximum of {self.max_steps} steps.",
            )
        if plan.initial_active_problem_id not in self._problem_ids:
            raise PlanValidationError(
                "unknown_initial_problem",
                f"Initial problem is not registered: {plan.initial_active_problem_id}",
            )

        seen: set[str] = set()
        prior_steps: dict[str, PlanStep] = {}
        simulated_active = plan.initial_active_problem_id
        historical_compare_seen = False
        for step in plan.steps:
            if not isinstance(step.step_id, str) or not step.step_id.strip():
                raise PlanValidationError("invalid_step_id", "Every plan step requires a non-empty step_id.")
            if step.step_id in seen:
                raise PlanValidationError("duplicate_step_id", f"Duplicate plan step_id: {step.step_id}")
            if not isinstance(step.objective, str) or not step.objective.strip():
                raise PlanValidationError(
                    "empty_step_objective",
                    f"Plan step {step.step_id} requires a non-empty objective.",
                )
            if step.status != "pending":
                raise PlanValidationError(
                    "invalid_initial_status",
                    f"Plan step {step.step_id} must begin with status=pending.",
                )

            tool = step.expected_application_tool
            if tool is not None and tool not in self._tool_names:
                raise PlanValidationError(
                    "unknown_application_tool",
                    f"Plan step {step.step_id} references unknown application tool: {tool}",
                )

            target = step.target_problem_id
            if target is not None and target not in self._problem_ids:
                raise PlanValidationError(
                    "unknown_target_problem",
                    f"Plan step {step.step_id} targets unregistered problem: {target}",
                )

            if tool == "select_problem":
                if target is None:
                    raise PlanValidationError(
                        "missing_selection_target",
                        f"select_problem step {step.step_id} requires target_problem_id.",
                    )
                simulated_active = target
            elif target is not None and target != simulated_active:
                raise PlanValidationError(
                    "invalid_domain_transition",
                    f"Plan step {step.step_id} targets {target} while active problem would be {simulated_active}; "
                    "an explicit select_problem step is required first.",
                )

            binding_paths: set[str] = set()
            for binding in step.argument_bindings:
                if binding.argument_path in binding_paths:
                    raise PlanValidationError(
                        "duplicate_argument_binding",
                        f"Plan step {step.step_id} binds {binding.argument_path!r} more than once.",
                    )
                binding_paths.add(binding.argument_path)
                if binding.binding_type == "deferred":
                    if binding.source_step_id not in prior_steps:
                        raise PlanValidationError(
                            "invalid_deferred_dependency",
                            f"Plan step {step.step_id} defers {binding.argument_path!r} to a non-prior step.",
                        )
                    source_tool = prior_steps[str(binding.source_step_id)].expected_application_tool
                    if source_tool != binding.source_tool_name:
                        raise PlanValidationError(
                            "invalid_deferred_selector",
                            f"Plan step {step.step_id} declares deferred source tool {binding.source_tool_name!r} "
                            f"but source step {binding.source_step_id!r} uses {source_tool!r}.",
                        )
                    selector_error = _selector_contract_error(source_tool, binding.selector)  # type: ignore[arg-type]
                    if selector_error is not None:
                        raise PlanValidationError("invalid_deferred_selector", selector_error)
                elif binding.binding_type == "session":
                    record = self._session_entities.get(str(binding.session_entity_ref))
                    if record is None:
                        raise PlanValidationError(
                            "unknown_session_entity",
                            f"Plan step {step.step_id} references unknown session entity {binding.session_entity_ref!r}.",
                        )
                    if record.entity_type != binding.session_entity_type:
                        raise PlanValidationError(
                            "incompatible_session_entity_type",
                            f"Plan step {step.step_id} declares {binding.session_entity_type!r} for a {record.entity_type!r} entity.",
                        )
                    if record.problem_id != binding.session_problem_id or record.problem_id != simulated_active:
                        raise PlanValidationError(
                            "incompatible_session_entity_problem",
                            f"Plan step {step.step_id} session entity belongs to {record.problem_id!r}, active problem is {simulated_active!r}.",
                        )
                    compatible_recent = [
                        candidate for candidate in self._session_entities.values()
                        if candidate.problem_id == simulated_active
                        and candidate.entity_type == record.entity_type
                    ]
                    if len(compatible_recent) != 1:
                        raise PlanValidationError(
                            "ambiguous_session_entity",
                            f"Plan step {step.step_id} cannot bind an ambiguous recent {record.entity_type} reference.",
                        )

            required_binding_paths: tuple[str, ...] = ()
            if tool == "select_problem":
                required_binding_paths = ("problem_id",)
            elif tool == "create_scenario":
                required_binding_paths = ("name",)
            elif tool == "run_scenario":
                required_binding_paths = ("scenario",)
            elif tool == "validate_solution":
                required_binding_paths = ("handle",)
            elif tool == "compare_solutions":
                required_binding_paths = ("solution_a", "solution_b")
            missing = [path for path in required_binding_paths if path not in binding_paths]
            if missing:
                raise PlanValidationError(
                    "missing_argument_binding",
                    f"Plan step {step.step_id} is missing argument-intent binding(s): {missing}.",
                )

            by_path = {binding.argument_path: binding for binding in step.argument_bindings}
            if tool == "select_problem":
                problem_binding = by_path["problem_id"]
                if problem_binding.binding_type != "literal" or problem_binding.literal_value != target:
                    raise PlanValidationError(
                        "selection_binding_mismatch",
                        "select_problem.problem_id must be a literal equal to target_problem_id.",
                    )
            if tool == "create_scenario" and by_path["name"].binding_type != "literal":
                raise PlanValidationError(
                    "invalid_name_binding",
                    "create_scenario.name must be a literal user-requested value.",
                )
            if tool == "compare_solutions" and self._historical_reference_anchor is not None:
                anchor = self._historical_reference_anchor
                solution_a = by_path["solution_a"]
                solution_b = by_path["solution_b"]
                selector = solution_b.selector
                historical_shape_valid = (
                    solution_a.binding_type == "literal"
                    and solution_a.literal_value == anchor
                    and solution_b.binding_type == "deferred"
                    and solution_b.source_tool_name == "list_runs"
                    and isinstance(selector, DeferredSelector)
                    and selector.selector_type == "collection_match"
                    and selector.collection == "runs"
                    and selector.match_field == "current_alias"
                    and selector.match_value == anchor
                    and selector.extract_field == "solution_handle"
                )
                if not historical_shape_valid:
                    raise PlanValidationError(
                        "invalid_historical_reference_binding",
                        "Historical snapshot comparison must preserve the explicit anchor as solution_a literal "
                        "and resolve only solution_b from list_runs by current_alias == anchor.",
                    )
                historical_compare_seen = True
            expected_session_types = {
                ("run_scenario", "scenario"): "scenario",
                ("validate_solution", "handle"): "solution",
                ("compare_solutions", "solution_a"): "solution",
                ("compare_solutions", "solution_b"): "solution",
            }
            for required_path in required_binding_paths:
                binding = by_path[required_path]
                if binding.binding_type == "session":
                    expected_entity_type = expected_session_types.get((tool, required_path))
                    if expected_entity_type is None or binding.session_entity_type != expected_entity_type:
                        raise PlanValidationError(
                            "incompatible_session_entity_type",
                            f"{tool}.{required_path} cannot consume session entity type {binding.session_entity_type!r}.",
                        )
                binding = by_path[required_path]
                if binding.binding_type == "literal" and (
                    not isinstance(binding.literal_value, str) or not binding.literal_value.strip()
                ):
                    raise PlanValidationError(
                        "invalid_required_literal",
                        f"{tool}.{required_path} literal binding must be non-empty text.",
                    )
            for binding in step.argument_bindings:
                if binding.argument_path == "fast":
                    if binding.binding_type != "literal" or type(binding.literal_value) is not bool:
                        raise PlanValidationError(
                            "invalid_fast_binding",
                            "fast may be bound only as an explicit boolean literal.",
                        )
                if binding.argument_path.startswith("parameter_overrides.") and binding.binding_type != "literal":
                    raise PlanValidationError(
                        "invalid_parameter_binding",
                        "Optimization parameter overrides must be explicit literal user values.",
                    )

            seen.add(step.step_id)
            prior_steps[step.step_id] = step

        if self._historical_reference_anchor is not None and not historical_compare_seen:
            raise PlanValidationError(
                "missing_historical_compare_step",
                "The anchored historical-reference request requires a compare_solutions step.",
            )


__all__ = [
    "ARGUMENT_BINDING_TYPES",
    "DEFERRED_OUTPUT_CONTRACTS",
    "DEFERRED_SELECTOR_TYPES",
    "ArgumentBinding",
    "DeferredSelector",
    "ExecutionPlan",
    "MAX_PLAN_STEPS",
    "PLAN_STEP_STATUSES",
    "SESSION_ENTITY_TYPES",
    "SessionEntityContext",
    "SessionEntityRecord",
    "PlanExecutionRecord",
    "PlanStep",
    "PlanStepExecutionSummary",
    "PlanValidationError",
    "PlanValidator",
]
