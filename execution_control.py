"""Session-local approval and execution-control metadata for Step 18.

This module contains no optimizer/domain logic and no provider-specific behavior. It stores
only compact control metadata, fingerprints, and deterministic policy decisions. Solution rows
and source datasets remain owned by the existing application stores.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Sequence

from optimization_problem import OptimizationProblem


APPROVAL_STATUSES = ("pending", "approved", "rejected", "expired")
EXECUTION_LEDGER_STATES = (
    "not_started",
    "approval_pending",
    "dispatch_started",
    "dispatch_completed",
    "failed_before_dispatch",
    "outcome_unknown",
    "rejected",
    "blocked_unsupported_requirement",
)


class ExecutionControlError(RuntimeError):
    """Base class for deterministic Step 18 control failures."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ApprovalNotFoundError(ExecutionControlError):
    def __init__(self, approval_id: str) -> None:
        super().__init__("approval_not_found", f"Approval is not registered: {approval_id}")


class PlanControlNotFoundError(ExecutionControlError):
    def __init__(self, plan_id: str) -> None:
        super().__init__("plan_not_found", f"Plan is not registered for controlled execution: {plan_id}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze_json(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _thaw_json(value: object) -> object:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {item[0]: _thaw_json(item[1]) for item in value}
        return [_thaw_json(item) for item in value]
    return value


def stable_fingerprint(value: object) -> str:
    """Return a deterministic SHA-256 over JSON-compatible control metadata."""
    frozen = _freeze_json(value)
    canonical = json.dumps(_thaw_json(frozen), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    reason_codes: tuple[str, ...]
    safe_action_summary: object

    def summary_dict(self) -> Dict[str, object]:
        thawed = _thaw_json(self.safe_action_summary)
        return dict(thawed) if isinstance(thawed, Mapping) else {"summary": thawed}


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    plan_id: str
    step_id: str
    tool_name: str
    problem_id: str
    reason_codes: tuple[str, ...]
    safe_action_summary: object
    plan_fingerprint: str
    step_fingerprint: str
    state_fingerprint: str
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.status not in APPROVAL_STATUSES:
            raise ValueError(f"Unsupported approval status: {self.status}")

    def summary_dict(self) -> Dict[str, object]:
        thawed = _thaw_json(self.safe_action_summary)
        return dict(thawed) if isinstance(thawed, Mapping) else {"summary": thawed}

    def to_dict(self) -> Dict[str, object]:
        return {
            "approval_id": self.approval_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "problem_id": self.problem_id,
            "reason_codes": list(self.reason_codes),
            "safe_action_summary": self.summary_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "step_fingerprint": self.step_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "status": self.status,
        }


class ApprovalStore:
    """Session-local immutable approval records with deterministic sequential IDs."""

    def __init__(self) -> None:
        self._next_sequence = 1
        self._records: Dict[str, ApprovalRecord] = {}

    def create(
        self,
        *,
        plan_id: str,
        step_id: str,
        tool_name: str,
        problem_id: str,
        reason_codes: Sequence[str],
        safe_action_summary: Mapping[str, object],
        plan_fingerprint: str,
        step_fingerprint: str,
        state_fingerprint: str,
    ) -> ApprovalRecord:
        approval_id = f"approval_{self._next_sequence:06d}"
        self._next_sequence += 1
        record = ApprovalRecord(
            approval_id=approval_id,
            plan_id=plan_id,
            step_id=step_id,
            tool_name=tool_name,
            problem_id=problem_id,
            reason_codes=tuple(str(code) for code in reason_codes),
            safe_action_summary=_freeze_json(dict(safe_action_summary)),
            plan_fingerprint=plan_fingerprint,
            step_fingerprint=step_fingerprint,
            state_fingerprint=state_fingerprint,
            status="pending",
        )
        self._records[approval_id] = record
        return record

    def get(self, approval_id: str) -> ApprovalRecord:
        try:
            return self._records[approval_id]
        except KeyError as exc:
            raise ApprovalNotFoundError(approval_id) from exc

    def pending(self) -> tuple[ApprovalRecord, ...]:
        return tuple(record for record in self._records.values() if record.status == "pending")

    def _set_status(self, approval_id: str, status: str) -> ApprovalRecord:
        record = self.get(approval_id)
        if record.status != "pending":
            raise ExecutionControlError(
                "approval_not_pending",
                f"Approval {approval_id} is {record.status}, not pending.",
            )
        updated = replace(record, status=status)
        self._records[approval_id] = updated
        return updated

    def approve(self, approval_id: str) -> ApprovalRecord:
        return self._set_status(approval_id, "approved")

    def reject(self, approval_id: str) -> ApprovalRecord:
        return self._set_status(approval_id, "rejected")

    def expire(self, approval_id: str) -> ApprovalRecord:
        return self._set_status(approval_id, "expired")

    def list(self) -> tuple[ApprovalRecord, ...]:
        return tuple(self._records.values())


@dataclass(frozen=True)
class ExecutionLedgerEntry:
    plan_id: str
    step_id: str
    state: str = "not_started"
    approval_id: str | None = None
    state_fingerprint: str | None = None
    actual_tool: str | None = None
    error_code: str | None = None
    receipt_metadata: object = ()

    def __post_init__(self) -> None:
        if self.state not in EXECUTION_LEDGER_STATES:
            raise ValueError(f"Unsupported execution-ledger state: {self.state}")

    def receipt_dict(self) -> Dict[str, object]:
        thawed = _thaw_json(self.receipt_metadata)
        return dict(thawed) if isinstance(thawed, Mapping) else {}

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "state": self.state,
            "approval_id": self.approval_id,
            "state_fingerprint": self.state_fingerprint,
            "actual_tool": self.actual_tool,
            "error_code": self.error_code,
            "receipt_metadata": self.receipt_dict(),
        }


class ExecutionControlLedger:
    """Independent session-local duplicate-dispatch guard for planned application steps."""

    def __init__(self) -> None:
        self._entries: Dict[tuple[str, str], ExecutionLedgerEntry] = {}

    def ensure(self, plan_id: str, step_id: str) -> ExecutionLedgerEntry:
        key = (plan_id, step_id)
        if key not in self._entries:
            self._entries[key] = ExecutionLedgerEntry(plan_id=plan_id, step_id=step_id)
        return self._entries[key]

    def get(self, plan_id: str, step_id: str) -> ExecutionLedgerEntry:
        return self.ensure(plan_id, step_id)

    def _replace(self, entry: ExecutionLedgerEntry, **changes: object) -> ExecutionLedgerEntry:
        updated = replace(entry, **changes)
        self._entries[(entry.plan_id, entry.step_id)] = updated
        return updated

    def mark_approval_pending(
        self, plan_id: str, step_id: str, *, approval_id: str, state_fingerprint: str
    ) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state not in {"not_started", "approval_pending"}:
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Cannot request approval from ledger state {entry.state}.",
            )
        return self._replace(
            entry,
            state="approval_pending",
            approval_id=approval_id,
            state_fingerprint=state_fingerprint,
            error_code=None,
        )

    def mark_dispatch_started(
        self, plan_id: str, step_id: str, *, state_fingerprint: str | None = None
    ) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state == "dispatch_completed":
            raise ExecutionControlError(
                "duplicate_dispatch_blocked",
                f"Step {plan_id}/{step_id} already completed and cannot dispatch again.",
            )
        if entry.state not in {"not_started", "approval_pending", "failed_before_dispatch"}:
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Cannot start dispatch from ledger state {entry.state}.",
            )
        return self._replace(
            entry,
            state="dispatch_started",
            state_fingerprint=state_fingerprint or entry.state_fingerprint,
            error_code=None,
        )

    def mark_dispatch_completed(
        self,
        plan_id: str,
        step_id: str,
        *,
        actual_tool: str,
        receipt_metadata: Mapping[str, object],
    ) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state != "dispatch_started":
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Cannot complete dispatch from ledger state {entry.state}.",
            )
        return self._replace(
            entry,
            state="dispatch_completed",
            actual_tool=actual_tool,
            error_code=None,
            receipt_metadata=_freeze_json(dict(receipt_metadata)),
        )

    def mark_failed_before_dispatch(
        self,
        plan_id: str,
        step_id: str,
        *,
        error_code: str,
        state_fingerprint: str | None = None,
    ) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state == "dispatch_completed":
            raise ExecutionControlError(
                "duplicate_dispatch_blocked",
                f"Completed step {plan_id}/{step_id} cannot become recoverable.",
            )
        return self._replace(
            entry,
            state="failed_before_dispatch",
            state_fingerprint=state_fingerprint or entry.state_fingerprint,
            error_code=error_code,
        )

    def prepare_resume(self, plan_id: str, step_id: str) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state != "failed_before_dispatch":
            raise ExecutionControlError(
                "resume_not_allowed",
                f"Step {plan_id}/{step_id} is not recoverable from ledger state {entry.state}.",
            )
        return self._replace(entry, state="not_started", error_code=None)

    def mark_outcome_unknown(
        self, plan_id: str, step_id: str, *, error_code: str
    ) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state != "dispatch_started":
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Unknown outcome requires dispatch_started, got {entry.state}.",
            )
        return self._replace(entry, state="outcome_unknown", error_code=error_code)

    def mark_rejected(self, plan_id: str, step_id: str, *, approval_id: str) -> ExecutionLedgerEntry:
        entry = self.ensure(plan_id, step_id)
        if entry.state != "approval_pending":
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Approval rejection requires approval_pending, got {entry.state}.",
            )
        return self._replace(entry, state="rejected", approval_id=approval_id, error_code="approval_rejected")

    def mark_blocked_unsupported_requirement(
        self,
        plan_id: str,
        step_id: str,
        *,
        error_code: str = "unsupported_required_capability",
    ) -> ExecutionLedgerEntry:
        """Record the known fail-closed transition before approval or dispatch."""
        entry = self.ensure(plan_id, step_id)
        if entry.state != "not_started":
            raise ExecutionControlError(
                "invalid_ledger_transition",
                f"Unsupported-requirement block requires not_started, got {entry.state}.",
            )
        return self._replace(
            entry,
            state="blocked_unsupported_requirement",
            approval_id=None,
            actual_tool=None,
            error_code=error_code,
        )

    def entries_for_plan(self, plan_id: str) -> tuple[ExecutionLedgerEntry, ...]:
        return tuple(
            entry for (stored_plan_id, _), entry in self._entries.items() if stored_plan_id == plan_id
        )


class ApprovalPolicy:
    """Deterministic provider-independent Step 18 approval policy."""

    READ_ONLY_TOOLS = frozenset(
        {
            "audit_data",
            "get_baseline",
            "validate_solution",
            "compare_solutions",
            "list_runs",
            "list_scenarios",
            "list_problems",
        }
    )
    NO_APPROVAL_SESSION_TOOLS = frozenset({"select_problem", "create_scenario"})

    @staticmethod
    def _non_default_overrides(
        problem: OptimizationProblem, overrides: Mapping[str, object]
    ) -> tuple[Dict[str, object], tuple[str, ...]]:
        non_default: Dict[str, object] = {}
        reasons: list[str] = []
        business = False
        solver = False
        for name, value in dict(overrides).items():
            definition = problem.parameters.get(name)
            if definition is None:
                # The normal argument/problem validation layer will reject this. Approval policy
                # does not duplicate parameter validity rules.
                continue
            if value == definition.default:
                continue
            non_default[name] = value
            if definition.category == "business":
                business = True
            elif definition.category == "solver":
                solver = True
        if business:
            reasons.append("non_default_business_context")
        if solver:
            reasons.append("non_default_solver_configuration")
        return non_default, tuple(reasons)

    def assess(
        self,
        *,
        tool_name: str,
        problem: OptimizationProblem,
        problem_id: str,
        fast: bool,
        parameter_overrides: Mapping[str, object],
        scenario_handle: str | None = None,
        scenario_name: str | None = None,
    ) -> ApprovalDecision:
        if tool_name in self.READ_ONLY_TOOLS or tool_name in self.NO_APPROVAL_SESSION_TOOLS:
            return ApprovalDecision(False, (), _freeze_json({"tool": tool_name}))
        if tool_name not in {"run_optimization", "run_scenario"}:
            return ApprovalDecision(False, (), _freeze_json({"tool": tool_name}))

        non_default, reasons = self._non_default_overrides(problem, parameter_overrides)
        reason_codes = list(reasons)
        if fast:
            reason_codes.append("fast_mode_requested")
        summary: Dict[str, object] = {
            "tool": tool_name,
            "problem_id": problem_id,
            "fast": bool(fast),
            "parameter_overrides": non_default,
        }
        if scenario_handle is not None:
            summary["scenario_handle"] = scenario_handle
        if scenario_name is not None:
            summary["scenario_name"] = scenario_name
        return ApprovalDecision(bool(reason_codes), tuple(reason_codes), _freeze_json(summary))


def control_debug_enabled() -> bool:
    value = (os.getenv("BONSAI_AGENT_SHOW_CONTROL") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def emit_control_debug(payload: Mapping[str, object]) -> None:
    if not control_debug_enabled():
        return
    # Payloads are built from allowlisted control metadata by the application. Never pass tool
    # arguments, solution rows, datasets, credentials, or model reasoning here.
    print(
        "CONTROL_DEBUG " + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )


__all__ = [
    "APPROVAL_STATUSES",
    "EXECUTION_LEDGER_STATES",
    "ApprovalDecision",
    "ApprovalNotFoundError",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ApprovalStore",
    "ExecutionControlError",
    "ExecutionControlLedger",
    "ExecutionLedgerEntry",
    "PlanControlNotFoundError",
    "control_debug_enabled",
    "emit_control_debug",
    "stable_fingerprint",
]
