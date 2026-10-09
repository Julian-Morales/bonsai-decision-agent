"""Pure projection of frozen core state into UX-1 business-facing components."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping


def _record_dict(record: object | None) -> Dict[str, object] | None:
    if record is None:
        return None
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else None
    return None


def _last_step(record: Mapping[str, object] | None, tool_names: Iterable[str]) -> Mapping[str, object] | None:
    if not record:
        return None
    targets = set(tool_names)
    steps = record.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if isinstance(step, Mapping) and step.get("expected_application_tool") in targets:
            return step
    return None


def _completed_step(record: Mapping[str, object] | None, tool_names: Iterable[str]) -> Mapping[str, object] | None:
    step = _last_step(record, tool_names)
    return step if step and step.get("status") == "completed" else None


# Exact PlanValidator diagnostic, not free-form assistant interpretation. The
# rejected record must independently identify invalid_domain_transition and
# zero dispatch before a switch action may be offered.
_DOMAIN_DIAGNOSTIC = re.compile(
    r"Execution plan rejected: invalid_domain_transition\. "
    r"Plan step [A-Za-z0-9_-]+ targets ([A-Za-z0-9_]+) "
    r"while active problem would be ([A-Za-z0-9_]+); "
    r"an explicit select_problem step is required first\."
)


def rejected_domain_transition_card(
    *, record: object | None, result_text: str,
    active_problem_id: str, registered_problem_ids: Iterable[str],
) -> Dict[str, object] | None:
    """Only a canonical, validated fail-closed diagnostic can enable switching."""
    data = _record_dict(record)
    if not data or data.get("validation_error") != "invalid_domain_transition":
        return None
    if data.get("plan_status") != "failed" or data.get("actual_tool_trace") != []:
        return None
    if data.get("initial_problem_id") != active_problem_id:
        return None
    match = _DOMAIN_DIAGNOSTIC.fullmatch(result_text)
    if match is None:
        return None
    target_id, diagnosed_active = match.groups()
    valid_ids = set(registered_problem_ids)
    if target_id not in valid_ids or diagnosed_active != active_problem_id or target_id == active_problem_id:
        return None
    return {
        "type": "invalid_domain_transition",
        "title": "Different problem selected",
        "requested_problem_id": target_id,
        "active_problem_id": active_problem_id,
        "why_not_executed": "The plan targeted another problem without an explicit select_problem step. PlanValidator rejected it before execution.",
        "no_optimization_executed": True,
    }


def uncertain_domain_transition_card(
    *, record: object | None, active_problem_id: str,
) -> Dict[str, object] | None:
    """Fail-closed conversation card when CORE rejected a switch without a safe target."""
    data = _record_dict(record)
    if (not data or data.get("validation_error") != "invalid_domain_transition"
            or data.get("plan_status") != "failed"
            or data.get("actual_tool_trace") != []
            or data.get("initial_problem_id") != active_problem_id):
        return None
    return {
        "type": "invalid_domain_transition", "title": "Different problem selected",
        "requested_problem_id": None, "active_problem_id": active_problem_id,
        "why_not_executed": "PlanValidator rejected a cross-problem operation without a valid explicit select_problem step.",
        "no_optimization_executed": True,
    }


def workflow_cards(
    *,
    record: object | None,
    approval: object | None,
    capability_decision: object | None,
) -> list[Dict[str, object]]:
    record_data = _record_dict(record)
    cards: list[Dict[str, object]] = []
    plan_id = record_data.get("plan_id") if record_data else None

    approval_data = _record_dict(approval)
    if approval_data and plan_id and approval_data.get("plan_id") == plan_id:
        status = approval_data.get("status")
        if status == "pending":
            cards.append(
                {
                    "type": "approval_pending",
                    "title": "Approval required",
                    "approval_id": approval_data.get("approval_id"),
                    "tool": approval_data.get("tool_name"),
                    "problem_id": approval_data.get("problem_id"),
                    "reason_codes": approval_data.get("reason_codes", []),
                    "summary": approval_data.get("safe_action_summary", {}),
                }
            )
        elif status in {"approved", "rejected"}:
            cards.append(
                {
                    "type": "approval_approved" if status == "approved" else "approval_rejected",
                    "title": "Approved" if status == "approved" else "Rejected",
                    "approval_id": approval_data.get("approval_id"),
                    "tool": approval_data.get("tool_name"),
                    "problem_id": approval_data.get("problem_id"),
                    "status": status,
                }
            )

    capability_data = _record_dict(capability_decision)
    blocked_steps = []
    if record_data and isinstance(record_data.get("steps"), list):
        blocked_steps = [
            step for step in record_data["steps"]
            if isinstance(step, Mapping) and step.get("status") == "blocked_unsupported_requirement"
        ]
    if capability_data and blocked_steps and capability_data.get("decision") == "blocked":
        extracted = capability_data.get("extracted_requirement")
        cards.append(
            {
                "type": "blocked_unsupported_requirement",
                "title": "Unsupported requirement",
                "requested_capability": capability_data.get("capability_id"),
                "requested_requirement": extracted if isinstance(extracted, Mapping) else {},
                "why_unsupported": capability_data.get("safe_explanation"),
                "problem_id": capability_data.get("problem_id"),
                "reason_code": capability_data.get("reason_code"),
                "no_optimization_executed": True,
            }
        )

    optimization = _completed_step(record_data, {"run_optimization", "run_scenario"})
    if optimization:
        outcome = optimization.get("outcome_metadata")
        if isinstance(outcome, Mapping) and outcome:
            validated = outcome.get("validated_final")
            cards.append(
                {
                    "type": "optimization_success",
                    "title": "Optimization completed",
                    "run_id": outcome.get("run_id"),
                    "solution_handle": outcome.get("solution_handle"),
                    "historical_solution_handle": outcome.get("historical_solution_handle"),
                    "validated": bool(validated.get("valid")) if isinstance(validated, Mapping) else None,
                }
            )

    validation = _completed_step(record_data, {"validate_solution"})
    if validation and isinstance(validation.get("outcome_metadata"), Mapping):
        outcome = validation["outcome_metadata"]
        cards.append(
            {
                "type": "validation_result",
                "title": "Validation result",
                "handle": outcome.get("handle"),
                "valid": outcome.get("valid"),
                "error_count": outcome.get("error_count"),
            }
        )

    if record_data and isinstance(record_data.get("steps"), list):
        errors = [
            step for step in record_data["steps"]
            if isinstance(step, Mapping)
            and step.get("status") in {"failed", "recoverable_failed", "outcome_unknown"}
        ]
        if errors:
            terminal = errors[-1]
            cards.append(
                {
                    "type": "outcome_unknown" if terminal.get("status") == "outcome_unknown" else "error",
                    "title": "Manual intervention required" if terminal.get("status") == "outcome_unknown" else "Execution issue",
                    "step_id": terminal.get("step_id"),
                    "status": terminal.get("status"),
                    "error_code": terminal.get("error_code"),
                    "problem_id": record_data.get("final_problem_id"),
                    "dispatch_state": (terminal.get("outcome_metadata") or {}).get("execution_ledger_state") if isinstance(terminal.get("outcome_metadata"), Mapping) else None,
                    "safe_message": (terminal.get("outcome_metadata") or {}).get("message") if isinstance(terminal.get("outcome_metadata"), Mapping) else None,
                }
            )
    return cards


def decision_view(*, record: object | None, capability_decision: object | None, approval: object | None = None, transition: Mapping[str, object] | None = None) -> Dict[str, object]:
    record_data = _record_dict(record)
    capability_data = _record_dict(capability_decision)
    approval_data = _record_dict(approval)
    if record_data and approval_data and approval_data.get("plan_id") == record_data.get("plan_id") and approval_data.get("status") == "pending":
        return {"type": "approval_review", "title": "Approval Review", "data": {
            "problem_id": approval_data.get("problem_id"),
            "action": approval_data.get("tool_name"),
            "requested_changes": (approval_data.get("safe_action_summary") or {}).get("parameter_overrides", {}),
            "action_summary": approval_data.get("safe_action_summary"),
            "reason_codes": approval_data.get("reason_codes", []),
            "status": "pending", "approval_id": approval_data.get("approval_id"),
        }}
    if transition:
        return {"type": "domain_transition", "title": "Different problem selected", "data": dict(transition)}
    if record_data and record_data.get("validation_error") == "invalid_domain_transition" and record_data.get("actual_tool_trace") == []:
        # The diagnostic may not safely identify a target; do not infer one.
        return {"type": "domain_transition", "title": "Different problem selected", "data": {
            "active_problem_id": record_data.get("initial_problem_id"),
            "requested_problem_id": None,
            "why_not_executed": "The plan failed the domain-transition safety check. No execution occurred.",
        }}

    if record_data and isinstance(record_data.get("steps"), list):
        if any(
            isinstance(step, Mapping) and step.get("status") == "blocked_unsupported_requirement"
            for step in record_data["steps"]
        ) and capability_data:
            return {
                "type": "unsupported_requirement",
                "title": "Unsupported requirement",
                "data": capability_data,
                "no_execution": True,
            }

    if record_data and isinstance(record_data.get("steps"), list):
        errors = [step for step in record_data["steps"] if isinstance(step, Mapping)
                  and step.get("status") in {"failed", "recoverable_failed", "outcome_unknown"}]
        if errors:
            step = errors[-1]
            outcome = step.get("outcome_metadata") if isinstance(step.get("outcome_metadata"), Mapping) else {}
            dispatched_tools = record_data.get("actual_tool_trace") or []
            previously_dispatched_optimization = any(tool in {"run_optimization", "run_scenario"}
                for tool in dispatched_tools)
            uncertain_optimization = step.get("status") == "outcome_unknown" and step.get("expected_application_tool") in {"run_optimization", "run_scenario"}
            return {"type": "execution_failure", "title": "Execution issue", "data": {
                "problem_id": record_data.get("final_problem_id"), "plan_status": record_data.get("plan_status"),
                "optimization_started": True if previously_dispatched_optimization else None if uncertain_optimization else False,
                "step_id": step.get("step_id"), "action": step.get("expected_application_tool"),
                "status": step.get("status"), "error_code": step.get("error_code"),
                "dispatch_state": outcome.get("execution_ledger_state") or
                    ("outcome_unknown" if step.get("status") == "outcome_unknown" else None),
                "dispatch_attempted": outcome.get("dispatch_attempted"),
                "safe_message": outcome.get("message"),
            }}

    if record_data and record_data.get("validation_error"):
        return {"type": "execution_failure", "title": "Execution issue", "data": {
            "problem_id": record_data.get("initial_problem_id"),
            "plan_status": record_data.get("plan_status"),
            "error_code": record_data.get("validation_error"),
            "dispatch_state": "failed_before_dispatch" if not record_data.get("actual_tool_trace") else None,
            "dispatch_attempted": bool(record_data.get("actual_tool_trace")),
            "optimization_started": any(t in {"run_optimization", "run_scenario"} for t in record_data.get("actual_tool_trace", [])),
        }}

    # A completed tool-free plan can lose only its final LLM narration. This is
    # not an execution failure and should not expose optimizer/ledger fields.
    if (record_data and record_data.get("final_synthesis_error") == "final_synthesis_provider_error"
            and record_data.get("succeeded") is True
            and record_data.get("actual_tool_trace") == []):
        return {"type": "response_generation_issue", "title": "Response generation issue", "data": {
            "problem_id": record_data.get("final_problem_id"),
            "message": "The request was processed, but the final response could not be generated.",
            "no_tools_executed": True,
        }}

    comparison = _completed_step(record_data, {"compare_solutions"})
    if comparison and isinstance(comparison.get("outcome_metadata"), Mapping):
        return {"type": "comparison", "title": "Solution comparison", "data": dict(comparison["outcome_metadata"])}

    optimization = _completed_step(record_data, {"run_optimization", "run_scenario"})
    if optimization and isinstance(optimization.get("outcome_metadata"), Mapping) and optimization.get("outcome_metadata"):
        data = dict(optimization["outcome_metadata"])
        view_type = "scenario" if optimization.get("expected_application_tool") == "run_scenario" else "optimization"
        return {"type": view_type, "title": "Decision result", "data": data}

    validation = _completed_step(record_data, {"validate_solution"})
    if validation and isinstance(validation.get("outcome_metadata"), Mapping):
        return {"type": "validation", "title": "Validation", "data": dict(validation["outcome_metadata"])}

    scenario = _completed_step(record_data, {"create_scenario"})
    if scenario and isinstance(scenario.get("outcome_metadata"), Mapping):
        return {"type": "scenario", "title": "Scenario", "data": dict(scenario["outcome_metadata"])}

    runs = _completed_step(record_data, {"list_runs"})
    if runs and isinstance(runs.get("outcome_metadata"), Mapping):
        return {"type": "historical_run", "title": "Run history", "data": dict(runs["outcome_metadata"])}

    baseline = _completed_step(record_data, {"get_baseline"})
    if baseline and isinstance(baseline.get("outcome_metadata"), Mapping):
        return {"type": "baseline", "title": "Current baseline", "data": dict(baseline["outcome_metadata"])}

    if record_data and approval_data and approval_data.get("plan_id") == record_data.get("plan_id") and approval_data.get("status") in {"approved", "rejected"}:
        return {"type": "approval_resolved", "title": "Approval decision", "data": {
            "problem_id": approval_data.get("problem_id"),
            "action": approval_data.get("tool_name"),
            "status": approval_data.get("status"),
            "reason_codes": approval_data.get("reason_codes", []),
        }}

    return {
        "type": "empty",
        "title": "Decision context",
        "data": {"message": "Run an analysis to populate the decision view."},
    }


def developer_details(agent: object) -> Dict[str, object]:
    record = getattr(agent, "last_plan_execution_record", None)
    plan = getattr(agent, "last_execution_plan", None)
    approval = getattr(agent, "last_approval_record", None)
    capability = getattr(agent, "last_capability_guard_decision", None)
    record_data = _record_dict(record)
    plan_data = _record_dict(plan)
    approval_data = _record_dict(approval)
    capability_data = _record_dict(capability)
    ledger: list[Dict[str, object]] = []
    if record_data:
        plan_id = record_data.get("plan_id")
        control = getattr(agent, "execution_control_ledger", None)
        entries_for_plan = getattr(control, "entries_for_plan", None)
        if plan_id and callable(entries_for_plan):
            for entry in entries_for_plan(str(plan_id)):
                entry_data = _record_dict(entry)
                if entry_data:
                    ledger.append(entry_data)
    plan_id = record_data.get("plan_id") if record_data else None
    if not plan_id or not plan_data or plan_data.get("plan_id") != plan_id:
        plan_data = None
    if not plan_id or not approval_data or approval_data.get("plan_id") != plan_id:
        approval_data = None
    blocked = bool(record_data and any(isinstance(step, Mapping) and
                   step.get("status") == "blocked_unsupported_requirement"
                   for step in record_data.get("steps", [])))
    return {
        "plan": plan_data,
        "execution_record": record_data,
        "capability_guard": capability_data if blocked else None,
        "approval": approval_data,
        "execution_ledger": ledger,
    }
