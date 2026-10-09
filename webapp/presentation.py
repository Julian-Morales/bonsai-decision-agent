"""UX-2 business presentation: deterministic projection, never assistant-text interpretation.

The original assistant message and interaction trace are always retained separately.
No business result, failure state, or approval is inferred from prose.
"""
from __future__ import annotations

from typing import Mapping


def _amount(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    import math
    if not math.isfinite(value):
        return None
    if abs(value) >= 1_000_000:
        return f"USD {value / 1_000_000:,.2f}M"
    return f"USD {value:,.2f}"


def presentation(view: Mapping[str, object], cards: list[Mapping[str, object]], *, record: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build a concise primary interaction label from first-class structured state."""
    kind = str(view.get("type", "empty"))
    data = view.get("data") if isinstance(view.get("data"), Mapping) else {}
    types = {str(card.get("type")) for card in cards}
    if kind == "response_generation_issue":
        return {"kind": "response_generation_issue", "summary":
                "The request was processed, but the final response could not be generated. No tools were executed and no decision state changed.",
                "show_full_response": True}
    # Preserve a successful deterministic result even when narration failed.
    synthesis_warning = bool(record and record.get("succeeded") is True
                             and record.get("final_synthesis_error") == "final_synthesis_provider_error"
                             and record.get("actual_tool_trace")
                             and kind in {"optimization", "scenario", "validation", "comparison", "baseline"})
    if "approval_pending" in types:
        return {"kind": "approval_pending", "summary": "Approval is required before the requested action can proceed.", "show_full_response": True}
    if "approval_approved" in types or "approval_rejected" in types:
        # A resume may also have completed the actual optimization; preserve the result.
        if kind not in {"optimization", "scenario"}:
            return {"kind": "approval_resolved", "summary": "Approval decision recorded.", "show_full_response": True}
    if "invalid_domain_transition" in types or kind == "domain_transition":
        return {"kind": "domain_transition_rejected", "summary": "The requested operation targets a different problem. No execution occurred.", "show_full_response": True}
    if kind == "unsupported_requirement" and "blocked_unsupported_requirement" in types:
        return {"kind": "unsupported_requirement", "summary": "This requirement is not supported by the current model. No optimization was executed.", "show_full_response": True}
    if "outcome_unknown" in types:
        return {"kind": "execution_failure", "summary": "Execution outcome is unknown. Review the execution issue before retrying.", "show_full_response": True}
    if "error" in types:
        return {"kind": "execution_failure", "summary": "The request could not be completed. See the execution issue for dispatch status.", "show_full_response": True}
    if kind in {"optimization", "scenario"} and isinstance(data, Mapping) and isinstance(data.get("validated_final"), Mapping):
        validated = data["validated_final"]
        valid = validated.get("valid") is True
        statement = "Optimization completed and independently validated." if valid else "Optimization result available; review independent validation."
        total = _amount(validated.get("total") if validated.get("total") is not None else validated.get("total_cost"))
        if total:
            statement += f" Total cost: {total}."
        saving_pct = validated.get("saving_pct")
        if isinstance(saving_pct, (int, float)) and not isinstance(saving_pct, bool):
            import math
            if math.isfinite(saving_pct):
                statement += f" Savings: {saving_pct:,.2f}%."
        result = {"kind": "optimization_result", "summary": statement, "show_full_response": True}
        if synthesis_warning: result["synthesis_warning"] = "Final explanation could not be generated."
        return result
    if kind == "comparison":
        return {"kind": "comparison", "summary": "Comparison results are available in the Decision View.", "show_full_response": True}
    if kind == "validation":
        status = "passed" if data.get("valid") is True else "did not pass" if data.get("valid") is False else "completed"
        return {"kind": "validation_result", "summary": f"Independent validation {status}. Review the Decision View.", "show_full_response": True}
    if kind == "scenario" and isinstance(data, Mapping) and data:
        return {"kind": "scenario", "summary": "Scenario details are available in the Decision View.", "show_full_response": True}
    return {"kind": "informational", "summary": None, "show_full_response": False}
