"""Deterministic unsupported-requirement safety boundary for controlled execution.

The guard is application-owned and provider-independent.  It evaluates a structurally valid
execution plan against problem-local request capability rules before ApprovalPolicy can run.
It is deliberately narrow: this module is not a generic natural-language constraint engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from execution_plan import ExecutionPlan, PlanStep
from problem_registry import ProblemRegistry


GUARDED_MUTATING_ACTIONS = frozenset({"run_optimization", "run_scenario", "create_scenario"})
CAPABILITY_GUARD_DECISIONS = ("allowed", "blocked", "informational")


@dataclass(frozen=True)
class CapabilityGuardDecision:
    decision: str
    request_kind: str
    problem_id: str | None = None
    capability_id: str | None = None
    category: str | None = None
    reason_code: str | None = None
    safe_explanation: str | None = None
    affected_step_ids: Tuple[str, ...] = ()
    extracted_requirement: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.decision not in CAPABILITY_GUARD_DECISIONS:
            raise ValueError(f"Unsupported capability-guard decision: {self.decision!r}")

    @property
    def blocked(self) -> bool:
        return self.decision == "blocked"

    @property
    def informational(self) -> bool:
        return self.decision == "informational"

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision,
            "request_kind": self.request_kind,
            "problem_id": self.problem_id,
            "capability_id": self.capability_id,
            "category": self.category,
            "reason_code": self.reason_code,
            "safe_explanation": self.safe_explanation,
            "affected_step_ids": list(self.affected_step_ids),
            "extracted_requirement": dict(self.extracted_requirement or {}),
        }

    def user_text(self) -> str:
        if self.decision == "allowed" or not self.safe_explanation:
            return ""
        parts = [self.safe_explanation]
        extracted = dict(self.extracted_requirement or {})
        if "max_box_types" in extracted:
            parts.append(f"Requested bound: {extracted['max_box_types']} box types.")
        if self.problem_id:
            parts.append(f"Problem: {self.problem_id}.")
        if self.capability_id:
            parts.append(f"Capability: {self.capability_id}.")
        if self.informational:
            parts.append("No optimization or scenario mutation was executed to answer this capability question.")
        else:
            parts.append("Because that condition was required, the requested mutating action was not executed.")
        return " ".join(parts)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _is_information_only_request(text: str) -> bool:
    """Recognize narrow capability questions without mistaking polite action requests.

    In particular ``Can Bonsai ...?`` is informational while ``Can you optimize ...?`` is not.
    """

    value = _normalized(text)
    patterns = (
        r"^(?:can|could|does|do|is|are)\s+(?:the\s+current\s+)?(?:bonsai|production|this\s+problem|the\s+problem)\b",
        r"^is\s+it\s+possible\b",
        r"^(?:what|which)\s+(?:constraints|requirements|capabilities)\b",
    )
    return any(re.search(pattern, value) is not None for pattern in patterns)


def _information_problem_id(
    text: str,
    *,
    registry: ProblemRegistry,
    active_problem_id: str,
) -> str:
    """Resolve an informational capability question without dispatching select_problem."""

    value = _normalized(text)
    matches: list[str] = []
    for metadata in registry.metadata():
        aliases = {
            metadata.problem_id.casefold(),
            metadata.problem_id.replace("_", " ").casefold(),
            metadata.display_name.casefold(),
            metadata.display_name.split()[0].casefold(),
        }
        if any(re.search(rf"\b{re.escape(alias)}\b", value) for alias in aliases if alias):
            matches.append(metadata.problem_id)
    unique = tuple(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else active_problem_id


def _step_problem_targets(plan: ExecutionPlan) -> tuple[tuple[PlanStep, str], ...]:
    """Simulate validated domain transitions without dispatching any application action."""

    active = plan.initial_active_problem_id
    targeted: list[tuple[PlanStep, str]] = []
    for step in plan.steps:
        if step.expected_application_tool == "select_problem":
            if step.target_problem_id is not None:
                active = step.target_problem_id
            continue
        target = step.target_problem_id or active
        targeted.append((step, target))
    return tuple(targeted)


class RequirementCapabilityGuard:
    """Evaluate known problem-local unsupported requirements before approval/execution."""

    def assess(
        self,
        *,
        user_text: str,
        plan: ExecutionPlan,
        registry: ProblemRegistry,
        active_problem_id: str,
    ) -> CapabilityGuardDecision:
        information_only = _is_information_only_request(user_text)

        if information_only:
            problem_id = _information_problem_id(
                user_text,
                registry=registry,
                active_problem_id=active_problem_id,
            )
            adapter = registry.get(problem_id)
            for rule in adapter.spec.request_capability_rules:
                extracted = rule.match(user_text)
                if extracted is not None:
                    affected = tuple(
                        step.step_id
                        for step, target in _step_problem_targets(plan)
                        if target == problem_id
                        and step.expected_application_tool in rule.affected_action_types
                    )
                    return CapabilityGuardDecision(
                        decision="informational",
                        request_kind="informational",
                        problem_id=problem_id,
                        capability_id=rule.capability_id,
                        category=rule.category,
                        reason_code=rule.reason_code,
                        safe_explanation=rule.safe_explanation,
                        affected_step_ids=affected,
                        extracted_requirement=extracted,
                    )
            return CapabilityGuardDecision(decision="allowed", request_kind="informational")

        matched_problem: str | None = None
        matched_rule = None
        matched_requirement: Mapping[str, object] | None = None
        affected_steps: list[str] = []
        for step, target_problem_id in _step_problem_targets(plan):
            tool = step.expected_application_tool
            if tool not in GUARDED_MUTATING_ACTIONS:
                continue
            adapter = registry.get(target_problem_id)
            for rule in adapter.spec.request_capability_rules:
                if tool not in rule.affected_action_types:
                    continue
                extracted = rule.match(user_text)
                if extracted is None:
                    continue
                if matched_rule is None:
                    matched_problem = target_problem_id
                    matched_rule = rule
                    matched_requirement = extracted
                if matched_problem == target_problem_id and matched_rule.capability_id == rule.capability_id:
                    affected_steps.append(step.step_id)

        if matched_rule is None:
            # A planner may misunderstand the request so badly that it proposes no mutating
            # action at all.  Required unsupported conditions still receive a deterministic
            # fail-closed answer rather than falling back to model narration.  Resolve a named
            # problem from the user text first, then a validated select_problem target, then the
            # currently active problem.
            fallback_problem = _information_problem_id(
                user_text,
                registry=registry,
                active_problem_id=active_problem_id,
            )
            select_targets = [
                step.target_problem_id
                for step in plan.steps
                if step.expected_application_tool == "select_problem" and step.target_problem_id is not None
            ]
            if fallback_problem == active_problem_id and select_targets:
                fallback_problem = str(select_targets[-1])
            adapter = registry.get(fallback_problem)
            for rule in adapter.spec.request_capability_rules:
                extracted = rule.match(user_text)
                if extracted is not None:
                    return CapabilityGuardDecision(
                        decision="blocked",
                        request_kind="required_action",
                        problem_id=fallback_problem,
                        capability_id=rule.capability_id,
                        category=rule.category,
                        reason_code=rule.reason_code,
                        safe_explanation=rule.safe_explanation,
                        affected_step_ids=(),
                        extracted_requirement=extracted,
                    )
            return CapabilityGuardDecision(decision="allowed", request_kind="action")

        return CapabilityGuardDecision(
            decision="blocked",
            request_kind="required_action",
            problem_id=matched_problem,
            capability_id=matched_rule.capability_id,
            category=matched_rule.category,
            reason_code=matched_rule.reason_code,
            safe_explanation=matched_rule.safe_explanation,
            affected_step_ids=tuple(dict.fromkeys(affected_steps)),
            extracted_requirement=matched_requirement,
        )


__all__ = [
    "CAPABILITY_GUARD_DECISIONS",
    "GUARDED_MUTATING_ACTIONS",
    "CapabilityGuardDecision",
    "RequirementCapabilityGuard",
]
