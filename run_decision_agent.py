"""Unified interactive runner with Step 17 planning and Step 18 controlled execution.

Per-problem state remains isolated through the Step 16 allowlisted ProblemRegistry. Legacy
single-domain runners remain reactive with their original nine-tool behavior. Approval/rejection
and recoverable-resume commands are handled deterministically here and never sent to the planner.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

from bonsai_agent import AgentConfig, AgentRunResult, DecisionAgentV0
from bonsai_problem_adapter import BonsaiProblemAdapter
from multi_domain_state import MultiDomainSessionState
from problem_definition import BONSAI_PROBLEM_ID
from problem_registry import ProblemRegistry
from production_problem_adapter import ProductionPlanningAdapter
from run_agent import (
    _safe_error_text,
    build_agent_context,
    build_solution_store,
    show_tools_enabled,
)


def build_multi_domain_agent(*, config: AgentConfig | None = None, client=None) -> DecisionAgentV0:
    """Construct one controlled session with Bonsai active and isolated state for both domains."""
    bonsai_context = build_agent_context()
    bonsai_adapter = BonsaiProblemAdapter(bonsai_context)
    production_adapter = ProductionPlanningAdapter()
    registry = ProblemRegistry([bonsai_adapter, production_adapter])
    session = MultiDomainSessionState(
        registry,
        active_problem_id=BONSAI_PROBLEM_ID,
        solution_stores={BONSAI_PROBLEM_ID: build_solution_store(bonsai_context)},
    )
    return DecisionAgentV0(
        bonsai_context,
        problem_registry=registry,
        multi_domain_session=session,
        config=config or AgentConfig(),
        client=client,
        explicit_planning=True,
        controlled_execution=True,
    )


def show_plan_enabled() -> bool:
    value = (os.getenv("BONSAI_AGENT_SHOW_PLAN") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def show_control_enabled() -> bool:
    value = (os.getenv("BONSAI_AGENT_SHOW_CONTROL") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_control_command(agent: DecisionAgentV0, user_text: str) -> AgentRunResult | None:
    parts = user_text.strip().split()
    if not parts:
        return None
    command = parts[0].casefold()
    if command not in {"approve", "reject", "resume"}:
        return None
    if len(parts) > 2:
        return AgentRunResult(
            text=f"Invalid control command. Use '{command}' or '{command} <id>'.",
            tool_names=(),
        )
    identifier = parts[1] if len(parts) == 2 else None
    if command == "approve":
        return agent.approve(identifier)
    if command == "reject":
        return agent.reject(identifier)
    if identifier is None:
        return AgentRunResult(text="resume requires a plan_id, e.g. 'resume plan_000001'.", tool_names=())
    return agent.resume(identifier)


def _show_plan_state(agent: DecisionAgentV0, output_func: Callable[[str], None]) -> None:
    record = agent.last_plan_execution_record
    if record is None:
        return
    planned = " -> ".join(record.planned_tool_trace) or "(no application tools)"
    actual = " -> ".join(record.actual_tool_trace) or "(no application tools)"
    statuses = ", ".join(f"{step.step_id}:{step.status}" for step in record.steps)
    output_func(f"Plan {record.plan_id}: {planned}")
    output_func(f"Plan actual: {actual}")
    output_func(f"Plan status: {record.plan_status}")
    if statuses:
        output_func(f"Step status: {statuses}")


def _show_control_state(agent: DecisionAgentV0, output_func: Callable[[str], None]) -> None:
    approval = agent.last_approval_record
    if approval is not None:
        output_func(f"Approval: {approval.approval_id} {approval.status}")
    record = agent.last_plan_execution_record
    if record is None:
        return
    runtime = agent._controlled_plan_runtimes.get(record.plan_id)  # compact debug/CLI metadata only
    if runtime is None:
        return
    for step in runtime.plan.steps:
        ledger = agent.execution_control_ledger.get(runtime.plan.plan_id, step.step_id)
        if ledger.state != "not_started":
            output_func(f"Control {step.step_id}: {ledger.state}")


def run_controlled_interactive_loop(
    agent: DecisionAgentV0,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    show_tools: bool = False,
    show_plan: bool = False,
    show_control: bool = False,
) -> None:
    """Interactive loop with deterministic Step 18 control commands."""
    while True:
        try:
            user_text = input_func("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_func("")
            break
        if user_text.casefold() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        try:
            result = _run_control_command(agent, user_text)
            if result is None:
                result = agent.run(user_text)
        except KeyboardInterrupt:
            output_func("")
            break
        except Exception as exc:
            output_func(f"Agent error: {_safe_error_text(exc)}")
            continue

        output_func(f"Agent: {result.text}")
        if show_tools and result.tool_names:
            output_func(f"Tools: {' -> '.join(result.tool_names)}")
        if show_plan:
            _show_plan_state(agent, output_func)
        if show_control:
            _show_control_state(agent, output_func)


def main() -> int:
    try:
        agent = build_multi_domain_agent()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {_safe_error_text(exc)}", file=sys.stderr)
        return 2

    print("Operations Decision Agent V0 — Multi-Domain Controlled Execution")
    print(f"Active problem: {agent.problem_adapter.spec.display_name} ({agent.problem_adapter.problem_id})")
    print(f"Provider: {agent.llm_config.provider}")
    print(f"Model: {agent.llm_config.model}")
    print("Control commands: approve [approval_id], reject [approval_id], resume <plan_id>.")
    print("Type 'exit' or 'quit' to stop.")
    run_controlled_interactive_loop(
        agent,
        show_tools=show_tools_enabled(),
        show_plan=show_plan_enabled(),
        show_control=show_control_enabled(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
