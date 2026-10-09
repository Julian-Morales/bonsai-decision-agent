"""Small interactive CLI for the existing Bonsai Decision Agent V0.

This module owns terminal I/O and application filesystem resolution only.  Provider
selection, model resolution, credentials, tool calling, and deterministic decisions
remain in ``bonsai_agent`` and ``bonsai_tools``.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Callable

from bonsai_agent import AgentConfig, BonsaiAgentContext, BonsaiDecisionAgentV0
from bonsai_problem_adapter import BonsaiProblemAdapter
from solution_store import GOLDEN_REFERENCE, SolutionStore


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILENAMES = {
    "catalog_path": "catalogo_productos.csv",
    "specs_path": "especificaciones_cajas.csv",
    "operations_path": "operaciones_planta.csv",
    "procurement_path": "procurement_cajas.csv",
}
GOLDEN_REFERENCE_FILENAME = "submission_target_subset_2dec.csv"


def build_agent_context(
    project_root: Path | None = None,
    output_dir: Path | None = None,
) -> BonsaiAgentContext:
    """Resolve project-owned source paths and create the shared agent output directory."""
    root = (project_root or PROJECT_ROOT).resolve()
    paths = {name: root / filename for name, filename in SOURCE_FILENAMES.items()}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing Bonsai source file(s):\n{missing_lines}")

    resolved_output = (output_dir or (root / "outputs" / "agent")).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    return BonsaiAgentContext(
        catalog_path=paths["catalog_path"],
        specs_path=paths["specs_path"],
        operations_path=paths["operations_path"],
        procurement_path=paths["procurement_path"],
        output_dir=resolved_output,
    )


def _read_solution_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def build_solution_store(
    context: BonsaiAgentContext,
    project_root: Path | None = None,
) -> SolutionStore:
    """Build one session store and register the independently validated golden reference."""
    root = (project_root or context.catalog_path.parent).resolve()
    golden_path = root / GOLDEN_REFERENCE_FILENAME
    if not golden_path.is_file():
        raise FileNotFoundError(f"Missing golden reference solution: {golden_path}")

    rows = _read_solution_rows(golden_path)
    adapter = BonsaiProblemAdapter(context)
    business_context = adapter.problem.business_parameters
    report = adapter.validate_solution(rows, business_context=business_context)
    if not bool(report.get("valid", False)):
        raise RuntimeError("Configured golden reference did not pass deterministic validation.")

    store = SolutionStore()
    store.register(
        GOLDEN_REFERENCE,
        rows,
        business_context=business_context,
        problem_id=adapter.problem_id,
    )
    return store


def show_tools_enabled() -> bool:
    """Return whether the optional compact tool trace is enabled."""
    value = (os.getenv("BONSAI_AGENT_SHOW_TOOLS") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_error_text(exc: BaseException) -> str:
    """Redact configured provider credentials from user-facing error text."""
    message = str(exc)
    for variable in ("OPENAI_API_KEY", "NVIDIA_API_KEY"):
        secret = os.getenv(variable)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def print_startup_banner(
    agent: BonsaiDecisionAgentV0,
    *,
    output_func: Callable[[str], None] = print,
) -> None:
    """Display only safe, already-resolved application configuration."""
    output_func("Bonsai Decision Agent V0")
    output_func(f"Provider: {agent.llm_config.provider}")
    output_func(f"Model: {agent.llm_config.model}")
    output_func("Type 'exit' or 'quit' to stop.")


def run_interactive_loop(
    agent: BonsaiDecisionAgentV0,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    show_tools: bool = False,
    show_plan: bool = False,
) -> None:
    """Run a request-local synchronous terminal loop around ``agent.run``."""
    while True:
        try:
            user_text = input_func("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_func("")
            break

        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        try:
            result = agent.run(user_text)
        except KeyboardInterrupt:
            output_func("")
            break
        except Exception as exc:  # turn-level API/provider failures should not kill the session
            output_func(f"Agent error: {_safe_error_text(exc)}")
            continue

        output_func(f"Agent: {result.text}")
        if show_tools and result.tool_names:
            output_func(f"Tools: {' -> '.join(result.tool_names)}")
        if show_plan:
            record = getattr(agent, "last_plan_execution_record", None)
            if record is not None:
                planned = " -> ".join(record.planned_tool_trace) or "(no application tools)"
                actual = " -> ".join(record.actual_tool_trace) or "(no application tools)"
                statuses = ", ".join(f"{step.step_id}:{step.status}" for step in record.steps)
                output_func(f"Plan {record.plan_id}: {planned}")
                output_func(f"Plan actual: {actual}")
                output_func(f"Plan status: {statuses or record.validation_error or 'completed'}")


def main() -> int:
    try:
        context = build_agent_context()
        solution_store = build_solution_store(context)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Startup error: {_safe_error_text(exc)}", file=sys.stderr)
        return 2

    try:
        config = AgentConfig()
        agent = BonsaiDecisionAgentV0(
            context,
            config=config,
            solution_store=solution_store,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Configuration error: {_safe_error_text(exc)}", file=sys.stderr)
        return 2

    print_startup_banner(agent)
    run_interactive_loop(agent, show_tools=show_tools_enabled())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
