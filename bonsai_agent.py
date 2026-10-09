"""Single-LLM Decision Agent V0 for Bonsai Corp.

This application layer is intentionally small. Legacy single-domain mode exposes nine
business-level capabilities; Step 16 multi-domain mode adds only two registry controls. Complete
solution rows remain in session-local Python state. Deterministic
work stays delegated through an application-level optimization-problem adapter to deterministic
domain services. No optimizer or validator implementation imports this module, ``solution_store``,
or the OpenAI SDK.
"""
from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol

from optimization_problem import OptimizationProblemError
from capability_guard import CapabilityGuardDecision, RequirementCapabilityGuard
from problem_adapter import OptimizationProblemAdapter, business_contexts_compatible
from bonsai_problem_adapter import BonsaiProblemAdapter
from problem_definition import OptimizationProblemSpec, bonsai_problem_spec
from problem_registry import ProblemNotRegisteredError, ProblemRegistry
from multi_domain_state import MultiDomainSessionState
from execution_plan import (
    ArgumentBinding,
    DeferredSelector,
    DEFERRED_OUTPUT_CONTRACTS,
    ExecutionPlan,
    MAX_PLAN_STEPS,
    PlanExecutionRecord,
    PlanStep,
    PlanStepExecutionSummary,
    PlanValidationError,
    PlanValidator,
    SessionEntityContext,
    SessionEntityRecord,
)
from execution_control import (
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalStore,
    ExecutionControlError,
    ExecutionControlLedger,
    PlanControlNotFoundError,
    emit_control_debug,
    stable_fingerprint,
)
from llm_provider import (
    ExecutorTransportUnsupportedError,
    PlannerTransportUnsupportedError,
    build_executor_request_options,
    build_planner_request_options,
    emit_executor_protocol_debug,
    emit_planner_protocol_debug,
    executor_protocol_debug_payload,
    normalize_executor_response,
    normalize_planner_response,
)
from solution_store import (
    GOLDEN_REFERENCE,
    LATEST_OPTIMIZER,
    SolutionNotFoundError,
    SolutionStore,
    SolutionStoreError,
    is_supported_solution_handle,
)
from run_history import RunHistoryStore
from scenario_store import (
    InvalidScenarioNameError,
    ScenarioExistsError,
    ScenarioNotFoundError,
    ScenarioStore,
    scenario_solution_handle,
)


class _LazyBonsaiToolsProxy:
    """Backward-compatible test hook without eager Bonsai optimizer imports."""

    @staticmethod
    def _module():
        import bonsai_tools as module
        return module

    def __getattr__(self, name: str) -> object:
        return getattr(self._module(), name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._module(), name, value)


bonsai_tools = _LazyBonsaiToolsProxy()


NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_SAFE_OPAQUE_SOLUTION_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_HISTORICAL_REFERENCE_PHRASES = (
    "historical snapshot",
    "immutable snapshot",
    "historical version",
)


class FinalSynthesisProviderError(RuntimeError):
    """Raised only when the provider call for tool-free final synthesis fails."""


_SYSTEM_INSTRUCTIONS_BASE = """You are an Operations Decision Agent V0 bound to one application-provided deterministic optimization problem.

SUPPORTED CAPABILITIES
- Audit the active problem's supplied deterministic data through audit_data.
- Retrieve the active problem's current deterministic baseline through get_baseline.
- Run the active deterministic optimizer using only parameters declared by the active problem definition.
- Create and run named session-local scenarios composed only of declared parameter overrides.
- Distinguish business/problem parameters from solver parameters when explaining overrides.
- Validate solutions already registered under supported solution handles.
- Compare two already-registered solutions under compatible stored provenance.
- Inspect compact session-local optimization run history.
- Explain deterministic tool results in business language.
- Perform transparent simple arithmetic derived from tool-established values when useful.

UNSUPPORTED CAPABILITIES
Do not claim support for parameters, objectives, constraints, source-data changes, optimization domains,
or mathematical formulations that are not declared by the active OptimizationProblem specification.
Do not generate optimization code, arbitrary constraints, or an optimization model. Do not validate
arbitrary user solutions or spreadsheets. Do not compare arbitrary or unregistered solutions.
Do not advertise capabilities beyond audit_data, get_baseline, run_optimization, validate_solution,
compare_solutions, create_scenario, run_scenario, list_scenarios, and list_runs. The active problem
and problem_id are application-provided; this runtime session cannot switch to another domain.

GROUNDING AND EVIDENCE
Use deterministic tools for numerical or business facts when a tool can answer them. Never invent
optimization results, costs, validation outcomes, causes, parameter support, or capabilities. Distinguish
clearly among (1) a tool-established fact, (2) arithmetic derived from tool-established facts, and (3)
a hypothesis. If you offer a possible business explanation not established by the tools, label it
explicitly as a hypothesis/example and state that available data does not establish the cause.
Preserve units returned by tools and never invent units.
When comparing registered solutions, treat deterministic deltas as descriptive facts only. Do not infer
causality unless the deterministic output explicitly establishes it. If compare_solutions reports
incompatible_problem, explain that the stored solutions belong to incompatible optimization problems.
If it reports incompatible_business_context, do not invent normalized deltas; explain that the solutions
were evaluated under different business parameter contexts.

STATIC SENSITIVITY VS PARAMETER OVERRIDES
For a simple arithmetic change that is not a supported optimizer parameter, you may perform arithmetic
only from tool-established values when it directly answers the request. Label it as static sensitivity,
state what is held constant, and state that the optimizer was NOT rerun under modified assumptions.
For a declared configurable parameter, use run_optimization with an explicit override when optimization
is requested. Never reinterpret an unsupported structural change as a supported parameter.

OPTIMIZATION AND VALIDATION LANGUAGE
Distinguish the baseline, optimizer output, and independently validated stored solution. Independent
validation means the solution passed the active problem's deterministic independent validator. It does
not prove global optimality, implementation readiness, contractual feasibility, organizational acceptance,
or risk-free implementation. Describe passing results conservatively and never claim global optimality
unless the deterministic domain explicitly establishes such a guarantee.

TOOL USE
Do not run the optimizer when audit_data or get_baseline is sufficient. Use run_optimization when the user
explicitly asks to optimize/re-optimize under current assumptions or with a declared parameter override.
Use validate_solution and compare_solutions only with registered handles. latest_optimizer means the most
recent successful independently validated base optimization in the CURRENT application session; it does
not mean run the optimizer now. If it exists, use it directly rather than rerunning merely to refresh it.
Never ask the user for raw solution rows or filesystem paths. If a request requires unsupported changed
assumptions, do not run the unchanged optimizer and pretend it represents that request.

SCENARIOS
Scenarios are session-local named configurations composed only of parameter overrides declared by the
active OptimizationProblem. Creating a scenario does not optimize it. Running one uses the same active
deterministic adapter and independent validator. Scenario solutions use application-generated
scenario_solution_* handles and do not update latest_optimizer.

RUN HISTORY
Every actual base or scenario optimization attempt receives an application-generated sequential run_id.
Successful independently validated attempts receive immutable run_solution_* historical handles.
latest_optimizer and scenario_solution_* are current aliases; run_solution_* handles are immutable.
Use list_runs for compact metadata. Never invent run IDs or historical solution handles. Failed attempts
have no historical solution handle and must not replace a previous valid alias.

"""



_BONSAI_COMPATIBILITY_GROUNDING = 'You are the Bonsai Corp Operations Decision Agent V0.\n\nSUPPORTED CAPABILITIES\n- Retrieve the current baseline packaging, freight, and total economics.\n- Audit the supplied Bonsai source data, including Operations-versus-Procurement consistency.\n- Run the existing deterministic optimizer with explicit parameters declared by the Bonsai problem definition.\n- Named session-local scenarios composed only of declared parameter overrides are supported; generic what-if/scenario optimization beyond those declared parameters is not.\n- Distinguish business/problem parameters from solver parameters when explaining overrides.\n- Validate solutions that are already registered under the supported solution handles.\n- Compare two already-registered solutions by supported handle using deterministic validation/economics.\n- Explain deterministic tool results in business language.\n- Perform transparent simple arithmetic derived from tool-established values when useful.\n\nUNSUPPORTED CAPABILITIES\nDo not claim you can modify freight assumptions outside the declared pallet_cost parameter, demand, cardboard prices,\ncarriers/routes, the maximum number of box types, the fixed 3.0-mm default, or arbitrary mathematical constraints. Only use parameter overrides explicitly declared by the Bonsai problem\ndefinition. A parameter override is not a generic scenario: it changes one declared configuration value and\nre-runs the existing deterministic optimizer. Do not invent unsupported parameters or values.\nDo not validate arbitrary user solutions or spreadsheets. Do not compare arbitrary solutions or unregistered solutions.\nDo not advertise capabilities beyond audit_data, get_baseline, run_optimization, validate_solution,\ncompare_solutions, create_scenario, run_scenario, list_scenarios, and list_runs. OptimizationProblem is the authoritative definition\nfor configurable optimization parameters; unsupported user requests must be rejected rather than approximated. The active optimization problem and its problem_id come from application-provided problem metadata; do not claim to solve an unrelated optimization domain.\n\nGROUNDING AND EVIDENCE\nUse deterministic tools for numerical or business facts when a tool can answer them. Never invent\noptimization results, costs, validation outcomes, causes, or capabilities. Distinguish clearly among\n(1) a tool-established fact, (2) arithmetic derived from tool-established facts, and (3) a hypothesis.\nIf you offer a possible business explanation not established by the tools, label it explicitly as a\nhypothesis/example and state that the available data does not establish the cause. Never upgrade a\nhypothesis into a conclusion. Do not assign a more specific meaning or cause to a deterministic field,\ncount, ERROR marker, or flag than the tool output establishes; for example, an is_error=false field does\nnot prove that a mismatch is intentional, legitimate, commercially justified, or correct. Preserve units\nreturned by tools. Current characterized Bonsai cost values are USD; never invent EUR or another unit,\nand never invent a unit for a field that has none.\nWhen comparing registered solutions, treat counts and cost deltas as descriptive facts only. A difference\nin box types, packaging cost, freight, pallets, or total cost does NOT by itself establish that one difference\ncaused another. State the observed values without causal wording unless the deterministic comparison output\nexplicitly establishes that causal relationship. A possible causal explanation may still be offered as a\nhypothesis, but it must remain explicitly framed as a hypothesis and state that the comparison does not\nestablish causality. If compare_solutions reports incompatible_problem, do not pass or reinterpret the foreign solution as Bonsai; explain that the stored solutions belong to incompatible optimization problems. If compare_solutions reports incompatible_business_context, do not invent normalized\neconomic deltas; explain that the stored solutions were evaluated under different business parameter contexts.\n\nSTATIC SENSITIVITY VS PARAMETER OVERRIDES\nFor a simple derived change that is not a supported optimizer parameter, you may perform the arithmetic,\nbut label it as a static sensitivity calculation (or equivalent), state which quantities are held constant,\nand state that the optimizer was NOT rerun under modified assumptions. For a declared configurable parameter,\nuse run_optimization with the explicit parameter override instead. Do not present static arithmetic as an\noptimized result. Demand changes, freight-model changes outside pallet_cost, and other unsupported scenarios\nremain unsupported.\n\nOPTIMIZATION AND VALIDATION LANGUAGE\nDistinguish the baseline, optimizer candidate, and independently validated final solution. The Bonsai problem\ndefinition declares 3.0 mm as the default thickness and permits only its explicitly supported thickness values\nas parameter overrides; do not describe an override as a new mathematical constraint.\nIndependent validation means the solution passed the encoded independent validator. It does not prove\nglobal optimality, production/manufacturing readiness, contractual feasibility, organizational\nacceptance, or risk-free implementation. Describe a passing output as a validated candidate/solution\nfor business review, not automatically as implementation-ready. Never claim global optimality.\n\nTOOL USE\nDo not run the optimizer when audit_data or get_baseline is sufficient. Use run_optimization when the\nuser explicitly asks to optimize/re-optimize under the current assumptions, or asks to optimize with\na declared parameter override. Apply only parameter values accepted by the problem definition. Use validate_solution only for a supported registered handle, and\nuse compare_solutions only for two supported registered handles.\nIMPORTANT: latest_optimizer means the most recent successfully completed and independently validated\noptimization result already registered in the CURRENT application session. It does NOT mean "run the\noptimizer now". If latest_optimizer exists, do not run run_optimization merely to refresh, recreate, or\nconfirm that handle before validate_solution or compare_solutions. Use the existing handle directly.\nNever ask the user to provide raw solution rows or filesystem paths to these tools. If the requested re-optimization requires unsupported changed assumptions, do not run the unchanged optimizer\nand pretend it represents that scenario. When the request is clearly unsupported, answer the capability\nlimitation directly; call a tool only when its result is\nrequired by the user\'s actual request, not merely to add related context to a refusal. If the user separately\nasks for supported information such as the current baseline, use the relevant tool for that part. You may\nanswer conversational or meta questions directly when no deterministic project data is required.\n\nScenarios are session-local named configurations composed only of parameter overrides already declared\nby OptimizationProblem. Creating a scenario does not optimize it. Running a scenario uses the existing\ndeterministic optimizer and independent validation. Scenarios do not permit arbitrary demand changes,\nobjectives, constraints, freight models, or optimization code. Scenario solutions are registered under\napplication-generated scenario_solution_* handles and may be passed to validate_solution or\ncompare_solutions exactly like other registered solution handles.\n\nRUN HISTORY\nEvery actual base or scenario optimization attempt receives an application-generated sequential run_id.\nSuccessful independently validated attempts also receive an immutable run_solution_* historical handle.\nlatest_optimizer and scenario_solution_* are current aliases; run_solution_* handles are immutable historical\nsnapshots. Scenario runs never update latest_optimizer. Use list_runs to inspect compact session-local run\nmetadata. Historical handles may be validated or compared only when they were actually returned by the\napplication; never invent run IDs or historical solution handles. Failed attempts have no historical solution\nhandle and must not replace a previous valid alias.\n\n'

_APPLICATION_CONSISTENCY_GROUNDING = """

REFERENCE-CONTEXT CONSISTENCY
A baseline/reference and an optimization result are like-for-like only when their stored business/problem
parameter contexts are compatible. If a tool reports reference_context_compatible=false, you may report
the result's absolute economics and describe the differing contexts, but do NOT calculate or report savings,
percentage savings, normalized deltas, or other economic comparisons against that incompatible reference.
Do not imply that an allocation/portfolio difference was caused by a parameter change unless a deterministic
tool explicitly establishes that causal claim.

MULTI-STEP REFERENCE RESOLUTION
When the user's requested validation or comparison refers indirectly to a registered solution and a handle
can be resolved with list_runs or list_scenarios, that lookup is an intermediate step, not completion of the
request. After a successful lookup, continue in the SAME agent run with validate_solution or compare_solutions
as requested. Stop after the lookup only when the user asked merely to list, identify, or inspect the reference.
A resolved immutable/historical handle may refer to the same stored solution as a current alias. That is still
a valid comparison target: do not require an earlier or distinct version, and still complete the requested
comparison. Equality/zero deltas are valid deterministic comparison results. Never invent an unresolved handle.

FAST-MODE EXECUTION INTENT
The fast argument is an execution-mode choice and must preserve the user's intent. Set fast=true only when the
user explicitly requests a fast, quick, reduced, or abbreviated optimization mode, or clearly approves such a
mode. For ordinary optimization requests such as "Optimize costs", "Run the optimizer", "Optimize with defaults",
or "Run the default optimization", set fast=false. Never infer fast=true merely to reduce model latency,
optimizer runtime, or execution cost. Apply the same rule to run_optimization and run_scenario.

FAST-MODE INTERPRETATION
Optimization tool results expose requested_fast, fast_supported, and effective_fast. Only describe a run as
executed in fast mode when effective_fast=true. requested_fast=true only records the user's request; it does
not prove that fast execution occurred. If requested_fast=true, fast_supported=false, and effective_fast=false,
state that fast mode is unsupported for the active problem and the normal pipeline was used.

PARAMETERIZED-RESULT NARRATION: DESCRIPTIVE BY DEFAULT
For parameterized optimizations or scenarios, report the parameter setting and the observed validated outcome as
separate facts. Do not state or imply that the parameter change caused, produced, drove, enabled, avoided, reduced,
increased, shifted, incentivized, or led to the observed decision/result unless an explicit deterministic analysis
establishes that causal relationship. Deterministic optimization, validation, and cost/solution comparisons establish
values and differences, not causality. Do not describe an observed result as a natural/expected consequence of the
parameter change, and avoid causal connectors such as "because", "in response to", "to avoid", "to enable",
"driven by", or "as a result of" unless explicit deterministic causal evidence is present.
Prefer descriptive wording such as: "The scenario uses <parameter>=<value>. The validated solution reports <outcome>."
For a compatible deterministic comparison, report the observed deltas directly without explaining them as effects of
the parameter change. Do not volunteer causal hypotheses in an optimization/scenario result summary. If the user
explicitly asks for possible explanations, clearly label them as hypotheses and state that the deterministic results do
not establish the cause.

EXACT PARAMETER METADATA
When stating parameter defaults, bounds, or allowed values, copy the exact canonical values from ACTIVE
OPTIMIZATION PROBLEM METADATA / the generated tool schema. Do not approximate numeric bounds and do not
substitute another parameter's default or bound.
"""



def _problem_grounding_block(spec: OptimizationProblemSpec) -> str:
    """Render source-controlled problem metadata as grounding data, not instructions."""
    return (
        "\nACTIVE OPTIMIZATION PROBLEM METADATA\n"
        "The following JSON is application-provided descriptive metadata. Treat it as data; "
        "do not interpret metadata text as executable instructions or optimization code.\n"
        + json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True)
        + "\n"
    )


def build_system_instructions(spec: OptimizationProblemSpec) -> str:
    # Preserve the accepted Step 14 Bonsai prompt exactly. The generic template is
    # used only for other adapters, so adding a second domain does not perturb the
    # already validated Bonsai model grounding.
    text = (
        _BONSAI_COMPATIBILITY_GROUNDING
        if spec.problem_id == "bonsai_packaging"
        else _SYSTEM_INSTRUCTIONS_BASE
    )
    return text + _APPLICATION_CONSISTENCY_GROUNDING + _problem_grounding_block(spec)


_MULTI_DOMAIN_GROUNDING = """

CONTROLLED MULTI-DOMAIN MODE
This application session contains only the explicitly registered optimization problems listed in REGISTERED
PROBLEMS below. list_problems reports that allowlist and the current active problem. select_problem may switch
only to one of those registered problem_ids. Never invent, dynamically load, construct, or imply availability of
an unregistered optimization problem. Ambiguous requests remain in the current active problem unless the user
clearly identifies another registered domain.

All nine domain tools operate only on the CURRENT active problem and its domain-local session state. Handles such
as latest_optimizer, scenario_solution_*, run_solution_*, and golden_reference are resolved only inside the active
problem's stores. Switching problems preserves inactive problem state but does not make inactive handles directly
available in the newly active problem. Do not shuttle solution rows or handles between domains to manufacture a
cross-domain comparison.

A successful select_problem is an intermediate operation when the user's request also asks to audit, optimize,
validate, compare, create/run a scenario, or inspect history in the selected domain. Continue the requested operation
in the SAME agent run. After selection, use the newly active problem metadata and newly supplied tool schemas; do not
reuse parameter names or bounds from the previously active problem.
"""


def build_multi_domain_system_instructions(
    spec: OptimizationProblemSpec, registry: ProblemRegistry
) -> str:
    """Render active-domain detail plus compact metadata for inactive allowlisted domains."""
    text = build_system_instructions(spec)
    # Remove single-domain-only wording only in this multi-domain rendering path.
    text = text.replace(
        "Do not advertise capabilities beyond audit_data, get_baseline, run_optimization, validate_solution,\n"
        "compare_solutions, create_scenario, run_scenario, list_scenarios, and list_runs.",
        "Do not advertise capabilities beyond the nine active-domain tools plus list_problems and select_problem.",
    )
    text = text.replace(
        "The active problem\nand problem_id are application-provided; this runtime session cannot switch to another domain.",
        "The active problem and problem_id are application-controlled and may change only through select_problem "
        "using the explicit registered allowlist.",
    )
    text = text.replace(
        "The active optimization problem and its problem_id come from application-provided problem metadata; do not claim to solve an unrelated optimization domain.",
        "The active optimization problem and problem_id are application-controlled. In multi-domain mode, "
        "select_problem may choose another registered adapter; never claim the current adapter itself solves an unrelated domain.",
    )
    registered = [metadata.to_dict() for metadata in registry.metadata()]
    registry_block = (
        "\nREGISTERED PROBLEMS (compact allowlist metadata)\n"
        + json.dumps(registered, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    return text + _MULTI_DOMAIN_GROUNDING + registry_block


SYSTEM_INSTRUCTIONS = build_system_instructions(bonsai_problem_spec())


@dataclass(frozen=True)
class BonsaiAgentContext:
    """Application-owned filesystem configuration hidden from LLM tool arguments."""

    catalog_path: Path
    specs_path: Path
    operations_path: Path
    procurement_path: Path
    output_dir: Path


@dataclass(frozen=True)
class LLMProviderConfig:
    """Provider/model routing only. Credentials are intentionally read at client creation."""

    provider: str
    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        if provider not in {"openai", "nvidia"}:
            raise ValueError(f"Proveedor LLM no soportado: {self.provider!r}")
        model = self.model.strip()
        if not model:
            raise ValueError("model no puede estar vacío")
        base_url = self.base_url.strip() if isinstance(self.base_url, str) else self.base_url
        if base_url == "":
            base_url = None
        if provider == "nvidia" and base_url is None:
            base_url = NVIDIA_NIM_BASE_URL

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_env(cls, *, model_override: str | None = None) -> "LLMProviderConfig":
        """Resolve provider settings with canonical Bonsai variables first.

        Precedence for model selection:
        1. explicit ``model_override``;
        2. ``BONSAI_LLM_MODEL``;
        3. ``OPENAI_MODEL`` for the OpenAI provider only;
        4. the historical OpenAI default ``gpt-5.6``.

        NVIDIA intentionally has no implicit model default because model-level tool/Responses
        capabilities differ across the NVIDIA catalog.
        """
        provider = (os.getenv("BONSAI_LLM_PROVIDER") or "openai").strip().lower()
        custom_base_url = os.getenv("BONSAI_LLM_BASE_URL")

        if model_override is not None:
            model = model_override
        else:
            model = os.getenv("BONSAI_LLM_MODEL")
            if not model and provider == "openai":
                model = os.getenv("OPENAI_MODEL") or "gpt-5.6"
            if not model and provider == "nvidia":
                raise ValueError(
                    "BONSAI_LLM_MODEL is required when BONSAI_LLM_PROVIDER=nvidia."
                )

        return cls(provider=provider, model=str(model), base_url=custom_base_url)


@dataclass(frozen=True)
class AgentConfig:
    """LLM orchestration settings only; no business or optimizer scenario parameters."""

    model: str | None = None
    max_tool_iterations: int = 6
    llm: LLMProviderConfig | None = None

    def __post_init__(self) -> None:
        resolved = self.llm or LLMProviderConfig.from_env(model_override=self.model)
        if self.llm is not None and self.model is not None and self.model.strip() != resolved.model:
            raise ValueError("AgentConfig.model conflicts with AgentConfig.llm.model")
        if self.max_tool_iterations < 1:
            raise ValueError("max_tool_iterations debe ser al menos 1")
        object.__setattr__(self, "llm", resolved)
        object.__setattr__(self, "model", resolved.model)


@dataclass(frozen=True)
class AgentRunResult:
    """Minimal application-visible result useful for testing and later evaluation."""

    text: str
    tool_names: tuple[str, ...]


@dataclass
class _AgentRunState:
    optimizer_calls: int = 0
    solution_store: SolutionStore = field(default_factory=SolutionStore)
    scenario_store: ScenarioStore = field(default_factory=ScenarioStore)
    run_history_store: RunHistoryStore = field(default_factory=RunHistoryStore)
    problem_adapter: OptimizationProblemAdapter | None = None


@dataclass
class _MultiDomainAgentRunState:
    """Request-local execution counter over persistent per-problem session state."""

    session: MultiDomainSessionState
    optimizer_calls: int = 0

    @property
    def problem_adapter(self) -> OptimizationProblemAdapter:
        return self.session.active.adapter

    @property
    def solution_store(self) -> SolutionStore:
        return self.session.active.solution_store

    @property
    def scenario_store(self) -> ScenarioStore:
        return self.session.active.scenario_store

    @property
    def run_history_store(self) -> RunHistoryStore:
        return self.session.active.run_history_store


@dataclass
class _ControlledPlanRuntime:
    """Session-local resumable Step 18 runtime for one already-validated plan."""

    plan: ExecutionPlan
    user_message: str
    initial_problem_id: str
    state: _AgentRunState | _MultiDomainAgentRunState
    plan_fingerprint: str
    actual_tool_trace: list[str] = field(default_factory=list)
    completed_outputs: list[Dict[str, object]] = field(default_factory=list)
    step_summaries: Dict[str, PlanStepExecutionSummary] = field(default_factory=dict)
    plan_status: str = "running"
    pending_approval_id: str | None = None
    recoverable_step_id: str | None = None
    turn_open: bool = True


class _ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ModelClient(Protocol):
    responses: _ResponsesAPI


ToolExecutor = Callable[[object, Mapping[str, object], object], Dict[str, object]]


@dataclass(frozen=True)
class AgentToolDefinition:
    schema: Dict[str, object]
    execute: ToolExecutor


def _empty_parameters() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def _adapter_for(context: object, state: _AgentRunState) -> OptimizationProblemAdapter:
    adapter = state.problem_adapter
    if adapter is None:
        adapter = BonsaiProblemAdapter(context)
        state.problem_adapter = adapter
    return adapter


def _execute_audit_data(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    del arguments
    return _adapter_for(context, state).audit_data()


def _execute_get_baseline(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    del arguments
    return _adapter_for(context, state).get_baseline()


def _problem_parameter_schema(
    spec: OptimizationProblemSpec | None = None,
) -> Dict[str, object]:
    projected = spec or bonsai_problem_spec()
    return {
        parameter.name: {
            "category": parameter.category,
            "default": parameter.default,
            "description": parameter.description,
            "allowed_values": list(parameter.allowed_values) if parameter.allowed_values is not None else None,
            "minimum": parameter.minimum,
            "maximum": parameter.maximum,
        }
        for parameter in projected.parameters
    }


def _parameter_override_tool_schema(
    spec: OptimizationProblemSpec | None = None,
    *,
    include_description: bool,
) -> Dict[str, object]:
    """Derive the strict override object from the projected executable parameter model."""
    projected = spec or bonsai_problem_spec()
    json_types = {"float": "number", "int": "integer", "str": "string", "bool": "boolean"}
    properties: Dict[str, object] = {}
    required: list[str] = []
    for parameter in projected.parameters:
        value_type = json_types.get(parameter.value_type)
        if value_type is None:
            raise RuntimeError(f"Unsupported parameter type in problem spec: {parameter.value_type}")
        property_schema: Dict[str, object] = {
            "type": [value_type, "null"],
            "description": parameter.description,
        }
        if parameter.minimum is not None:
            property_schema["minimum"] = parameter.minimum
        if parameter.maximum is not None:
            property_schema["maximum"] = parameter.maximum
        if parameter.allowed_values is not None:
            property_schema["enum"] = [*parameter.allowed_values, None]
        properties[parameter.name] = property_schema
        required.append(parameter.name)
    schema: Dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if include_description:
        schema["description"] = (
            "Explicit overrides declared by the active optimization problem. Use null for "
            "parameters that are not being overridden. Business/problem parameters and solver "
            "parameters are intentionally distinguished by the problem definition."
        )
    return schema


def _effective_fast_mode(adapter: OptimizationProblemAdapter, requested: bool) -> bool:
    """Normalize the shared fast flag to behavior the active adapter actually supports."""
    return bool(requested) if adapter.supports_fast_mode else False



def _execute_run_optimization(
    context: object,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    if state.optimizer_calls >= 1:
        return {
            "success": False,
            "error": {
                "code": "optimizer_already_run",
                "message": "run_optimization may execute at most once per agent run.",
            },
        }

    overrides = {
        name: value
        for name, value in (arguments.get("parameter_overrides") or {}).items()
        if value is not None
    }
    adapter = _adapter_for(context, state)
    resolved_problem = adapter.resolve_problem(overrides)
    business_context = resolved_problem.business_parameters
    requested_fast = bool(arguments["fast"])
    fast = _effective_fast_mode(adapter, requested_fast)
    run_id = state.run_history_store.begin(
        run_kind="base",
        fast=fast,
        parameter_overrides=overrides,
        problem_id=adapter.problem_id,
    )
    state.optimizer_calls += 1
    try:
        result = adapter.run_optimizer(fast=fast, parameter_overrides=overrides)
        outcome = adapter.interpret_optimizer_result(result)
    except Exception:
        state.run_history_store.fail(run_id, error_code="optimizer_exception")
        raise

    registered_handle: str | None = None
    historical_handle: str | None = None
    if outcome.validated_success and outcome.rows:
        try:
            historical_handle = state.run_history_store.solution_handle_for(run_id)
            state.solution_store.register_run_result(
                historical_handle,
                LATEST_OPTIMIZER,
                outcome.rows,
                business_context=business_context,
                problem_id=adapter.problem_id,
            )
            registered_handle = LATEST_OPTIMIZER
        except (TypeError, SolutionStoreError):
            historical_handle = None
            registered_handle = None

    if historical_handle is not None and registered_handle is not None:
        state.run_history_store.complete(
            run_id,
            solution_handle=historical_handle,
            current_alias=registered_handle,
        )
    else:
        state.run_history_store.fail(
            run_id,
            error_code=outcome.failure_code or "solution_registration_failed",
        )

    compact = dict(outcome.compact_result)
    compact["requested_fast"] = requested_fast
    compact["fast_supported"] = adapter.supports_fast_mode
    compact["effective_fast"] = fast
    compact["fast_mode"] = {
        "requested": requested_fast,
        "effective": fast,
        "supported": adapter.supports_fast_mode,
    }
    if registered_handle is not None:
        compact["solution_handle"] = registered_handle
    compact["run_id"] = run_id
    compact["historical_solution_handle"] = historical_handle
    return compact

def _compact_scenario_record(record: object) -> Dict[str, object]:
    definition = record.definition
    return {
        "handle": definition.handle,
        "name": definition.name,
        "parameter_overrides": dict(definition.parameter_overrides),
        "solution_handle": record.solution_handle,
        "optimized": record.solution_handle is not None,
    }


def _execute_create_scenario(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    adapter = _adapter_for(context, state)
    if state.scenario_store.problem_id != adapter.problem_id:
        return {
            "success": False,
            "error": "incompatible_problem",
            "active_problem_id": adapter.problem_id,
            "scenario_problem_id": state.scenario_store.problem_id,
        }
    overrides = {
        name: value
        for name, value in (arguments.get("parameter_overrides") or {}).items()
        if value is not None
    }
    try:
        record = state.scenario_store.create(str(arguments["name"]), overrides)
    except (OptimizationProblemError, InvalidScenarioNameError, ScenarioExistsError) as exc:
        return {
            "success": False,
            "error": {"code": "scenario_creation_failed", "message": str(exc)},
        }
    return {"success": True, "scenario": _compact_scenario_record(record)}


def _execute_list_scenarios(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    del context, arguments
    records = state.scenario_store.list()
    return {
        "success": True,
        "count": len(records),
        "scenarios": [_compact_scenario_record(record) for record in records],
    }


def _execute_run_scenario(
    context: object,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    if state.optimizer_calls >= 1:
        return {
            "success": False,
            "error": {
                "code": "optimizer_already_run",
                "message": "An optimizer execution may occur at most once per agent run.",
            },
        }
    try:
        record = state.scenario_store.get(str(arguments["scenario"]))
    except ScenarioNotFoundError:
        return {
            "success": False,
            "error": "scenario_not_found",
            "scenario": str(arguments["scenario"]),
        }

    adapter = _adapter_for(context, state)
    if record.problem_id != adapter.problem_id:
        return {
            "success": False,
            "error": "incompatible_problem",
            "active_problem_id": adapter.problem_id,
            "scenario_problem_id": record.problem_id,
            "scenario": record.definition.handle,
        }
    overrides = dict(record.definition.parameter_overrides)
    resolved_problem = adapter.resolve_problem(overrides)
    business_context = resolved_problem.business_parameters
    requested_fast = bool(arguments["fast"])
    fast = _effective_fast_mode(adapter, requested_fast)
    run_id = state.run_history_store.begin(
        run_kind="scenario",
        fast=fast,
        parameter_overrides=overrides,
        scenario_handle=record.definition.handle,
        problem_id=adapter.problem_id,
    )
    state.optimizer_calls += 1
    try:
        result = adapter.run_optimizer(fast=fast, parameter_overrides=overrides)
        outcome = adapter.interpret_optimizer_result(result)
    except Exception:
        state.run_history_store.fail(run_id, error_code="optimizer_exception")
        raise

    registered_handle: str | None = None
    historical_handle: str | None = None
    if outcome.validated_success and outcome.rows:
        try:
            candidate_handle = scenario_solution_handle(record.definition.handle)
            historical_handle = state.run_history_store.solution_handle_for(run_id)
            state.solution_store.register_run_result(
                historical_handle,
                candidate_handle,
                outcome.rows,
                business_context=business_context,
                problem_id=adapter.problem_id,
            )
            state.scenario_store.set_solution(record.definition.handle, candidate_handle)
            registered_handle = candidate_handle
        except (TypeError, SolutionStoreError):
            historical_handle = None
            registered_handle = None

    if historical_handle is not None and registered_handle is not None:
        state.run_history_store.complete(
            run_id,
            solution_handle=historical_handle,
            current_alias=registered_handle,
        )
    else:
        state.run_history_store.fail(
            run_id,
            error_code=outcome.failure_code or "solution_registration_failed",
        )

    compact = dict(outcome.compact_result)
    compact["requested_fast"] = requested_fast
    compact["fast_supported"] = adapter.supports_fast_mode
    compact["effective_fast"] = fast
    compact["fast_mode"] = {
        "requested": requested_fast,
        "effective": fast,
        "supported": adapter.supports_fast_mode,
    }
    if registered_handle is not None:
        compact["solution_handle"] = registered_handle
    compact["run_id"] = run_id
    compact["historical_solution_handle"] = historical_handle
    compact["scenario"] = {
        "handle": record.definition.handle,
        "name": record.definition.name,
        "parameter_overrides": dict(record.definition.parameter_overrides),
        "solution_handle": registered_handle or record.solution_handle,
    }
    return compact

def _compact_run_record(record: object) -> Dict[str, object]:
    return {
        "run_id": record.run_id,
        "run_kind": record.run_kind,
        "status": record.status,
        "fast": record.fast,
        "parameter_overrides": record.overrides,
        "scenario_handle": record.scenario_handle,
        "solution_handle": record.solution_handle,
        "current_alias": record.current_alias,
        "error_code": record.error_code,
    }


def _execute_list_runs(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    del context, arguments
    records = state.run_history_store.list()
    return {
        "success": True,
        "count": len(records),
        "runs": [_compact_run_record(record) for record in records],
    }


def _solution_not_found(handle: str) -> Dict[str, object]:
    return {
        "success": False,
        "error": "solution_not_found",
        "handle": handle,
    }


def _solution_context_missing(handle: str) -> Dict[str, object]:
    return {
        "success": False,
        "error": "solution_context_missing",
        "handle": handle,
    }


def _validated_business_context(
    adapter: OptimizationProblemAdapter,
    business_context: Mapping[str, object] | None,
) -> Dict[str, object] | None:
    """Validate stored business provenance through the active problem adapter."""
    if business_context is None:
        return None
    problem = adapter.problem
    business_names = {
        name
        for name, definition in problem.parameters.items()
        if definition.category == "business"
    }
    if set(business_context) != business_names:
        return None
    try:
        resolved = adapter.resolve_problem(dict(business_context))
    except OptimizationProblemError:
        return None
    return resolved.business_parameters


def _incompatible_problem_validation(
    handle: str, *, active_problem_id: str, solution_problem_id: str
) -> Dict[str, object]:
    return {
        "success": False,
        "error": "incompatible_problem",
        "handle": handle,
        "active_problem_id": active_problem_id,
        "solution_problem_id": solution_problem_id,
    }


def _execute_validate_solution(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    handle = str(arguments["handle"])
    # Planned execution may intentionally preserve an explicit user-supplied safe opaque
    # handle that is not currently registered. Resolve absence deterministically here instead
    # of coercing it to a known alias or passing it into SolutionStore's generated-handle API.
    if handle not in state.solution_store:
        return _solution_not_found(handle)
    try:
        rows = state.solution_store.get(handle)
        stored_context = state.solution_store.get_business_context(handle)
        solution_problem_id = state.solution_store.get_problem_id(handle)
    except SolutionNotFoundError:
        return _solution_not_found(handle)

    adapter = _adapter_for(context, state)
    if solution_problem_id != adapter.problem_id:
        return _incompatible_problem_validation(
            handle,
            active_problem_id=adapter.problem_id,
            solution_problem_id=solution_problem_id,
        )

    business_context = _validated_business_context(adapter, stored_context)
    if business_context is None:
        return _solution_context_missing(handle)

    report = adapter.validate_solution(rows, business_context=business_context)
    return adapter.compact_validation_result(handle, report)


def _incompatible_problem_comparison(
    solution_a_handle: str,
    solution_b_handle: str,
    *,
    active_problem_id: str,
    problem_id_a: str,
    problem_id_b: str,
) -> Dict[str, object]:
    return {
        "success": True,
        "solution_a_handle": solution_a_handle,
        "solution_b_handle": solution_b_handle,
        "comparable": False,
        "reason": "incompatible_problem",
        "active_problem_id": active_problem_id,
        "solution_a_problem_id": problem_id_a,
        "solution_b_problem_id": problem_id_b,
        "solution_a": None,
        "solution_b": None,
        "delta_b_minus_a": None,
    }


def _execute_compare_solutions(
    context: BonsaiAgentContext,
    arguments: Mapping[str, object],
    state: _AgentRunState,
) -> Dict[str, object]:
    solution_a_handle = str(arguments["solution_a"])
    solution_b_handle = str(arguments["solution_b"])
    try:
        solution_a = state.solution_store.get(solution_a_handle)
        stored_context_a = state.solution_store.get_business_context(solution_a_handle)
        problem_id_a = state.solution_store.get_problem_id(solution_a_handle)
    except SolutionNotFoundError:
        return _solution_not_found(solution_a_handle)
    try:
        solution_b = state.solution_store.get(solution_b_handle)
        stored_context_b = state.solution_store.get_business_context(solution_b_handle)
        problem_id_b = state.solution_store.get_problem_id(solution_b_handle)
    except SolutionNotFoundError:
        return _solution_not_found(solution_b_handle)

    adapter = _adapter_for(context, state)
    if (
        problem_id_a != problem_id_b
        or problem_id_a != adapter.problem_id
        or problem_id_b != adapter.problem_id
    ):
        return _incompatible_problem_comparison(
            solution_a_handle,
            solution_b_handle,
            active_problem_id=adapter.problem_id,
            problem_id_a=problem_id_a,
            problem_id_b=problem_id_b,
        )

    business_context_a = _validated_business_context(adapter, stored_context_a)
    if business_context_a is None:
        return _solution_context_missing(solution_a_handle)
    business_context_b = _validated_business_context(adapter, stored_context_b)
    if business_context_b is None:
        return _solution_context_missing(solution_b_handle)

    if not business_contexts_compatible(business_context_a, business_context_b):
        return {
            "success": True,
            "solution_a_handle": solution_a_handle,
            "solution_b_handle": solution_b_handle,
            "comparable": False,
            "reason": "incompatible_business_context",
            "solution_a": None,
            "solution_b": None,
            "delta_b_minus_a": None,
        }

    comparison = adapter.compare_solutions(
        solution_a,
        solution_b,
        business_context=business_context_a,
    )
    return adapter.compact_comparison_result(
        solution_a_handle,
        solution_b_handle,
        comparison,
    )

def _handle_schema(description: str) -> Dict[str, object]:
    return {
        "type": "string",
        "pattern": r"^(golden_reference|latest_optimizer|scenario_solution_[a-z0-9_]+|run_solution_[0-9]{6})$",
        "description": description + " Supports latest_optimizer, registered scenario_solution_* aliases, registered immutable run_solution_* history handles, and golden_reference when the active problem provides one.",
    }


TOOL_REGISTRY: Dict[str, AgentToolDefinition] = {
    "audit_data": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "audit_data",
            "description": "Inspect deterministic source-data quality for the active optimization problem.",
            "parameters": _empty_parameters(),
            "strict": True,
        },
        execute=_execute_audit_data,
    ),
    "get_baseline": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "get_baseline",
            "description": "Return the deterministic current baseline for the active optimization problem.",
            "parameters": _empty_parameters(),
            "strict": True,
        },
        execute=_execute_get_baseline,
    ),
    "run_optimization": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "run_optimization",
            "description": (
                "Run the active deterministic optimizer. Use only when the user explicitly asks "
                "to optimize, re-optimize, or obtain an optimized recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fast": {
                        "type": "boolean",
                        "description": "Whether to use the adapter's existing fast execution mode when applicable.",
                    },
                    "parameter_overrides": _parameter_override_tool_schema(include_description=True),
                },
                "required": ["fast", "parameter_overrides"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        execute=_execute_run_optimization,
    ),
    "validate_solution": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "validate_solution",
            "description": (
                "Independently validate a solution already registered in this application session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": _handle_schema(
                        "Registered solution handle to validate."
                    )
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        execute=_execute_validate_solution,
    ),
    "compare_solutions": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "compare_solutions",
            "description": (
                "Independently validate and compare two solutions already registered in this application session. "
                "Use the existing handles directly; do not run or refresh the optimizer to obtain them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "solution_a": _handle_schema("First registered solution handle."),
                    "solution_b": _handle_schema("Second registered solution handle."),
                },
                "required": ["solution_a", "solution_b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        execute=_execute_compare_solutions,
    ),
    "create_scenario": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "create_scenario",
            "description": (
                "Create a session-local named scenario from supported OptimizationProblem parameter overrides. "
                "Creation validates parameters but does not run optimization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "parameter_overrides": _parameter_override_tool_schema(include_description=False),
                },
                "required": ["name", "parameter_overrides"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        execute=_execute_create_scenario,
    ),
    "run_scenario": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "run_scenario",
            "description": (
                "Run a previously created named scenario with the existing deterministic optimizer. "
                "A solution handle is registered only after successful independent validation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "minLength": 1},
                    "fast": {"type": "boolean"},
                },
                "required": ["scenario", "fast"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        execute=_execute_run_scenario,
    ),
    "list_scenarios": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "list_scenarios",
            "description": (
                "List compact session-local named scenario definitions and their current solution handles. "
                "When used only to resolve a solution reference for a requested validation/comparison, "
                "continue to that requested operation in the same agent run after resolution. If the resolved "
                "handle refers to the same stored solution as another requested handle, still perform the "
                "requested comparison; equality is a valid deterministic result."
            ),
            "parameters": _empty_parameters(),
            "strict": True,
        },
        execute=_execute_list_scenarios,
    ),
    "list_runs": AgentToolDefinition(
        schema={
            "type": "function",
            "name": "list_runs",
            "description": (
                "List compact metadata for optimization attempts in the current application session. "
                "Run IDs and immutable run_solution_* handles are generated by the application. "
                "When used only to resolve a registered solution reference for a requested validation/comparison, "
                "continue to that requested operation in the same agent run after resolution. If the resolved "
                "historical handle refers to the same stored solution as a current alias, still perform the "
                "requested comparison; equality is a valid deterministic result."
            ),
            "parameters": _empty_parameters(),
            "strict": True,
        },
        execute=_execute_list_runs,
    ),
}


def _llm_tools_for_spec(spec: OptimizationProblemSpec) -> list[Dict[str, object]]:
    tools = deepcopy([definition.schema for definition in TOOL_REGISTRY.values()])
    is_bonsai = spec.problem_id == "bonsai_packaging"
    for schema in tools:
        name = schema.get("name")
        if is_bonsai:
            # Keep the accepted Step 14 Bonsai tool schemas byte-for-byte equivalent
            # in content while allowing other adapters to project their own metadata.
            if name == "audit_data":
                schema["description"] = (
                    "Inspect Bonsai source-data quality and the consistency of Procurement pricing "
                    "with Operations-derived volume tiers."
                )
            elif name == "get_baseline":
                schema["description"] = (
                    "Return the current baseline annual packaging, freight, and total cost."
                )
            elif name == "run_optimization":
                schema["description"] = (
                    "Run the existing deterministic Bonsai optimizer. Use only when the user explicitly "
                    "asks to optimize, re-optimize, or obtain an optimized recommendation. Set fast=true "
                    "only for a quicker pipeline check that skips post-MILP local search."
                )
                schema["parameters"]["properties"]["fast"]["description"] = (
                    "Whether to use the existing fast execution mode, which skips post-MILP local search."
                )
            elif name == "validate_solution":
                schema["parameters"]["properties"]["handle"]["description"] = (
                    "Registered solution handle to validate. Supports golden_reference, latest_optimizer, "
                    "registered scenario_solution_* aliases, and registered immutable run_solution_* history handles."
                )
            elif name == "compare_solutions":
                schema["parameters"]["properties"]["solution_a"]["description"] = (
                    "First registered solution handle. Supports golden_reference, latest_optimizer, registered "
                    "scenario_solution_* aliases, and registered immutable run_solution_* history handles."
                )
                schema["parameters"]["properties"]["solution_b"]["description"] = (
                    "Second registered solution handle. Supports golden_reference, latest_optimizer, registered "
                    "scenario_solution_* aliases, and registered immutable run_solution_* history handles."
                )
        else:
            if name == "audit_data":
                schema["description"] = f"Inspect deterministic source-data quality for {spec.display_name}."
            elif name == "get_baseline":
                schema["description"] = f"Return the deterministic current baseline for {spec.display_name}."
            elif name == "run_optimization":
                schema["description"] = (
                    f"Run the deterministic optimizer for {spec.display_name}. Use only declared "
                    "parameters from the active problem specification."
                )

        if name not in {"run_optimization", "create_scenario"}:
            continue
        parameters = schema.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            continue
        properties["parameter_overrides"] = _parameter_override_tool_schema(
            spec, include_description=name == "run_optimization"
        )
        if is_bonsai and name == "run_optimization":
            properties["parameter_overrides"]["description"] = (
                "Explicit overrides declared by the Bonsai optimization problem. Use null for parameters "
                "that are not being overridden. Business/problem parameters and solver parameters are "
                "intentionally distinguished by the problem definition."
            )
    return tools


LLM_TOOLS = _llm_tools_for_spec(bonsai_problem_spec())


def _execute_list_problems(state: _MultiDomainAgentRunState) -> Dict[str, object]:
    active = state.session.active_problem_id
    return {
        "success": True,
        "active_problem_id": active,
        "problems": [
            {**metadata.to_dict(), "active": metadata.problem_id == active}
            for metadata in state.session.registry.metadata()
        ],
    }


def _execute_select_problem(
    arguments: Mapping[str, object], state: _MultiDomainAgentRunState
) -> Dict[str, object]:
    requested = str(arguments.get("problem_id", "")).strip()
    previous = state.session.active_problem_id
    try:
        selected = state.session.select(requested)
    except ProblemNotRegisteredError:
        return {
            "success": False,
            "error": "problem_not_registered",
            "requested_problem_id": requested,
            "active_problem_id": previous,
            "registered_problem_ids": list(state.session.registry.problem_ids),
        }
    return {
        "success": True,
        "previous_problem_id": previous,
        "active_problem_id": selected.adapter.problem_id,
        "problem": {
            "problem_id": selected.adapter.problem_id,
            "display_name": selected.adapter.spec.display_name,
            "description": selected.adapter.spec.description,
        },
    }


def _llm_tools_for_registry(
    spec: OptimizationProblemSpec, registry: ProblemRegistry
) -> list[Dict[str, object]]:
    """Return the active domain's nine tools plus two fixed registry controls."""
    tools = _llm_tools_for_spec(spec)
    tools.extend([
        {
            "type": "function",
            "name": "list_problems",
            "description": (
                "List the explicitly registered optimization problems available in this session and identify "
                "the current active problem. This returns compact registry metadata only."
            ),
            "parameters": _empty_parameters(),
            "strict": True,
        },
        {
            "type": "function",
            "name": "select_problem",
            "description": (
                "Select one explicitly registered optimization problem as active. Use only for an explicit or "
                "unambiguous request for a registered domain. After successful selection, continue any requested "
                "domain operation in the same agent run using the refreshed active-domain tool schemas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_id": {
                        "type": "string",
                        "enum": list(registry.problem_ids),
                        "description": "Registered problem_id to make active.",
                    }
                },
                "required": ["problem_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ])
    return tools




_PLANNER_INSTRUCTIONS = """You are the internal execution planner for a controlled optimization application.
Return exactly one submit_plan function call. Planning does not execute business tools.
Create the smallest ordered plan that can satisfy the user's stated goal using only the allowlisted
application tools and registered optimization problems supplied by the application. Do not invent tools,
problems, solution handles, run IDs, parameter values, or future results.

ARGUMENT BINDING PROTOCOL
Each plan step contains tool-specific required bindings plus optional explicit bindings.
- literal binding: use only for a scalar value explicitly stated by the user. Preserve it exactly.
- deferred binding: use only when a required reference must be obtained from a declared PRIOR read/action step.
  Every deferred binding must declare source_step_id AND source_tool_name. source_tool_name must equal the
  application tool used by that prior step, and the selector must come from that source tool's compact-output
  contract. Never invent raw JSON paths or expressions.
- session binding: use only for an unambiguous recent entity explicitly supplied by APPLICATION PLANNING CONTEXT.
  The planner selects the application-generated session_entity_ref and preserves its declared entity_type and
  problem_id. Never copy or invent the underlying handle as a literal. Session references are typed state from
  an earlier completed turn, not defaults and not user literals.
- There is NO planner-authored default binding. If the user did not specify an optional argument, omit it.
  Canonical defaults are application behavior at execution time.

Required semantic references must be bound for the selected tool. In particular:
- select_problem requires problem_id.
- validate_solution requires handle.
- create_scenario requires name.
- run_scenario requires scenario.
- compare_solutions requires solution_a and solution_b.
Do not leave these required bindings absent.

Use typed deferred selectors only:
- direct_field: extract one declared scalar field from a prior compact result.
- collection_match: match exactly one declared collection record by an equality field/value and extract one field.
- unique_collection_item: extract one field only if the declared collection contains exactly one record.
The application validates that the selected collection/fields are part of the source tool's compact output contract.
Do not use JSONPath, JMESPath, source_output_path strings, executable expressions, or guessed handles.

RECENT SESSION ENTITY REFERENCES
APPLICATION PLANNING CONTEXT lists compact recent_session_entities for the CURRENT active problem only. Preserve
the declared entity type of anaphoric references across turns. If exactly one compatible recent scenario is marked
unambiguous_for_type=true and the user says "run it", "that scenario", or "the scenario", plan run_scenario with
a session binding to that scenario. Do NOT replace a scenario reference with run_optimization. If the same request
also asks to validate the run result, use run_scenario -> validate_solution and bind validate_solution.handle as a
deferred direct_field=solution_handle from the run_scenario step. Never fabricate scenario_* or
scenario_solution_* handles.
If no unique compatible recent entity exists, use an existing read-only resolver such as list_scenarios/list_runs
when it can deterministically resolve the reference, or fail closed. A unique_collection_item selector is allowed
only when the collection actually contains one record; ambiguity must not be guessed. Session entities from a
different active problem are inactive for the current plan. A scenario entity is not a solution entity, and a
solution entity is not a scenario entity.
For a historical alias snapshot, the source step MUST be list_runs. Use a collection_match selector from that
list_runs step that matches the requested current_alias and extracts solution_handle. validate_solution does not
resolve historical aliases and must never be used as the deferred source for this workflow. Do not add a
validate_solution step unless the user separately asked to validate. Do not use the newest run blindly and do not
fabricate run_solution_* IDs. Example shape: list_runs -> compare_solutions, with solution_a as the requested alias
and solution_b deferred from source_tool_name=list_runs, collection=runs, match_field=current_alias,
match_value=<requested alias>, extract_field=solution_handle.
EXPLICIT HISTORICAL ANCHOR RULE: when the user explicitly names a registered solution alias/handle X and asks to
compare X with "its historical snapshot", "its immutable snapshot", or "the historical version of X", X MUST
remain a literal binding. Only the historical side is deferred. Never turn both comparison operands into deferred
lookups. APPLICATION PLANNING CONTEXT may provide historical_reference_anchor when this exact anchored relation is
detected deterministically; when present, preserve that value as compare_solutions.solution_a literal and use it as
the list_runs selector match_value for compare_solutions.solution_b.

Do not add optional optimization parameter bindings the user did not request. In particular, do not invent solver
overrides to make optimization faster or easier. If fast/quick/reduced execution was not explicitly requested,
omit the fast binding; the application will deterministically use fast=false.

Exact tool arguments are still produced at execution time. Use target_problem_id only to make domain transitions
explicit. If the user request needs no application tool, create one descriptive no-tool step with no bindings.
Do not add retries, alternatives, approval gates, recovery, or replanning steps.
"""

_EXECUTOR_PROTOCOL_GROUNDING = """

STEP 17 CONTROLLED EXECUTION
You are executing exactly one already-validated plan step. You are not planning or replanning.
Call exactly the single application tool exposed for this step, once, and no other tool. Produce exact
arguments now using the original user goal, current active-problem grounding, the plan-step objective,
its argument_bindings, and compact deterministic outputs from completed earlier steps.
Literal bindings are immutable. Deferred bindings may use only values resolved by their declared typed selector
from their declared completed source step. Session bindings resolve only through deterministic recent session-entity
state for the declared entity type and active problem; never substitute a different entity or fabricate its handle.
Application defaults are not planner bindings: if fast is unbound,
use fast=false; if an optional optimization parameter is unbound, leave it null/unspecified so the deterministic
OptimizationProblem applies its canonical default. Do not invent optional parameter overrides.
If the step is select_problem, select exactly the step's target_problem_id. Execution-mode options must preserve
explicit user intent, including the existing fast-mode rules.
"""

_FINAL_SYNTHESIS_GROUNDING = """

STEP 17 FINAL SYNTHESIS
No application tools are available in this call. Summarize only the deterministic outcomes and protocol
errors contained in the completed execution record below. Do not invent missing values, future actions,
causes, solution rows, or unexecuted results. Preserve problem/business provenance, incompatible-context
rules, fast-mode intent semantics, and descriptive/non-causal narration. If execution failed, state the
failure and do not imply skipped steps ran.
"""


def _planner_selector_schema_for_source_tool(source_tool: str) -> Dict[str, object]:
    """Build typed deferred selectors from one source tool's compact-output contract.

    The private planner schema never exposes a selector shape that the declared source tool
    cannot satisfy. PlanValidator still verifies that source_step_id actually points to a prior
    step using that declared source tool.
    """
    contract = DEFERRED_OUTPUT_CONTRACTS.get(source_tool)
    if not isinstance(contract, Mapping):
        raise ValueError(f"Unknown deferred source tool: {source_tool!r}")
    scalar = {"type": ["string", "number", "boolean", "null"]}
    variants: list[Dict[str, object]] = []

    direct_fields = [str(field) for field in contract.get("direct_fields", ())]
    if direct_fields:
        variants.append({
            "type": "object",
            "properties": {
                "selector_type": {"type": "string", "enum": ["direct_field"]},
                "extract_field": {"type": "string", "enum": direct_fields},
            },
            "required": ["selector_type", "extract_field"],
            "additionalProperties": False,
        })

    collections = contract.get("collections")
    if isinstance(collections, Mapping):
        for collection_name, collection_contract in collections.items():
            if not isinstance(collection_contract, Mapping):
                continue
            match_fields = [str(field) for field in collection_contract.get("match_fields", ())]
            extract_fields = [str(field) for field in collection_contract.get("extract_fields", ())]
            if match_fields and extract_fields:
                variants.append({
                    "type": "object",
                    "properties": {
                        "selector_type": {"type": "string", "enum": ["collection_match"]},
                        "collection": {"type": "string", "enum": [str(collection_name)]},
                        "match_field": {"type": "string", "enum": match_fields},
                        "match_value": scalar,
                        "extract_field": {"type": "string", "enum": extract_fields},
                    },
                    "required": ["selector_type", "collection", "match_field", "match_value", "extract_field"],
                    "additionalProperties": False,
                })
            if extract_fields:
                variants.append({
                    "type": "object",
                    "properties": {
                        "selector_type": {"type": "string", "enum": ["unique_collection_item"]},
                        "collection": {"type": "string", "enum": [str(collection_name)]},
                        "extract_field": {"type": "string", "enum": extract_fields},
                    },
                    "required": ["selector_type", "collection", "extract_field"],
                    "additionalProperties": False,
                })

    if not variants:
        raise ValueError(f"Deferred source tool {source_tool!r} exposes no usable selector shapes.")
    return variants[0] if len(variants) == 1 else {"oneOf": variants}


def _planner_session_binding_schema(
    *,
    session_entities: tuple[Mapping[str, object], ...],
    expected_entity_type: str,
) -> Dict[str, object] | None:
    compatible = [
        entity for entity in session_entities
        if entity.get("entity_type") == expected_entity_type
        and entity.get("unambiguous_for_type") is True
    ]
    if len(compatible) != 1:
        return None
    entity = compatible[0]
    reference_id = entity.get("reference_id")
    problem_id = entity.get("problem_id")
    if not isinstance(reference_id, str) or not isinstance(problem_id, str):
        return None
    return {
        "type": "object",
        "properties": {
            "binding_type": {"type": "string", "enum": ["session"]},
            "literal_value": {"type": "null"},
            "source_step_id": {"type": "null"},
            "source_tool_name": {"type": "null"},
            "selector": {"type": "null"},
            "session_entity_type": {"type": "string", "enum": [expected_entity_type]},
            "session_entity_ref": {"type": "string", "enum": [reference_id]},
            "session_problem_id": {"type": "string", "enum": [problem_id]},
        },
        "required": [
            "binding_type", "literal_value", "source_step_id", "source_tool_name", "selector",
            "session_entity_type", "session_entity_ref", "session_problem_id",
        ],
        "additionalProperties": False,
    }


def _planner_binding_value_schema(
    *,
    literal_only: bool = False,
    literal_enum: list[object] | None = None,
    session_entities: tuple[Mapping[str, object], ...] = (),
    session_entity_type: str | None = None,
) -> Dict[str, object]:
    literal_value_schema: Dict[str, object] = {"type": ["string", "number", "boolean", "null"]}
    if literal_enum is not None:
        literal_value_schema["enum"] = list(literal_enum)
    literal = {
        "type": "object",
        "properties": {
            "binding_type": {"type": "string", "enum": ["literal"]},
            "literal_value": literal_value_schema,
            "source_step_id": {"type": "null"},
            "selector": {"type": "null"},
        },
        "required": ["binding_type", "literal_value", "source_step_id", "selector"],
        "additionalProperties": False,
    }
    if literal_only:
        return literal

    session_variant = (
        _planner_session_binding_schema(
            session_entities=session_entities, expected_entity_type=session_entity_type
        )
        if session_entity_type is not None
        else None
    )
    deferred_variants: list[Dict[str, object]] = []
    for source_tool in sorted(DEFERRED_OUTPUT_CONTRACTS):
        deferred_variants.append({
            "type": "object",
            "properties": {
                "binding_type": {"type": "string", "enum": ["deferred"]},
                "literal_value": {"type": "null"},
                "source_step_id": {"type": "string", "minLength": 1},
                "source_tool_name": {"type": "string", "enum": [source_tool]},
                "selector": _planner_selector_schema_for_source_tool(source_tool),
            },
            "required": ["binding_type", "literal_value", "source_step_id", "source_tool_name", "selector"],
            "additionalProperties": False,
        })
    variants = [literal, *deferred_variants]
    if session_variant is not None:
        variants.append(session_variant)
    return {"oneOf": variants}


def _planner_historical_snapshot_binding_schema(anchor: str) -> Dict[str, object]:
    """Private planner schema for the deferred side of one anchored historical comparison."""
    return {
        "type": "object",
        "description": (
            "Resolve only the immutable historical solution corresponding to the explicit anchor. "
            "Do not choose the newest run or a scenario run unless its current_alias equals the anchor."
        ),
        "properties": {
            "binding_type": {"type": "string", "enum": ["deferred"]},
            "literal_value": {"type": "null"},
            "source_step_id": {"type": "string", "minLength": 1},
            "source_tool_name": {"type": "string", "enum": ["list_runs"]},
            "selector": {
                "type": "object",
                "properties": {
                    "selector_type": {"type": "string", "enum": ["collection_match"]},
                    "collection": {"type": "string", "enum": ["runs"]},
                    "match_field": {"type": "string", "enum": ["current_alias"]},
                    "match_value": {"type": "string", "enum": [anchor]},
                    "extract_field": {"type": "string", "enum": ["solution_handle"]},
                },
                "required": [
                    "selector_type", "collection", "match_field", "match_value", "extract_field"
                ],
                "additionalProperties": False,
            },
        },
        "required": ["binding_type", "literal_value", "source_step_id", "source_tool_name", "selector"],
        "additionalProperties": False,
    }

def _planner_optional_binding_schema(registry: ProblemRegistry, tool_name: str) -> Dict[str, object]:
    parameter_paths = sorted({
        f"parameter_overrides.{parameter.name}"
        for problem_id in registry.problem_ids
        for parameter in registry.get(problem_id).spec.parameters
    })
    variants: list[Dict[str, object]] = []
    if tool_name in {"run_optimization", "run_scenario"}:
        variants.append({
            "type": "object",
            "properties": {
                "argument_path": {"type": "string", "enum": ["fast"]},
                "binding_type": {"type": "string", "enum": ["literal"]},
                "literal_value": {"type": "boolean"},
                "source_step_id": {"type": "null"},
                "selector": {"type": "null"},
            },
            "required": ["argument_path", "binding_type", "literal_value", "source_step_id", "selector"],
            "additionalProperties": False,
        })
    if tool_name in {"run_optimization", "create_scenario"}:
        variants.append({
            "type": "object",
            "properties": {
                "argument_path": {"type": "string", "enum": parameter_paths},
                "binding_type": {"type": "string", "enum": ["literal"]},
                "literal_value": {"type": ["string", "number", "boolean"]},
                "source_step_id": {"type": "null"},
                "selector": {"type": "null"},
            },
            "required": ["argument_path", "binding_type", "literal_value", "source_step_id", "selector"],
            "additionalProperties": False,
        })
    if not variants:
        return {"type": "array", "maxItems": 0, "items": {"type": "object"}}
    item_schema: Dict[str, object] = variants[0] if len(variants) == 1 else {"oneOf": variants}
    return {"type": "array", "maxItems": 32, "items": item_schema}


def _planner_required_bindings_schema(
    tool_name: str | None,
    problem_ids: list[str],
    session_entities: tuple[Mapping[str, object], ...],
    historical_reference_anchor: str | None = None,
) -> Dict[str, object]:
    properties: Dict[str, object] = {}
    required: list[str] = []
    if tool_name == "select_problem":
        properties["problem_id"] = _planner_binding_value_schema(
            literal_only=True, literal_enum=problem_ids
        )
        required = ["problem_id"]
    elif tool_name == "validate_solution":
        properties["handle"] = _planner_binding_value_schema(
            session_entities=session_entities, session_entity_type="solution"
        )
        required = ["handle"]
    elif tool_name == "create_scenario":
        properties["name"] = _planner_binding_value_schema(literal_only=True)
        required = ["name"]
    elif tool_name == "run_scenario":
        properties["scenario"] = _planner_binding_value_schema(
            session_entities=session_entities, session_entity_type="scenario"
        )
        required = ["scenario"]
    elif tool_name == "compare_solutions":
        if historical_reference_anchor is not None:
            properties["solution_a"] = _planner_binding_value_schema(
                literal_only=True, literal_enum=[historical_reference_anchor]
            )
            properties["solution_a"]["description"] = (
                "Explicit user-named historical anchor. Preserve it literally; do not resolve it through list_runs."
            )
            properties["solution_b"] = _planner_historical_snapshot_binding_schema(
                historical_reference_anchor
            )
        else:
            properties["solution_a"] = _planner_binding_value_schema(
                session_entities=session_entities, session_entity_type="solution"
            )
            properties["solution_b"] = _planner_binding_value_schema(
                session_entities=session_entities, session_entity_type="solution"
            )
        required = ["solution_a", "solution_b"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _planner_step_variant(
    registry: ProblemRegistry,
    tool_name: str | None,
    *,
    session_entities: tuple[Mapping[str, object], ...] = (),
    historical_reference_anchor: str | None = None,
) -> Dict[str, object]:
    problem_ids = list(registry.problem_ids)
    if tool_name is None:
        tool_schema: Dict[str, object] = {"type": "null"}
        target_schema: Dict[str, object] = {"type": ["string", "null"], "enum": [*problem_ids, None]}
    else:
        tool_schema = {"type": "string", "enum": [tool_name]}
        target_schema = (
            {"type": "string", "enum": problem_ids}
            if tool_name == "select_problem"
            else {"type": ["string", "null"], "enum": [*problem_ids, None]}
        )
    return {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1},
            "objective": {
                "type": "string",
                "minLength": 1,
                "description": "What this step must accomplish; do not include fabricated results.",
            },
            "expected_application_tool": tool_schema,
            "target_problem_id": target_schema,
            "argument_bindings": {
                "type": "object",
                "properties": {
                    "required": _planner_required_bindings_schema(
                        tool_name,
                        problem_ids,
                        session_entities,
                        historical_reference_anchor=historical_reference_anchor,
                    ),
                    "optional": _planner_optional_binding_schema(registry, tool_name or ""),
                },
                "required": ["required", "optional"],
                "additionalProperties": False,
            },
        },
        "required": [
            "step_id", "objective", "expected_application_tool", "target_problem_id", "argument_bindings"
        ],
        "additionalProperties": False,
    }


def _planner_submit_plan_schema(
    registry: ProblemRegistry,
    *,
    session_entities: tuple[Mapping[str, object], ...] = (),
    historical_reference_anchor: str | None = None,
) -> Dict[str, object]:
    application_tools = list(TOOL_REGISTRY) + ["list_problems", "select_problem"]
    variants = [
        _planner_step_variant(
            registry,
            tool,
            session_entities=session_entities,
            historical_reference_anchor=historical_reference_anchor,
        )
        for tool in application_tools
    ]
    variants.append(
        _planner_step_variant(
            registry,
            None,
            session_entities=session_entities,
            historical_reference_anchor=historical_reference_anchor,
        )
    )
    return {
        "type": "function",
        "name": "submit_plan",
        "description": (
            "Submit an ordered execution plan only. This internal protocol never executes an application action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PLAN_STEPS,
                    "items": {"oneOf": variants},
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _explicit_historical_reference_anchor(user_message: str) -> str | None:
    """Return one exact user-named supported solution handle for a historical relation.

    This is deliberately narrow deterministic grounding, not general natural-language planning.
    It only activates for explicit historical/immutable-snapshot language plus exactly one
    syntactically supported solution handle token already present in the user's text.
    """
    if not isinstance(user_message, str):
        return None
    lowered = user_message.lower()
    if not any(phrase in lowered for phrase in _HISTORICAL_REFERENCE_PHRASES):
        return None
    references: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", user_message):
        if is_supported_solution_handle(token) and token not in references:
            references.append(token)
    return references[0] if len(references) == 1 else None


def _parse_deferred_selector(raw_selector: object) -> DeferredSelector:
    if not isinstance(raw_selector, Mapping):
        raise ValueError("Deferred bindings require a selector object.")
    selector_type = raw_selector.get("selector_type")
    if selector_type == "direct_field":
        expected = {"selector_type", "extract_field"}
        if set(raw_selector) != expected:
            raise ValueError(f"direct_field selector requires exactly {sorted(expected)}.")
        return DeferredSelector(
            selector_type="direct_field",
            extract_field=str(raw_selector.get("extract_field", "")),
        )
    if selector_type == "collection_match":
        expected = {"selector_type", "collection", "match_field", "match_value", "extract_field"}
        if set(raw_selector) != expected:
            raise ValueError(f"collection_match selector requires exactly {sorted(expected)}.")
        return DeferredSelector(
            selector_type="collection_match",
            collection=str(raw_selector.get("collection", "")),
            match_field=str(raw_selector.get("match_field", "")),
            match_value=raw_selector.get("match_value"),
            extract_field=str(raw_selector.get("extract_field", "")),
        )
    if selector_type == "unique_collection_item":
        expected = {"selector_type", "collection", "extract_field"}
        if set(raw_selector) != expected:
            raise ValueError(f"unique_collection_item selector requires exactly {sorted(expected)}.")
        return DeferredSelector(
            selector_type="unique_collection_item",
            collection=str(raw_selector.get("collection", "")),
            extract_field=str(raw_selector.get("extract_field", "")),
        )
    raise ValueError(f"Unknown deferred selector type: {selector_type!r}")


def _parse_binding_payload(argument_path: str, payload: object) -> ArgumentBinding:
    if not isinstance(payload, Mapping):
        raise ValueError("Argument binding must be an object.")
    binding_type = str(payload.get("binding_type", ""))
    if binding_type == "deferred":
        expected = {"binding_type", "literal_value", "source_step_id", "source_tool_name", "selector"}
    elif binding_type == "session":
        expected = {
            "binding_type", "literal_value", "source_step_id", "source_tool_name", "selector",
            "session_entity_type", "session_entity_ref", "session_problem_id",
        }
    else:
        expected = {"binding_type", "literal_value", "source_step_id", "selector"}
    if set(payload) != expected:
        raise ValueError(f"Argument binding must contain exactly {sorted(expected)}.")
    selector = None
    if binding_type == "deferred":
        selector = _parse_deferred_selector(payload.get("selector"))
    elif payload.get("selector") is not None:
        raise ValueError("Non-deferred bindings require selector=null.")
    return ArgumentBinding(
        argument_path=argument_path,
        binding_type=binding_type,
        literal_value=payload.get("literal_value"),
        source_step_id=payload.get("source_step_id"),
        source_tool_name=payload.get("source_tool_name"),
        selector=selector,
        session_entity_type=payload.get("session_entity_type"),
        session_entity_ref=payload.get("session_entity_ref"),
        session_problem_id=payload.get("session_problem_id"),
    )


def _plan_from_submission(
    *,
    plan_id: str,
    user_goal: str,
    initial_problem_id: str,
    arguments: object,
) -> ExecutionPlan:
    if not isinstance(arguments, Mapping) or set(arguments) != {"steps"}:
        raise PlanValidationError(
            "malformed_plan_submission",
            "submit_plan requires exactly one steps array.",
        )
    raw_steps = arguments.get("steps")
    if not isinstance(raw_steps, list):
        raise PlanValidationError("malformed_plan_submission", "submit_plan.steps must be an array.")
    steps: list[PlanStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, Mapping):
            raise PlanValidationError("malformed_plan_step", f"Plan step {index} must be an object.")
        expected_keys = {
            "step_id", "objective", "expected_application_tool", "target_problem_id", "argument_bindings",
        }
        if set(item) != expected_keys:
            raise PlanValidationError(
                "malformed_plan_step",
                f"Plan step {index} must contain exactly {sorted(expected_keys)}.",
            )
        tool = item.get("expected_application_tool")
        target = item.get("target_problem_id")
        if tool is not None and not isinstance(tool, str):
            raise PlanValidationError("malformed_plan_step", f"Plan step {index} tool must be text or null.")
        if target is not None and not isinstance(target, str):
            raise PlanValidationError(
                "malformed_plan_step", f"Plan step {index} target_problem_id must be text or null."
            )

        raw_bindings = item.get("argument_bindings")
        if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != {"required", "optional"}:
            raise PlanValidationError(
                "malformed_plan_step",
                f"Plan step {index} argument_bindings must contain exactly required and optional.",
            )
        raw_required = raw_bindings.get("required")
        raw_optional = raw_bindings.get("optional")
        if not isinstance(raw_required, Mapping) or not isinstance(raw_optional, list):
            raise PlanValidationError(
                "malformed_plan_step",
                f"Plan step {index} bindings require an object plus an optional array.",
            )

        bindings: list[ArgumentBinding] = []
        try:
            for argument_path, payload in raw_required.items():
                bindings.append(_parse_binding_payload(str(argument_path), payload))
            for binding_index, raw_binding in enumerate(raw_optional, start=1):
                if not isinstance(raw_binding, Mapping) or "argument_path" not in raw_binding:
                    raise ValueError(f"Optional binding {binding_index} requires argument_path.")
                expected_optional = {
                    "argument_path", "binding_type", "literal_value", "source_step_id", "selector"
                }
                if set(raw_binding) != expected_optional:
                    raise ValueError(
                        f"Optional binding {binding_index} must contain exactly {sorted(expected_optional)}."
                    )
                payload = {key: raw_binding[key] for key in expected_optional if key != "argument_path"}
                bindings.append(_parse_binding_payload(str(raw_binding["argument_path"]), payload))
        except ValueError as exc:
            raise PlanValidationError("malformed_argument_binding", str(exc)) from exc

        steps.append(
            PlanStep(
                step_id=str(item.get("step_id", "")),
                objective=str(item.get("objective", "")),
                expected_application_tool=tool,
                target_problem_id=target,
                argument_bindings=tuple(bindings),
            )
        )
    return ExecutionPlan(
        plan_id=plan_id,
        user_goal=user_goal,
        initial_active_problem_id=initial_problem_id,
        steps=steps,
    )


def _plan_safe_value(value: object, *, depth: int = 0) -> object:
    """Defensively copy compact tool metadata while excluding row/artifact payload keys."""
    if depth > 8:
        return "[truncated]"
    if isinstance(value, Mapping):
        safe: Dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in {
                "rows", "solution_rows", "raw_rows", "dataset", "datasets", "artifact", "artifacts"
            }:
                continue
            safe[key_text] = _plan_safe_value(item, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_plan_safe_value(item, depth=depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _record_recent_session_entities(
    *,
    entity_context: SessionEntityContext,
    plan_id: str,
    step_id: str,
    tool_name: str,
    arguments: Mapping[str, object],
    tool_result: Mapping[str, object],
    state: _MultiDomainAgentRunState,
) -> None:
    """Record compact successful entity references without storing rows or datasets."""

    problem_id = state.problem_adapter.problem_id

    def record(entity_type: str, handle: object, display_name: object | None = None) -> None:
        if not isinstance(handle, str) or not handle.strip():
            return
        name = display_name if isinstance(display_name, str) and display_name.strip() else None
        entity_context.record(
            entity_type=entity_type,
            handle=handle,
            display_name=name,
            problem_id=problem_id,
            source_plan_id=plan_id,
            source_step_id=step_id,
        )

    if tool_name == "create_scenario":
        scenario = tool_result.get("scenario")
        if isinstance(scenario, Mapping):
            record("scenario", scenario.get("handle"), scenario.get("name"))
        return

    if tool_name == "run_scenario":
        scenario = tool_result.get("scenario")
        if isinstance(scenario, Mapping):
            record("scenario", scenario.get("handle"), scenario.get("name"))
        record("solution", tool_result.get("solution_handle"), tool_result.get("solution_handle"))
        record("run", tool_result.get("run_id"), tool_result.get("run_id"))
        return

    if tool_name == "run_optimization":
        record("solution", tool_result.get("solution_handle"), tool_result.get("solution_handle"))
        record("run", tool_result.get("run_id"), tool_result.get("run_id"))
        return

    if tool_name == "validate_solution":
        record("solution", arguments.get("handle"), arguments.get("handle"))
        return

    if tool_name == "compare_solutions":
        record("solution", arguments.get("solution_a"), arguments.get("solution_a"))
        record("solution", arguments.get("solution_b"), arguments.get("solution_b"))


def _tool_result_failed(result: Mapping[str, object]) -> bool:
    return result.get("success") is False or result.get("ok") is False


def _tool_result_error_code(result: Mapping[str, object]) -> str | None:
    error = result.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        return str(code) if code is not None else "tool_error"
    if error is not None:
        return str(error)
    return None

def _safe_float_parameter_value(value: object) -> object:
    """CORE-FIX-01: exact int->float conversion, never bool/string/nonfinite/rounded.

    All other types are left to the authoritative ParameterDefinition validator.
    """
    if type(value) is int:
        try:
            converted = float(value)
        except OverflowError:
            return value
        if math.isfinite(converted) and int(converted) == value:
            return converted
    return value


def _canonicalize_plan_numeric_bindings(plan: ExecutionPlan, registry: ProblemRegistry) -> None:
    """Canonicalize approved literal bindings BEFORE plan/step/state fingerprints.

    The validated select_problem sequence is simulated for schema lookup only. This
    never selects a session problem or changes an optimization parameter value.
    """
    problem_id = plan.initial_active_problem_id
    for step in plan.steps:
        if step.expected_application_tool == "select_problem":
            problem_id = str(step.target_problem_id)
            continue
        if step.expected_application_tool not in {"run_optimization", "create_scenario"}:
            continue
        definitions = registry.get(problem_id).problem.parameters
        bindings = []
        for binding in step.argument_bindings:
            prefix = "parameter_overrides."
            if binding.binding_type != "literal" or not binding.argument_path.startswith(prefix):
                bindings.append(binding)
                continue
            name = binding.argument_path[len(prefix):]
            definition = definitions.get(name)
            if definition is None or binding.literal_value is None:
                # Unknown overrides and explicit null retain their existing guards.
                bindings.append(binding)
                continue
            value = binding.literal_value
            if definition.value_type is float:
                value = _safe_float_parameter_value(value)
            try:
                definition.validate(value)
            except OptimizationProblemError as exc:
                raise PlanValidationError("invalid_numeric_parameter", str(exc)) from exc
            bindings.append(replace(binding, literal_value=value) if type(value) is not type(binding.literal_value) else binding)
        step.argument_bindings = tuple(bindings)


def _restore_approved_float_literals(
    *, step: PlanStep, arguments: object, problem_adapter: OptimizationProblemAdapter
) -> None:
    """Use the *already approved* canonical binding when JSON executor emits an int.

    This is not a fresh conversion of approved plan/state: the executor's integer
    spelling is permitted only if it exactly represents the approved float literal.
    Other executor arguments are unmodified and retain all existing intent guards.
    """
    if step.expected_application_tool not in {"run_optimization", "create_scenario"}:
        return
    if not isinstance(arguments, dict) or not isinstance(arguments.get("parameter_overrides"), dict):
        return
    overrides = arguments["parameter_overrides"]
    definitions = problem_adapter.problem.parameters
    for binding in step.argument_bindings:
        prefix = "parameter_overrides."
        if binding.binding_type != "literal" or not binding.argument_path.startswith(prefix):
            continue
        name = binding.argument_path[len(prefix):]
        definition = definitions.get(name)
        canonical = binding.literal_value
        if definition is None or definition.value_type is not float or type(canonical) is not float:
            continue
        actual = overrides.get(name)
        if type(actual) is int and _safe_float_parameter_value(actual) == canonical and type(_safe_float_parameter_value(actual)) is float:
            overrides[name] = canonical


def _validate_multi_domain_arguments(
    name: str, arguments: object, state: _MultiDomainAgentRunState
) -> str | None:
    if name == "list_problems":
        if not isinstance(arguments, dict) or arguments:
            return "list_problems accepts no arguments."
        return None
    if name == "select_problem":
        if not isinstance(arguments, dict) or set(arguments) != {"problem_id"}:
            return "select_problem requires exactly the 'problem_id' argument."
        problem_id = arguments.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            return "select_problem.problem_id must be non-empty text."
        # Unknown ids are handled by the executor so callers receive a deterministic
        # problem_not_registered result and active state is provably unchanged.
        return None
    return _validate_arguments(name, arguments, state.problem_adapter)


def _validate_arguments(
    name: str,
    arguments: object,
    problem_adapter: OptimizationProblemAdapter,
) -> str | None:
    """Validate the tiny V0 schemas without adding a JSON-schema dependency."""
    if not isinstance(arguments, dict):
        return "Tool arguments must decode to a JSON object."
    if name in {"audit_data", "get_baseline", "list_scenarios", "list_runs"}:
        if arguments:
            return "This tool accepts no arguments."
        return None
    if name == "run_optimization":
        allowed = {"fast", "parameter_overrides"}
        if not set(arguments).issubset(allowed) or "fast" not in arguments:
            return "run_optimization requires 'fast' and optionally 'parameter_overrides'."
        if type(arguments["fast"]) is not bool:
            return "run_optimization.fast must be a boolean."
        if "parameter_overrides" in arguments:
            overrides = arguments["parameter_overrides"]
            if not isinstance(overrides, dict):
                return "run_optimization.parameter_overrides must be an object."
            problem = problem_adapter.problem
            unknown = set(overrides) - set(problem.parameters)
            if unknown:
                return f"Unknown optimization parameter(s): {sorted(unknown)}"
            for name_, value in overrides.items():
                if value is None:
                    continue
                try:
                    problem.parameters[name_].validate(value)
                except OptimizationProblemError as exc:
                    return str(exc)
        return None
    if name == "create_scenario":
        if set(arguments) != {"name", "parameter_overrides"}:
            return "create_scenario requires exactly 'name' and 'parameter_overrides'."
        if not isinstance(arguments["name"], str) or not arguments["name"].strip():
            return "create_scenario.name must be non-empty text."
        overrides = arguments["parameter_overrides"]
        if not isinstance(overrides, dict):
            return "create_scenario.parameter_overrides must be an object."
        problem = problem_adapter.problem
        unknown = set(overrides) - set(problem.parameters)
        if unknown:
            return f"Unknown optimization parameter(s): {sorted(unknown)}"
        for name_, value in overrides.items():
            if value is None:
                continue
            try:
                problem.parameters[name_].validate(value)
            except OptimizationProblemError as exc:
                return str(exc)
        return None
    if name == "run_scenario":
        if set(arguments) != {"scenario", "fast"}:
            return "run_scenario requires exactly 'scenario' and 'fast'."
        if not isinstance(arguments["scenario"], str) or not arguments["scenario"].strip():
            return "run_scenario.scenario must be non-empty text."
        if type(arguments["fast"]) is not bool:
            return "run_scenario.fast must be a boolean."
        return None
    if name == "validate_solution":
        if set(arguments) != {"handle"}:
            return "validate_solution requires exactly the 'handle' argument."
        handle = arguments["handle"]
        if not isinstance(handle, str) or _SAFE_OPAQUE_SOLUTION_REFERENCE.fullmatch(handle) is None:
            return "validate_solution.handle must be a safe opaque solution reference."
        return None
    if name == "compare_solutions":
        if set(arguments) != {"solution_a", "solution_b"}:
            return "compare_solutions requires exactly 'solution_a' and 'solution_b'."
        if not is_supported_solution_handle(arguments["solution_a"]):
            return "compare_solutions.solution_a must be a supported solution handle."
        if not is_supported_solution_handle(arguments["solution_b"]):
            return "compare_solutions.solution_b must be a supported solution handle."
        return None
    return "Unknown tool."


def _schema_type_matches(value: object, expected_type: object) -> bool:
    types = expected_type if isinstance(expected_type, list) else [expected_type]
    for type_name in types:
        if type_name == "null" and value is None:
            return True
        if type_name == "object" and isinstance(value, dict):
            return True
        if type_name == "array" and isinstance(value, list):
            return True
        if type_name == "string" and isinstance(value, str):
            return True
        if type_name == "boolean" and type(value) is bool:
            return True
        if type_name == "integer" and isinstance(value, int) and type(value) is not bool:
            return True
        if type_name == "number" and isinstance(value, (int, float)) and type(value) is not bool:
            return True
    return False


def _validate_json_schema_value(value: object, schema: Mapping[str, object], *, path: str = "$") -> str | None:
    """Validate the small JSON-schema subset emitted by current application tools.

    This does not define domain rules. It consumes the already-generated active tool schema so
    executor protocol validation cannot drift from the schema exposed to the provider.
    """

    expected_type = schema.get("type")
    if expected_type is not None and not _schema_type_matches(value, expected_type):
        return f"{path} does not match schema type {expected_type!r}."

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} is not one of the allowed values."

    if value is None:
        return None

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        required_names = required if isinstance(required, list) else []
        for name in required_names:
            if name not in value:
                return f"{path}.{name} is required."
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                return f"{path} contains unsupported properties: {sorted(unknown)}"
        for name, item in value.items():
            property_schema = properties.get(name)
            if isinstance(property_schema, Mapping):
                error = _validate_json_schema_value(item, property_schema, path=f"{path}.{name}")
                if error is not None:
                    return error
        return None

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            return f"{path} contains fewer than {minimum_items} items."
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            return f"{path} contains more than {maximum_items} items."
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                error = _validate_json_schema_value(item, item_schema, path=f"{path}[{index}]")
                if error is not None:
                    return error
        return None

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            return f"{path} must contain at least {min_length} character(s)."
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            return f"{path} does not match the required pattern."
        return None

    if isinstance(value, (int, float)) and type(value) is not bool:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return f"{path} is below the minimum {minimum}."
        if isinstance(maximum, (int, float)) and value > maximum:
            return f"{path} is above the maximum {maximum}."

    return None


def _validate_executor_arguments_against_current_tool(
    *,
    tool_name: str,
    arguments: object,
    tool_schema: Mapping[str, object],
    state: _MultiDomainAgentRunState,
) -> str | None:
    parameters = tool_schema.get("parameters")
    if not isinstance(parameters, Mapping):
        return "Current tool schema has no parameters object."
    schema_error = _validate_json_schema_value(arguments, parameters)
    if schema_error is not None:
        return schema_error
    # Retain the existing executable argument guards as a second deterministic boundary.
    # They resolve against the CURRENT active adapter and therefore preserve parameter provenance.
    return _validate_multi_domain_arguments(tool_name, arguments, state)


@dataclass(frozen=True)
class _ArgumentIntentResult:
    valid: bool
    message: str | None = None
    literal_binding_count: int = 0
    literal_bindings_preserved: bool | None = None
    deferred_binding_count: int = 0
    deferred_bindings_resolved: bool | None = None
    unexpected_argument_paths: tuple[str, ...] = ()


def _argument_path_value(arguments: object, path: str) -> tuple[bool, object]:
    current = arguments
    for token in path.split("."):
        if not isinstance(current, Mapping) or token not in current:
            return False, None
        current = current[token]
    return True, current


def _scalar_values_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


_DIRECT_DEFERRED_FIELD_PATHS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("create_scenario", "scenario_handle"): ("scenario", "handle"),
    ("run_scenario", "scenario_handle"): ("scenario", "handle"),
    ("run_scenario", "solution_handle"): ("solution_handle",),
    ("run_scenario", "historical_solution_handle"): ("historical_solution_handle",),
    ("run_scenario", "run_id"): ("run_id",),
    ("run_optimization", "solution_handle"): ("solution_handle",),
    ("run_optimization", "historical_solution_handle"): ("historical_solution_handle",),
    ("run_optimization", "run_id"): ("run_id",),
    ("select_problem", "active_problem_id"): ("active_problem_id",),
}


def _completed_output_for_step(
    completed_outputs: list[Dict[str, object]], source_step_id: str
) -> tuple[str, Mapping[str, object]] | None:
    for output in completed_outputs:
        if output.get("step_id") != source_step_id:
            continue
        tool = output.get("tool")
        result = output.get("result")
        if isinstance(tool, str) and isinstance(result, Mapping):
            return tool, result
        return None
    return None


def _nested_scalar(mapping: Mapping[str, object], path: tuple[str, ...]) -> tuple[bool, object]:
    current: object = mapping
    for token in path:
        if not isinstance(current, Mapping) or token not in current:
            return False, None
        current = current[token]
    if isinstance(current, (str, int, float, bool)) or current is None:
        return True, current
    return False, None


def _resolve_deferred_selector(
    *,
    source_tool: str,
    result: Mapping[str, object],
    selector: DeferredSelector,
) -> tuple[tuple[object, ...], str | None]:
    contract = DEFERRED_OUTPUT_CONTRACTS.get(source_tool)
    if contract is None:
        return (), f"Source tool {source_tool!r} exposes no deferred-reference contract."

    if selector.selector_type == "direct_field":
        path = _DIRECT_DEFERRED_FIELD_PATHS.get((source_tool, selector.extract_field))
        if path is None:
            return (), (
                f"Source tool {source_tool!r} does not expose direct field "
                f"{selector.extract_field!r}."
            )
        found, value = _nested_scalar(result, path)
        if not found or value is None:
            return (), (
                f"Deferred direct field {selector.extract_field!r} is unavailable in "
                f"completed step output from {source_tool!r}."
            )
        return (value,), None

    collections = contract.get("collections")
    collection_contract = collections.get(selector.collection) if isinstance(collections, Mapping) else None
    if not isinstance(collection_contract, Mapping):
        return (), (
            f"Source tool {source_tool!r} does not expose collection {selector.collection!r}."
        )
    raw_collection = result.get(str(selector.collection))
    if not isinstance(raw_collection, list):
        return (), (
            f"Completed output from {source_tool!r} has no collection {selector.collection!r}."
        )
    records = [item for item in raw_collection if isinstance(item, Mapping)]

    if selector.selector_type == "unique_collection_item":
        if len(records) != 1:
            return (), (
                f"Deferred selector expected exactly one {selector.collection!r} record but found {len(records)}."
            )
        selected = records
    else:
        selected = [
            item for item in records
            if selector.match_field in item
            and _scalar_values_equal(item.get(str(selector.match_field)), selector.match_value)
        ]
        if len(selected) != 1:
            return (), (
                f"Deferred selector expected exactly one {selector.collection!r} record matching "
                f"{selector.match_field!r} but found {len(selected)}."
            )

    value = selected[0].get(selector.extract_field)
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        return (), f"Deferred selector extracted a non-scalar {selector.extract_field!r} value."
    if value is None:
        return (), f"Deferred selector extracted no usable {selector.extract_field!r} value."
    return (value,), None


def _binding_allowed_values(
    binding: ArgumentBinding,
    *,
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext | None = None,
    active_problem_id: str | None = None,
) -> tuple[tuple[object, ...], str | None]:
    if binding.binding_type == "literal":
        return (binding.literal_value,), None
    if binding.binding_type == "session":
        if session_entity_context is None or active_problem_id is None:
            return (), "Session binding cannot resolve without session entity context and active problem."
        try:
            record = session_entity_context.resolve(
                str(binding.session_entity_ref),
                expected_entity_type=str(binding.session_entity_type),
                expected_problem_id=active_problem_id,
            )
        except PlanValidationError as exc:
            return (), f"{exc.code}: {exc.message}"
        if binding.session_problem_id != active_problem_id:
            return (), (
                f"Session binding expected problem {binding.session_problem_id!r} but active problem is "
                f"{active_problem_id!r}."
            )
        return (record.handle,), None
    completed = _completed_output_for_step(completed_outputs, str(binding.source_step_id))
    if completed is None:
        return (), f"Deferred source step {binding.source_step_id!r} has no completed output."
    source_tool, result = completed
    if binding.source_tool_name != source_tool:
        return (), (
            f"Deferred binding declared source tool {binding.source_tool_name!r} but completed "
            f"step {binding.source_step_id!r} used {source_tool!r}."
        )
    if binding.selector is None:
        return (), "Deferred binding is missing its typed selector."
    return _resolve_deferred_selector(
        source_tool=source_tool,
        result=result,
        selector=binding.selector,
    )


def _plan_contract_payload(plan: ExecutionPlan) -> Dict[str, object]:
    """Return stable plan intent metadata excluding mutable execution statuses."""
    return {
        "plan_id": plan.plan_id,
        "user_goal": plan.user_goal,
        "initial_active_problem_id": plan.initial_active_problem_id,
        "steps": [
            {
                "step_id": step.step_id,
                "objective": step.objective,
                "expected_application_tool": step.expected_application_tool,
                "target_problem_id": step.target_problem_id,
                "argument_bindings": [binding.to_dict() for binding in step.argument_bindings],
            }
            for step in plan.steps
        ],
    }


def _step_contract_payload(step: PlanStep) -> Dict[str, object]:
    return {
        "step_id": step.step_id,
        "objective": step.objective,
        "expected_application_tool": step.expected_application_tool,
        "target_problem_id": step.target_problem_id,
        "argument_bindings": [binding.to_dict() for binding in step.argument_bindings],
    }


def _resolved_binding_values(
    *,
    step: PlanStep,
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext,
    active_problem_id: str,
) -> tuple[Dict[str, object], str | None]:
    resolved: Dict[str, object] = {}
    for binding in step.argument_bindings:
        values, error = _binding_allowed_values(
            binding,
            completed_outputs=completed_outputs,
            session_entity_context=session_entity_context,
            active_problem_id=active_problem_id,
        )
        if error is not None:
            return {}, error
        if len(values) != 1:
            return {}, f"Binding {binding.argument_path!r} did not resolve to exactly one value."
        resolved[binding.argument_path] = values[0]
    return resolved, None


def _approval_context_for_step(
    *,
    step: PlanStep,
    state: _AgentRunState | _MultiDomainAgentRunState,
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext,
) -> tuple[object, Dict[str, object], str | None]:
    """Return ApprovalDecision plus deterministic state metadata for one planned step."""
    tool_name = step.expected_application_tool
    if tool_name is None:
        return ApprovalPolicy().assess(
            tool_name="", problem=state.problem_adapter.problem, problem_id=state.problem_adapter.problem_id,
            fast=False, parameter_overrides={},
        ), {}, None
    active_problem_id = state.problem_adapter.problem_id
    resolved, error = _resolved_binding_values(
        step=step,
        completed_outputs=completed_outputs,
        session_entity_context=session_entity_context,
        active_problem_id=active_problem_id,
    )
    if error is not None:
        return None, {}, error

    fast = bool(resolved.get("fast", False))
    parameter_overrides: Dict[str, object] = {}
    scenario_handle: str | None = None
    scenario_name: str | None = None
    scenario_solution_handle: str | None = None

    if tool_name == "run_optimization":
        for path, value in resolved.items():
            if path.startswith("parameter_overrides."):
                parameter_overrides[path.split(".", 1)[1]] = value
    elif tool_name == "run_scenario":
        raw_scenario = resolved.get("scenario")
        if not isinstance(raw_scenario, str) or not raw_scenario:
            return None, {}, "run_scenario approval assessment requires a resolved scenario reference."
        try:
            scenario_record = state.scenario_store.get(raw_scenario)
        except ScenarioNotFoundError:
            # The application tool will deterministically report scenario_not_found. Approval is not
            # needed to discover that a referenced entity is absent.
            decision = ApprovalPolicy().assess(
                tool_name=tool_name,
                problem=state.problem_adapter.problem,
                problem_id=active_problem_id,
                fast=fast,
                parameter_overrides={},
            )
            return decision, {
                "active_problem_id": active_problem_id,
                "tool_name": tool_name,
                "scenario_reference": raw_scenario,
                "scenario_exists": False,
                "fast": fast,
            }, None
        scenario_handle = scenario_record.definition.handle
        scenario_name = scenario_record.definition.name
        scenario_solution_handle = scenario_record.solution_handle
        parameter_overrides = dict(scenario_record.definition.parameter_overrides)

    decision = ApprovalPolicy().assess(
        tool_name=tool_name,
        problem=state.problem_adapter.problem,
        problem_id=active_problem_id,
        fast=fast,
        parameter_overrides=parameter_overrides,
        scenario_handle=scenario_handle,
        scenario_name=scenario_name,
    )
    state_payload: Dict[str, object] = {
        "active_problem_id": active_problem_id,
        "tool_name": tool_name,
        "fast": fast,
        "parameter_overrides": dict(sorted(parameter_overrides.items())),
        "scenario_handle": scenario_handle,
        "scenario_name": scenario_name,
        "scenario_solution_handle": scenario_solution_handle,
        "resolved_binding_values": dict(sorted(resolved.items())),
    }
    return decision, state_payload, None


def _control_state_fingerprint(
    *,
    step: PlanStep,
    state: _AgentRunState | _MultiDomainAgentRunState,
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext,
) -> tuple[str | None, object | None, Dict[str, object], str | None]:
    decision, state_payload, error = _approval_context_for_step(
        step=step,
        state=state,
        completed_outputs=completed_outputs,
        session_entity_context=session_entity_context,
    )
    if error is not None:
        return None, decision, state_payload, error
    return stable_fingerprint(state_payload), decision, state_payload, None


def _schema_node_for_argument_path(
    tool_schema: Mapping[str, object], path: str
) -> Dict[str, object] | None:
    parameters = tool_schema.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    node: object = parameters
    for token in path.split("."):
        if not isinstance(node, Mapping):
            return None
        properties = node.get("properties")
        if not isinstance(properties, Mapping):
            return None
        child = properties.get(token)
        if not isinstance(child, Mapping):
            return None
        node = child
    return node if isinstance(node, dict) else dict(node)


def _constrain_executor_schema_to_plan(
    *,
    step: PlanStep,
    tool_schema: Mapping[str, object],
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext | None = None,
    active_problem_id: str | None = None,
) -> tuple[Dict[str, object] | None, str | None]:
    """Return a private per-step schema constrained by validated argument provenance.

    Public application tool schemas remain unchanged. This private schema makes the provider
    structurally prefer the literal/deferred values that the plan permits while application defaults remain internal.
    """

    constrained = deepcopy(dict(tool_schema))
    tool_name = str(constrained.get("name", ""))

    # target_problem_id is already an immutable plan literal and remains outside the generic
    # argument binding list to avoid duplicating domain-transition semantics.
    if tool_name == "select_problem" and step.target_problem_id is not None:
        node = _schema_node_for_argument_path(constrained, "problem_id")
        if node is None:
            return None, "select_problem schema has no problem_id argument."
        # mutate the real nested mapping rather than the detached fallback returned above
        constrained["parameters"]["properties"]["problem_id"]["enum"] = [step.target_problem_id]

    binding_paths = {binding.argument_path for binding in step.argument_bindings}
    for binding in step.argument_bindings:
        values, error = _binding_allowed_values(
            binding,
            completed_outputs=completed_outputs,
            session_entity_context=session_entity_context,
            active_problem_id=active_problem_id,
        )
        if error is not None:
            return None, error
        # Traverse the mutable copied schema.
        node: object = constrained.get("parameters")
        for token in binding.argument_path.split("."):
            if not isinstance(node, dict):
                return None, f"Binding path {binding.argument_path!r} is not present in the current tool schema."
            properties = node.get("properties")
            if not isinstance(properties, dict) or token not in properties:
                return None, f"Binding path {binding.argument_path!r} is not present in the current tool schema."
            node = properties[token]
        if not isinstance(node, dict):
            return None, f"Binding path {binding.argument_path!r} is not a scalar schema node."
        node["enum"] = list(values)
        # A user-provided safe opaque missing handle must remain literal instead of being
        # silently normalized to a generated handle. Only the private validate schema widens
        # its lexical pattern; the public application schema remains unchanged.
        if tool_name == "validate_solution" and binding.argument_path == "handle":
            node.pop("pattern", None)

    # Planner-authored defaults are forbidden. Fast=false is an application execution default
    # when the user did not explicitly bind fast/quick mode.
    if tool_name in {"run_optimization", "run_scenario"} and "fast" not in binding_paths:
        fast_node = _schema_node_for_argument_path(constrained, "fast")
        if fast_node is None:
            return None, f"{tool_name} schema has no fast argument."
        constrained["parameters"]["properties"]["fast"]["enum"] = [False]

    # Existing public schemas require every OptimizationProblem parameter key with null for
    # unspecified overrides. Constrain every *unbound* override leaf to null so the executor
    # cannot introduce a schema-valid but user-undeclared business/solver override.
    if tool_name in {"run_optimization", "create_scenario"}:
        parameters = constrained.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        override_schema = properties.get("parameter_overrides") if isinstance(properties, dict) else None
        override_props = override_schema.get("properties") if isinstance(override_schema, dict) else None
        if isinstance(override_props, dict):
            for parameter_name, parameter_schema in override_props.items():
                path = f"parameter_overrides.{parameter_name}"
                if path not in binding_paths and isinstance(parameter_schema, dict):
                    parameter_schema["enum"] = [None]

    return constrained, None


def _validate_argument_intent(
    *,
    step: PlanStep,
    tool_name: str,
    arguments: object,
    completed_outputs: list[Dict[str, object]],
    session_entity_context: SessionEntityContext | None = None,
    active_problem_id: str | None = None,
) -> _ArgumentIntentResult:
    if not isinstance(arguments, Mapping):
        return _ArgumentIntentResult(False, "Tool arguments are not an object.")

    literal_bindings = [b for b in step.argument_bindings if b.binding_type == "literal"]
    deferred_bindings = [b for b in step.argument_bindings if b.binding_type == "deferred"]
    literal_ok = True if literal_bindings else None
    deferred_ok = True if deferred_bindings else None
    messages: list[str] = []

    for binding in step.argument_bindings:
        present, actual = _argument_path_value(arguments, binding.argument_path)
        if not present:
            messages.append(f"Bound argument {binding.argument_path!r} is missing.")
            if binding.binding_type == "literal":
                literal_ok = False
            if binding.binding_type == "deferred":
                deferred_ok = False
            continue
        allowed, error = _binding_allowed_values(
            binding,
            completed_outputs=completed_outputs,
            session_entity_context=session_entity_context,
            active_problem_id=active_problem_id,
        )
        if error is not None:
            messages.append(error)
            if binding.binding_type == "literal":
                literal_ok = False
            if binding.binding_type == "deferred":
                deferred_ok = False
            continue
        if not any(_scalar_values_equal(actual, candidate) for candidate in allowed):
            messages.append(f"Argument {binding.argument_path!r} violates its {binding.binding_type} binding.")
            if binding.binding_type == "literal":
                literal_ok = False
            if binding.binding_type == "deferred":
                deferred_ok = False

    unexpected: list[str] = []
    bound_paths = {binding.argument_path for binding in step.argument_bindings}
    if tool_name in {"run_optimization", "run_scenario"} and "fast" not in bound_paths:
        if arguments.get("fast") is not False:
            unexpected.append("fast")
            messages.append("Unbound fast mode must use the application default false.")

    if tool_name in {"run_optimization", "create_scenario"}:
        overrides = arguments.get("parameter_overrides")
        if isinstance(overrides, Mapping):
            for parameter_name, value in overrides.items():
                path = f"parameter_overrides.{parameter_name}"
                if value is not None and path not in bound_paths:
                    unexpected.append(path)
        elif overrides is not None:
            messages.append("parameter_overrides is not an object.")

    if unexpected:
        messages.append(f"Undeclared parameter override(s): {sorted(unexpected)}")

    valid = not messages
    return _ArgumentIntentResult(
        valid=valid,
        message=" ".join(messages) if messages else None,
        literal_binding_count=len(literal_bindings),
        literal_bindings_preserved=literal_ok,
        deferred_binding_count=len(deferred_bindings),
        deferred_bindings_resolved=deferred_ok,
        unexpected_argument_paths=tuple(sorted(unexpected)),
    )


def dispatch_tool(
    name: str,
    arguments: object,
    context: object,
    state: _AgentRunState | _MultiDomainAgentRunState | None = None,
) -> Dict[str, object]:
    """Execute only explicitly registered agent-facing tools.

    Legacy single-domain callers see exactly the original nine-tool surface. Multi-domain
    run state additionally enables the two registry controls without altering TOOL_REGISTRY.
    """
    if isinstance(state, _MultiDomainAgentRunState) and name in {"list_problems", "select_problem"}:
        error = _validate_multi_domain_arguments(name, arguments, state)
        if error is not None:
            return {
                "ok": False,
                "error": {"code": "invalid_tool_arguments", "message": error},
            }
        if name == "list_problems":
            return _execute_list_problems(state)
        return _execute_select_problem(arguments, state)

    definition = TOOL_REGISTRY.get(name)
    if definition is None:
        return {
            "ok": False,
            "error": {
                "code": "unknown_tool",
                "message": f"Tool '{name}' is not registered.",
            },
        }

    if state is None:
        adapter = BonsaiProblemAdapter(context)
        run_state: _AgentRunState | _MultiDomainAgentRunState = _AgentRunState(
            problem_adapter=adapter,
            scenario_store=ScenarioStore(
                problem=adapter.problem,
                problem_id=adapter.problem_id,
            ),
            run_history_store=RunHistoryStore(
                default_problem_id=adapter.problem_id
            ),
        )
    else:
        run_state = state
    adapter = _adapter_for(context, run_state)
    error = (
        _validate_multi_domain_arguments(name, arguments, run_state)
        if isinstance(run_state, _MultiDomainAgentRunState)
        else _validate_arguments(name, arguments, adapter)
    )
    if error is not None:
        return {
            "ok": False,
            "error": {"code": "invalid_tool_arguments", "message": error},
        }

    return definition.execute(context, arguments, run_state)


def create_llm_client(config: LLMProviderConfig) -> Any:
    """Create an OpenAI-compatible client for the configured application provider."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenAI SDK not installed. Install requirements-agent.txt to run Agent V0."
        ) from exc

    if config.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when provider=openai.")
        kwargs: Dict[str, object] = {}
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        # Deliberately let the SDK read OPENAI_API_KEY through its normal environment path.
        return OpenAI(**kwargs)

    if config.provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is required when provider=nvidia.")
        return OpenAI(api_key=api_key, base_url=config.base_url or NVIDIA_NIM_BASE_URL)

    # LLMProviderConfig validates this earlier; keep the factory closed as a second guard.
    raise ValueError(f"Proveedor LLM no soportado: {config.provider!r}")


def create_openai_client() -> Any:
    """Backward-compatible OpenAI-only factory retained from Agent V0."""
    model = os.getenv("BONSAI_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.6"
    base_url = os.getenv("BONSAI_LLM_BASE_URL")
    return create_llm_client(
        LLMProviderConfig(provider="openai", model=model, base_url=base_url)
    )


class DecisionAgentV0:
    """Responses API orchestration for one domain or an explicit multi-domain session."""

    def __init__(
        self,
        context: object,
        *,
        problem_adapter: OptimizationProblemAdapter | None = None,
        problem_registry: ProblemRegistry | None = None,
        multi_domain_session: MultiDomainSessionState | None = None,
        config: AgentConfig | None = None,
        client: _ModelClient | None = None,
        solution_store: SolutionStore | None = None,
        scenario_store: ScenarioStore | None = None,
        run_history_store: RunHistoryStore | None = None,
        explicit_planning: bool = False,
        controlled_execution: bool = False,
    ) -> None:
        self.context = context
        self.problem_registry = problem_registry
        self.multi_domain_session = multi_domain_session
        self.explicit_planning = bool(explicit_planning)
        self.controlled_execution = bool(controlled_execution)
        self._plan_counter = 0
        self.plan_execution_records: list[PlanExecutionRecord] = []
        self.last_execution_plan: ExecutionPlan | None = None
        self.last_plan_execution_record: PlanExecutionRecord | None = None
        self.session_entity_context = SessionEntityContext()
        self.requirement_capability_guard = RequirementCapabilityGuard()
        self.last_capability_guard_decision: CapabilityGuardDecision | None = None
        self.approval_policy = ApprovalPolicy()
        self.approval_store = ApprovalStore()
        self.execution_control_ledger = ExecutionControlLedger()
        self._controlled_plan_runtimes: Dict[str, _ControlledPlanRuntime] = {}
        self.last_approval_record: ApprovalRecord | None = None

        if problem_registry is None:
            if multi_domain_session is not None:
                raise ValueError("multi_domain_session requires problem_registry.")
            if problem_adapter is None:
                raise ValueError("problem_adapter is required in single-domain mode.")
            self._problem_adapter = problem_adapter
            self._solution_store = solution_store or SolutionStore()
            self._scenario_store = scenario_store or ScenarioStore(
                problem=problem_adapter.problem,
                problem_id=problem_adapter.problem_id,
            )
            self._run_history_store = run_history_store or RunHistoryStore(
                default_problem_id=problem_adapter.problem_id
            )
        else:
            if problem_adapter is not None:
                raise ValueError("Pass either problem_adapter or problem_registry, not both.")
            if any(item is not None for item in (solution_store, scenario_store, run_history_store)):
                raise ValueError(
                    "Multi-domain stores must be supplied through MultiDomainSessionState."
                )
            if multi_domain_session is None:
                multi_domain_session = MultiDomainSessionState(
                    problem_registry, active_problem_id=problem_registry.problem_ids[0]
                )
                self.multi_domain_session = multi_domain_session
            if multi_domain_session.registry is not problem_registry:
                raise ValueError("MultiDomainSessionState must use the supplied ProblemRegistry.")
            self._problem_adapter = multi_domain_session.active.adapter
            self._solution_store = multi_domain_session.active.solution_store
            self._scenario_store = multi_domain_session.active.scenario_store
            self._run_history_store = multi_domain_session.active.run_history_store

        if self.explicit_planning and not self.is_multi_domain:
            raise ValueError("Explicit planning is supported only in unified multi-domain mode.")
        if self.controlled_execution and not self.explicit_planning:
            raise ValueError("Controlled execution requires explicit_planning=True.")
        if self.controlled_execution and not self.is_multi_domain:
            raise ValueError("Controlled execution is supported only in unified multi-domain mode.")

        self.system_instructions = self._current_system_instructions()
        self.llm_tools = self._current_llm_tools()
        self.config = config or AgentConfig()
        resolved_llm = self.config.llm
        if resolved_llm is None:  # defensive: AgentConfig.__post_init__ always resolves it
            raise RuntimeError("AgentConfig did not resolve an LLM provider configuration.")
        self.llm_config = resolved_llm
        self.client = client if client is not None else create_llm_client(resolved_llm)

    @property
    def is_multi_domain(self) -> bool:
        return self.problem_registry is not None and self.multi_domain_session is not None

    @property
    def problem_adapter(self) -> OptimizationProblemAdapter:
        if self.is_multi_domain:
            return self.multi_domain_session.active.adapter
        return self._problem_adapter

    @problem_adapter.setter
    def problem_adapter(self, value: OptimizationProblemAdapter) -> None:
        if self.is_multi_domain:
            raise AttributeError("problem_adapter is selected through the multi-domain session.")
        self._problem_adapter = value

    @property
    def solution_store(self) -> SolutionStore:
        if self.is_multi_domain:
            return self.multi_domain_session.active.solution_store
        return self._solution_store

    @solution_store.setter
    def solution_store(self, value: SolutionStore) -> None:
        if self.is_multi_domain:
            self.multi_domain_session.active.solution_store = value
        else:
            self._solution_store = value

    @property
    def scenario_store(self) -> ScenarioStore:
        if self.is_multi_domain:
            return self.multi_domain_session.active.scenario_store
        return self._scenario_store

    @scenario_store.setter
    def scenario_store(self, value: ScenarioStore) -> None:
        if self.is_multi_domain:
            self.multi_domain_session.active.scenario_store = value
        else:
            self._scenario_store = value

    @property
    def run_history_store(self) -> RunHistoryStore:
        if self.is_multi_domain:
            return self.multi_domain_session.active.run_history_store
        return self._run_history_store

    @run_history_store.setter
    def run_history_store(self, value: RunHistoryStore) -> None:
        if self.is_multi_domain:
            self.multi_domain_session.active.run_history_store = value
        else:
            self._run_history_store = value

    def _current_system_instructions(self) -> str:
        adapter = self.problem_adapter
        if self.is_multi_domain:
            return build_multi_domain_system_instructions(adapter.spec, self.problem_registry)
        return build_system_instructions(adapter.spec)

    def _current_llm_tools(self) -> list[Dict[str, object]]:
        adapter = self.problem_adapter
        if self.is_multi_domain:
            return _llm_tools_for_registry(adapter.spec, self.problem_registry)
        return _llm_tools_for_spec(adapter.spec)

    def _new_run_state(self) -> _AgentRunState | _MultiDomainAgentRunState:
        if self.is_multi_domain:
            return _MultiDomainAgentRunState(self.multi_domain_session)
        return _AgentRunState(
            solution_store=self.solution_store,
            scenario_store=self.scenario_store,
            run_history_store=self.run_history_store,
            problem_adapter=self.problem_adapter,
        )

    def run(self, user_message: str) -> AgentRunResult:
        if self.controlled_execution and any(
            runtime.plan_status in {"awaiting_approval", "paused_recoverable", "manual_intervention_required"}
            for runtime in self._controlled_plan_runtimes.values()
        ):
            return AgentRunResult(
                text=(
                    "A controlled plan is paused. Use approve/reject for a pending approval or "
                    "resume for a recoverable pre-dispatch provider failure before starting a new plan."
                ),
                tool_names=(),
            )
        if self.explicit_planning:
            return self._run_planned(user_message)
        return self._run_reactive(user_message)

    def _next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"plan_{self._plan_counter:06d}"

    def _plan_validator(self, *, historical_reference_anchor: str | None = None) -> PlanValidator:
        if not self.is_multi_domain:
            raise RuntimeError("Plan validation requires multi-domain mode.")
        active_problem_id = self.multi_domain_session.active_problem_id
        active_recent_entities = {
            record.reference_id: record
            for record in self.session_entity_context.current(problem_id=active_problem_id)
        }
        return PlanValidator(
            application_tool_names=[*TOOL_REGISTRY.keys(), "list_problems", "select_problem"],
            registered_problem_ids=self.problem_registry.problem_ids,
            session_entities=active_recent_entities,
            historical_reference_anchor=historical_reference_anchor,
            max_steps=MAX_PLAN_STEPS,
        )

    def _request_execution_plan(self, user_message: str) -> ExecutionPlan:
        if not self.is_multi_domain:
            raise RuntimeError("Explicit planning requires multi-domain mode.")
        plan_id = self._next_plan_id()
        active_problem_id = self.multi_domain_session.active_problem_id
        historical_reference_anchor = _explicit_historical_reference_anchor(user_message)
        recent_session_entities = self.session_entity_context.planner_entities(
            problem_id=active_problem_id
        )
        planner_context = {
            "active_problem_id": active_problem_id,
            "active_problem": {
                "problem_id": self.problem_adapter.problem_id,
                "display_name": self.problem_adapter.spec.display_name,
                "supported_capabilities": list(self.problem_adapter.spec.supported_capabilities),
                "unsupported_change_categories": list(self.problem_adapter.spec.unsupported_change_categories),
            },
            "registered_problems": [metadata.to_dict() for metadata in self.problem_registry.metadata()],
            "application_tools": [*TOOL_REGISTRY.keys(), "list_problems", "select_problem"],
            "recent_session_entities": list(recent_session_entities),
            "historical_reference_anchor": historical_reference_anchor,
            "max_plan_steps": MAX_PLAN_STEPS,
        }
        submit_plan_schema = _planner_submit_plan_schema(
            self.problem_registry,
            session_entities=recent_session_entities,
            historical_reference_anchor=historical_reference_anchor,
        )
        try:
            planner_transport, planner_options = build_planner_request_options(
                provider=self.llm_config.provider,
                submit_plan_schema=submit_plan_schema,
            )
        except PlannerTransportUnsupportedError as exc:
            raise PlanValidationError(
                "planner_transport_unsupported",
                str(exc),
            ) from exc

        response = self.client.responses.create(
            model=self.llm_config.model,
            instructions=(
                _PLANNER_INSTRUCTIONS
                + "\nAPPLICATION PLANNING CONTEXT\n"
                + json.dumps(planner_context, ensure_ascii=False, sort_keys=True)
            ),
            input=[{"role": "user", "content": user_message}],
            **planner_options,
        )
        normalized = normalize_planner_response(response, transport=planner_transport)
        emit_planner_protocol_debug(normalized)
        calls = normalized.tool_calls
        if len(calls) != 1 or calls[0].name != "submit_plan":
            raise PlanValidationError(
                "planner_protocol_error",
                "Planner must return exactly one submit_plan function call.",
            )
        raw_arguments = calls[0].arguments
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PlanValidationError(
                "planner_invalid_json",
                "Planner submit_plan arguments were not valid JSON.",
            ) from exc
        plan = _plan_from_submission(
            plan_id=plan_id,
            user_goal=user_message,
            initial_problem_id=active_problem_id,
            arguments=arguments,
        )
        self._plan_validator(historical_reference_anchor=historical_reference_anchor).validate(plan)
        # CORE-FIX-01: the accepted plan and every approval fingerprint see typed literals.
        _canonicalize_plan_numeric_bindings(plan, self.problem_registry)
        return plan

    def _record_plan_execution(
        self,
        plan: ExecutionPlan,
        *,
        initial_problem_id: str,
        actual_tool_trace: list[str],
        step_summaries: list[PlanStepExecutionSummary],
        validation_error: str | None = None,
        final_synthesis_error: str | None = None,
        plan_status: str | None = None,
    ) -> PlanExecutionRecord:
        final_problem_id = (
            self.multi_domain_session.active_problem_id
            if self.is_multi_domain
            else self.problem_adapter.problem_id
        )
        if plan_status is None:
            if validation_error is not None or any(step.status == "failed" for step in step_summaries):
                plan_status = "failed"
            else:
                plan_status = "completed"
        record = PlanExecutionRecord(
            plan_id=plan.plan_id,
            initial_problem_id=initial_problem_id,
            final_problem_id=final_problem_id,
            planned_tool_trace=plan.planned_tool_trace,
            actual_tool_trace=tuple(actual_tool_trace),
            steps=tuple(step_summaries),
            validation_error=validation_error,
            final_synthesis_error=final_synthesis_error,
            plan_status=plan_status,
        )
        replaced_existing = False
        for index, existing in enumerate(self.plan_execution_records):
            if existing.plan_id == plan.plan_id:
                self.plan_execution_records[index] = record
                replaced_existing = True
                break
        if not replaced_existing:
            self.plan_execution_records.append(record)
        self.last_execution_plan = plan
        self.last_plan_execution_record = record
        return record

    def _synthesize_plan_result(
        self,
        *,
        user_message: str,
        plan: ExecutionPlan,
        record: PlanExecutionRecord,
    ) -> str:
        self.system_instructions = self._current_system_instructions()
        payload = {
            "original_user_goal": user_message,
            "execution_plan": plan.to_dict(),
            "execution_record": record.to_dict(),
        }
        try:
            response = self.client.responses.create(
                model=self.llm_config.model,
                instructions=self.system_instructions + _FINAL_SYNTHESIS_GROUNDING,
                input=[
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    }
                ],
            )
        except Exception as exc:
            raise FinalSynthesisProviderError("Final synthesis provider call failed.") from exc
        return str(getattr(response, "output_text", "") or "")

    @staticmethod
    def _final_synthesis_fallback(record: PlanExecutionRecord) -> str:
        """Return deterministic safe metadata when tool-free synthesis provider call fails."""
        handles: dict[str, list[object]] = {}
        safe_keys = {
            "solution_handle", "historical_solution_handle", "scenario_handle",
            "current_alias", "run_id", "active_problem_id",
        }

        def collect(value: object, *, parent_key: str | None = None) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    key_text = str(key)
                    if key_text in safe_keys and isinstance(item, (str, int, float, bool)):
                        handles.setdefault(key_text, []).append(item)
                    elif key_text == "handle" and parent_key == "scenario" and isinstance(item, str):
                        handles.setdefault("scenario_handle", []).append(item)
                    collect(item, parent_key=key_text)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item, parent_key=parent_key)

        for step in record.steps:
            if step.status == "completed":
                collect(dict(step.outcome_metadata or {}))
        deduped = {
            key: list(dict.fromkeys(values))
            for key, values in sorted(handles.items())
            if values
        }
        status_map = {step.step_id: step.status for step in record.steps}
        prefix = (
            "Execution completed, but final LLM synthesis failed "
            if record.succeeded
            else "Plan execution reached its recorded terminal state, but final LLM synthesis failed "
        )
        return (
            prefix
            + "(final_synthesis_provider_error). "
            f"Plan ID: {record.plan_id}. "
            f"Actual tool trace: {list(record.actual_tool_trace)}. "
            f"Step statuses: {status_map}. "
            f"Registered handle metadata: {deduped}."
        )

    def _mark_final_synthesis_error(
        self, record: PlanExecutionRecord, error_code: str = "final_synthesis_provider_error"
    ) -> PlanExecutionRecord:
        updated = replace(record, final_synthesis_error=error_code)
        for index in range(len(self.plan_execution_records) - 1, -1, -1):
            if self.plan_execution_records[index].plan_id == record.plan_id:
                self.plan_execution_records[index] = updated
                break
        self.last_plan_execution_record = updated
        return updated

    def _controlled_record_steps(self, runtime: _ControlledPlanRuntime) -> list[PlanStepExecutionSummary]:
        summaries: list[PlanStepExecutionSummary] = []
        for step in runtime.plan.steps:
            existing = runtime.step_summaries.get(step.step_id)
            if existing is not None:
                summaries.append(existing)
            else:
                summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=step.expected_application_tool,
                        status=step.status,
                    )
                )
        return summaries

    def _update_controlled_record(
        self,
        runtime: _ControlledPlanRuntime,
        *,
        validation_error: str | None = None,
        final_synthesis_error: str | None = None,
    ) -> PlanExecutionRecord:
        return self._record_plan_execution(
            runtime.plan,
            initial_problem_id=runtime.initial_problem_id,
            actual_tool_trace=runtime.actual_tool_trace,
            step_summaries=self._controlled_record_steps(runtime),
            validation_error=validation_error,
            final_synthesis_error=final_synthesis_error,
            plan_status=runtime.plan_status,
        )

    def _finalize_controlled_turn(self, runtime: _ControlledPlanRuntime) -> None:
        if runtime.turn_open:
            self.session_entity_context.finalize_turn()
            runtime.turn_open = False

    @staticmethod
    def _skip_remaining_controlled_steps(
        runtime: _ControlledPlanRuntime,
        *,
        after_step_id: str,
        error_code: str,
    ) -> None:
        seen_current = False
        for step in runtime.plan.steps:
            if step.step_id == after_step_id:
                seen_current = True
                continue
            if not seen_current:
                continue
            if step.status in {"completed", "failed", "rejected", "outcome_unknown", "skipped", "blocked_unsupported_requirement"}:
                continue
            step.status = "skipped"
            runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                step_id=step.step_id,
                expected_application_tool=step.expected_application_tool,
                status="skipped",
                error_code=error_code,
            )

    def _apply_capability_guard_decision(
        self,
        runtime: _ControlledPlanRuntime,
        decision: CapabilityGuardDecision,
    ) -> AgentRunResult:
        """Terminate a validated plan deterministically before approval/executor dispatch."""

        metadata = {"capability_guard": decision.to_dict()}
        affected = set(decision.affected_step_ids)
        if decision.informational:
            # The user asked what is supported, not to execute an unsupported action.  The guard
            # answers from problem-owned metadata and skips any planner-proposed mutation/read used
            # only to answer that capability question.
            for step in runtime.plan.steps:
                if step.status != "pending":
                    continue
                step.status = "skipped"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=step.expected_application_tool,
                    status="skipped",
                    outcome_metadata=metadata,
                )
            runtime.plan_status = "completed"
            self._finalize_controlled_turn(runtime)
            self._update_controlled_record(runtime)
            return AgentRunResult(text=decision.user_text(), tool_names=())

        if not decision.blocked:
            raise RuntimeError("Capability-guard terminal handling requires blocked/informational decision.")

        for step in runtime.plan.steps:
            if step.status != "pending":
                continue
            if step.step_id in affected:
                self.execution_control_ledger.mark_blocked_unsupported_requirement(
                    runtime.plan.plan_id,
                    step.step_id,
                    error_code=decision.reason_code or "unsupported_required_capability",
                )
                step.status = "blocked_unsupported_requirement"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=step.expected_application_tool,
                    status="blocked_unsupported_requirement",
                    error_code=decision.reason_code or "unsupported_required_capability",
                    outcome_metadata=metadata,
                )
            else:
                step.status = "skipped"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=step.expected_application_tool,
                    status="skipped",
                    error_code="unsupported_required_capability",
                    outcome_metadata=metadata,
                )
        runtime.plan_status = "blocked_unsupported_requirement"
        runtime.pending_approval_id = None
        runtime.recoverable_step_id = None
        self._finalize_controlled_turn(runtime)
        self._update_controlled_record(runtime)
        return AgentRunResult(text=decision.user_text(), tool_names=())

    @staticmethod
    def _approval_required_text(record: ApprovalRecord) -> str:
        summary = record.summary_dict()
        parts = [
            f"Approval required: {record.approval_id}",
            f"Plan: {record.plan_id}",
            f"Step: {record.tool_name}",
            f"Problem: {record.problem_id}",
            f"Reason: {', '.join(record.reason_codes)}",
        ]
        scenario_name = summary.get("scenario_name")
        if isinstance(scenario_name, str) and scenario_name:
            parts.append(f"Scenario: {scenario_name}")
        overrides = summary.get("parameter_overrides")
        if isinstance(overrides, Mapping) and overrides:
            rendered = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
            parts.append(f"Overrides: {rendered}")
        if summary.get("fast") is True:
            parts.append("Fast mode: requested")
        parts.append(f"Use 'approve {record.approval_id}' or 'reject {record.approval_id}'.")
        return "\n".join(parts)

    def _pending_approval(self, approval_id: str | None) -> ApprovalRecord:
        if approval_id is not None:
            return self.approval_store.get(approval_id)
        pending = self.approval_store.pending()
        if len(pending) != 1:
            raise ExecutionControlError(
                "approval_id_required",
                "An approval ID is required unless exactly one approval request is pending.",
            )
        return pending[0]

    def _controlled_runtime(self, plan_id: str) -> _ControlledPlanRuntime:
        try:
            return self._controlled_plan_runtimes[plan_id]
        except KeyError as exc:
            raise PlanControlNotFoundError(plan_id) from exc

    def _approval_still_matches(
        self,
        runtime: _ControlledPlanRuntime,
        step: PlanStep,
        approval: ApprovalRecord,
    ) -> tuple[bool, str | None]:
        plan_fp = stable_fingerprint(_plan_contract_payload(runtime.plan))
        step_fp = stable_fingerprint(_step_contract_payload(step))
        state_fp, _decision, _state_payload, error = _control_state_fingerprint(
            step=step,
            state=runtime.state,
            completed_outputs=runtime.completed_outputs,
            session_entity_context=self.session_entity_context,
        )
        if error is not None or state_fp is None:
            return False, error or "Unable to recompute approval state fingerprint."
        valid = (
            approval.plan_fingerprint == plan_fp == runtime.plan_fingerprint
            and approval.step_fingerprint == step_fp
            and approval.state_fingerprint == state_fp
            and approval.problem_id == self.multi_domain_session.active_problem_id
        )
        emit_control_debug(
            {
                "plan_id": runtime.plan.plan_id,
                "step_id": step.step_id,
                "approval_id": approval.approval_id,
                "approval_status": approval.status,
                "approval_reason_codes": list(approval.reason_codes),
                "execution_ledger_state": self.execution_control_ledger.get(runtime.plan.plan_id, step.step_id).state,
                "resume_allowed": False,
                "state_fingerprint_match": valid,
            }
        )
        return valid, None if valid else "Approval context no longer matches the plan/step/application state."

    def approve(self, approval_id: str | None = None) -> AgentRunResult:
        """Deterministically approve and resume one paused Step 18 plan without replanning."""
        if not self.controlled_execution:
            return AgentRunResult(text="Controlled approvals are not enabled for this agent.", tool_names=())
        try:
            approval = self._pending_approval(approval_id)
            runtime = self._controlled_runtime(approval.plan_id)
        except ExecutionControlError as exc:
            return AgentRunResult(text=f"Approval failed: {exc.code}. {exc.message}", tool_names=())
        step = next((item for item in runtime.plan.steps if item.step_id == approval.step_id), None)
        if step is None or step.status != "pending_approval" or runtime.plan_status != "awaiting_approval":
            return AgentRunResult(
                text=f"Approval failed: approval_not_pending. Approval {approval.approval_id} is not awaiting execution.",
                tool_names=(),
            )
        matches, message = self._approval_still_matches(runtime, step, approval)
        if not matches:
            try:
                expired = self.approval_store.expire(approval.approval_id)
            except ExecutionControlError:
                expired = approval
            self.last_approval_record = expired
            step.status = "failed"
            runtime.plan_status = "failed"
            runtime.pending_approval_id = None
            runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                step_id=step.step_id,
                expected_application_tool=step.expected_application_tool,
                status="failed",
                error_code="stale_approval_context",
                outcome_metadata={"message": message or "Approval context changed."},
            )
            self._skip_remaining_controlled_steps(
                runtime, after_step_id=step.step_id, error_code="stale_approval_context"
            )
            self._finalize_controlled_turn(runtime)
            self._update_controlled_record(runtime)
            return AgentRunResult(
                text=f"Approval expired: stale_approval_context. {message or ''}".strip(),
                tool_names=(),
            )
        try:
            approved = self.approval_store.approve(approval.approval_id)
        except ExecutionControlError as exc:
            return AgentRunResult(text=f"Approval failed: {exc.code}. {exc.message}", tool_names=())
        self.last_approval_record = approved
        runtime.pending_approval_id = None
        runtime.plan_status = "running"
        step.status = "pending"
        return self._execute_controlled_runtime(runtime)

    def reject(self, approval_id: str | None = None) -> AgentRunResult:
        """Reject one pending approval without executing the protected application step."""
        if not self.controlled_execution:
            return AgentRunResult(text="Controlled approvals are not enabled for this agent.", tool_names=())
        try:
            approval = self._pending_approval(approval_id)
            runtime = self._controlled_runtime(approval.plan_id)
            rejected = self.approval_store.reject(approval.approval_id)
        except ExecutionControlError as exc:
            return AgentRunResult(text=f"Rejection failed: {exc.code}. {exc.message}", tool_names=())
        step = next((item for item in runtime.plan.steps if item.step_id == approval.step_id), None)
        if step is None or step.status != "pending_approval":
            return AgentRunResult(text="Rejection failed: approval_not_pending.", tool_names=())
        self.last_approval_record = rejected
        self.execution_control_ledger.mark_rejected(
            runtime.plan.plan_id, step.step_id, approval_id=approval.approval_id
        )
        step.status = "rejected"
        runtime.plan_status = "rejected"
        runtime.pending_approval_id = None
        runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
            step_id=step.step_id,
            expected_application_tool=step.expected_application_tool,
            status="rejected",
            error_code="approval_rejected",
        )
        self._skip_remaining_controlled_steps(
            runtime, after_step_id=step.step_id, error_code="approval_rejected"
        )
        self._finalize_controlled_turn(runtime)
        self._update_controlled_record(runtime)
        emit_control_debug(
            {
                "plan_id": runtime.plan.plan_id,
                "step_id": step.step_id,
                "approval_id": approval.approval_id,
                "approval_status": "rejected",
                "approval_reason_codes": list(approval.reason_codes),
                "execution_ledger_state": "rejected",
                "resume_allowed": False,
                "state_fingerprint_match": True,
            }
        )
        return AgentRunResult(
            text=(
                f"Plan {runtime.plan.plan_id} rejected. Step {step.step_id} ({step.expected_application_tool}) "
                "was not executed; remaining dependent steps were skipped."
            ),
            tool_names=(),
        )

    def resume(self, plan_id: str) -> AgentRunResult:
        """Resume only a pre-dispatch provider failure on the same existing plan/step."""
        if not self.controlled_execution:
            return AgentRunResult(text="Controlled resume is not enabled for this agent.", tool_names=())
        try:
            runtime = self._controlled_runtime(plan_id)
        except ExecutionControlError as exc:
            return AgentRunResult(text=f"Resume failed: {exc.code}. {exc.message}", tool_names=())
        if runtime.plan_status != "paused_recoverable" or runtime.recoverable_step_id is None:
            return AgentRunResult(text=f"Resume failed: resume_not_allowed. Plan {plan_id} is not recoverable.", tool_names=())
        step = next(item for item in runtime.plan.steps if item.step_id == runtime.recoverable_step_id)
        ledger = self.execution_control_ledger.get(plan_id, step.step_id)
        if ledger.state != "failed_before_dispatch" or ledger.error_code != "executor_provider_error":
            return AgentRunResult(text=f"Resume failed: resume_not_allowed. Step {step.step_id} is not a recoverable provider failure.", tool_names=())

        # Revalidate the immutable plan contract through the existing Step 17 validator without
        # changing semantics. Mutable execution statuses are reset only on this validation copy.
        plan_copy = deepcopy(runtime.plan)
        for item in plan_copy.steps:
            item.status = "pending"
        try:
            self._plan_validator(
                historical_reference_anchor=_explicit_historical_reference_anchor(runtime.user_message)
            ).validate(plan_copy)
        except PlanValidationError as exc:
            return AgentRunResult(text=f"Resume failed: {exc.code}. {exc.message}", tool_names=())
        if stable_fingerprint(_plan_contract_payload(runtime.plan)) != runtime.plan_fingerprint:
            return AgentRunResult(text="Resume failed: stale_resume_context. Plan contract changed.", tool_names=())
        state_fp, _decision, _payload, error = _control_state_fingerprint(
            step=step,
            state=runtime.state,
            completed_outputs=runtime.completed_outputs,
            session_entity_context=self.session_entity_context,
        )
        if error is not None or state_fp != ledger.state_fingerprint:
            step.status = "failed"
            runtime.plan_status = "failed"
            runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                step_id=step.step_id,
                expected_application_tool=step.expected_application_tool,
                status="failed",
                error_code="stale_resume_context",
            )
            self._skip_remaining_controlled_steps(
                runtime, after_step_id=step.step_id, error_code="stale_resume_context"
            )
            self._finalize_controlled_turn(runtime)
            self._update_controlled_record(runtime)
            return AgentRunResult(text="Resume failed: stale_resume_context.", tool_names=())

        self.execution_control_ledger.prepare_resume(plan_id, step.step_id)
        step.status = "pending"
        runtime.step_summaries.pop(step.step_id, None)
        runtime.plan_status = "running"
        runtime.recoverable_step_id = None
        emit_control_debug(
            {
                "plan_id": plan_id,
                "step_id": step.step_id,
                "approval_id": None,
                "approval_status": None,
                "approval_reason_codes": [],
                "execution_ledger_state": "not_started",
                "resume_allowed": True,
                "state_fingerprint_match": True,
            }
        )
        return self._execute_controlled_runtime(runtime)

    def _controlled_terminal_result(self, runtime: _ControlledPlanRuntime) -> AgentRunResult:
        self._finalize_controlled_turn(runtime)
        record = self._update_controlled_record(runtime)
        trace_start = getattr(runtime, "_result_trace_start", 0)
        new_tools = tuple(runtime.actual_tool_trace[trace_start:])
        try:
            text = self._synthesize_plan_result(
                user_message=runtime.user_message,
                plan=runtime.plan,
                record=record,
            )
        except FinalSynthesisProviderError:
            record = self._mark_final_synthesis_error(record)
            text = self._final_synthesis_fallback(record)
        return AgentRunResult(text=text, tool_names=new_tools)

    def _execute_controlled_runtime(self, runtime: _ControlledPlanRuntime) -> AgentRunResult:
        """Continue one validated plan under Step 18 approval/recovery/duplicate controls."""
        runtime._result_trace_start = len(runtime.actual_tool_trace)  # type: ignore[attr-defined]

        for step in runtime.plan.steps:
            if step.status == "completed":
                continue
            if step.status in {"skipped", "rejected", "failed", "outcome_unknown", "blocked_unsupported_requirement"}:
                continue
            if step.status == "recoverable_failed":
                runtime.plan_status = "paused_recoverable"
                self._update_controlled_record(runtime)
                return AgentRunResult(
                    text=f"Plan {runtime.plan.plan_id} is paused after a recoverable pre-dispatch provider failure. Use 'resume {runtime.plan.plan_id}'.",
                    tool_names=(),
                )

            expected_tool = step.expected_application_tool
            ledger = self.execution_control_ledger.ensure(runtime.plan.plan_id, step.step_id)

            # A proven completed dispatch is replayed only from its compact receipt, never dispatched again.
            if ledger.state == "dispatch_completed":
                receipt = ledger.receipt_dict()
                result = receipt.get("result")
                actual_tool = receipt.get("tool")
                if isinstance(actual_tool, str) and isinstance(result, Mapping):
                    if actual_tool not in runtime.actual_tool_trace:
                        runtime.actual_tool_trace.append(actual_tool)
                    if not any(output.get("step_id") == step.step_id for output in runtime.completed_outputs):
                        runtime.completed_outputs.append(
                            {"step_id": step.step_id, "tool": actual_tool, "result": dict(result)}
                        )
                    step.status = "completed"
                    runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="completed",
                        actual_tool=actual_tool,
                        outcome_metadata=dict(result),
                    )
                    continue
                step.status = "outcome_unknown"
                runtime.plan_status = "manual_intervention_required"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="outcome_unknown",
                    error_code="execution_receipt_missing",
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="manual_intervention_required"
                )
                self._finalize_controlled_turn(runtime)
                self._update_controlled_record(runtime)
                return AgentRunResult(
                    text="Manual intervention required: a completed dispatch lacks a safe execution receipt and will not be repeated.",
                    tool_names=(),
                )

            if ledger.state == "outcome_unknown":
                step.status = "outcome_unknown"
                runtime.plan_status = "manual_intervention_required"
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="manual_intervention_required"
                )
                self._finalize_controlled_turn(runtime)
                self._update_controlled_record(runtime)
                return AgentRunResult(
                    text=(
                        f"Manual intervention required for plan {runtime.plan.plan_id}, step {step.step_id}: "
                        "the application cannot safely determine whether the side effect occurred and will not retry it automatically."
                    ),
                    tool_names=(),
                )

            if expected_tool is None:
                step.status = "completed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=None,
                    status="completed",
                    outcome_metadata={"kind": "no_tool_step"},
                )
                continue

            current_problem_id = self.multi_domain_session.active_problem_id
            if (
                expected_tool != "select_problem"
                and step.target_problem_id is not None
                and step.target_problem_id != current_problem_id
            ):
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    error_code="execution_domain_mismatch",
                    outcome_metadata={
                        "active_problem_id": current_problem_id,
                        "target_problem_id": step.target_problem_id,
                    },
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break

            current_tools = self._current_llm_tools()
            tool_schema = next((schema for schema in current_tools if schema.get("name") == expected_tool), None)
            if tool_schema is None:
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    error_code="planned_tool_unavailable",
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break

            # Approval is assessed deterministically before calling the executor model for this step.
            state_fp, approval_decision, _state_payload, approval_error = _control_state_fingerprint(
                step=step,
                state=runtime.state,
                completed_outputs=runtime.completed_outputs,
                session_entity_context=self.session_entity_context,
            )
            if approval_error is not None or state_fp is None:
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    error_code="argument_intent_violation",
                    outcome_metadata={"message": approval_error or "Unable to resolve execution intent."},
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break

            if getattr(approval_decision, "required", False):
                if ledger.approval_id is None:
                    approval = self.approval_store.create(
                        plan_id=runtime.plan.plan_id,
                        step_id=step.step_id,
                        tool_name=expected_tool,
                        problem_id=current_problem_id,
                        reason_codes=approval_decision.reason_codes,
                        safe_action_summary=approval_decision.summary_dict(),
                        plan_fingerprint=runtime.plan_fingerprint,
                        step_fingerprint=stable_fingerprint(_step_contract_payload(step)),
                        state_fingerprint=state_fp,
                    )
                    self.last_approval_record = approval
                    self.execution_control_ledger.mark_approval_pending(
                        runtime.plan.plan_id,
                        step.step_id,
                        approval_id=approval.approval_id,
                        state_fingerprint=state_fp,
                    )
                    step.status = "pending_approval"
                    runtime.plan_status = "awaiting_approval"
                    runtime.pending_approval_id = approval.approval_id
                    runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="pending_approval",
                        outcome_metadata={
                            "approval_id": approval.approval_id,
                            "reason_codes": list(approval.reason_codes),
                            "safe_action_summary": approval.summary_dict(),
                        },
                    )
                    self._update_controlled_record(runtime)
                    emit_control_debug(
                        {
                            "plan_id": runtime.plan.plan_id,
                            "step_id": step.step_id,
                            "approval_id": approval.approval_id,
                            "approval_status": approval.status,
                            "approval_reason_codes": list(approval.reason_codes),
                            "execution_ledger_state": "approval_pending",
                            "resume_allowed": False,
                            "state_fingerprint_match": True,
                        }
                    )
                    return AgentRunResult(
                        text=self._approval_required_text(approval),
                        tool_names=tuple(runtime.actual_tool_trace[getattr(runtime, "_result_trace_start", 0):]),
                    )
                approval = self.approval_store.get(str(ledger.approval_id))
                if approval.status == "pending":
                    step.status = "pending_approval"
                    runtime.plan_status = "awaiting_approval"
                    runtime.pending_approval_id = approval.approval_id
                    self._update_controlled_record(runtime)
                    return AgentRunResult(text=self._approval_required_text(approval), tool_names=())
                if approval.status != "approved":
                    step.status = "failed"
                    runtime.plan_status = "failed"
                    runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="approval_not_usable",
                    )
                    self._skip_remaining_controlled_steps(
                        runtime, after_step_id=step.step_id, error_code="approval_not_usable"
                    )
                    break
                matches, message = self._approval_still_matches(runtime, step, approval)
                if not matches:
                    step.status = "failed"
                    runtime.plan_status = "failed"
                    runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="stale_approval_context",
                        outcome_metadata={"message": message or "Approval context changed."},
                    )
                    self._skip_remaining_controlled_steps(
                        runtime, after_step_id=step.step_id, error_code="stale_approval_context"
                    )
                    break

            constrained_tool_schema, intent_schema_error = _constrain_executor_schema_to_plan(
                step=step,
                tool_schema=tool_schema,
                completed_outputs=runtime.completed_outputs,
                session_entity_context=self.session_entity_context,
                active_problem_id=current_problem_id,
            )
            if intent_schema_error is not None or constrained_tool_schema is None:
                step.status = "failed"
                runtime.plan_status = "failed"
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id,
                    step.step_id,
                    error_code="argument_intent_violation",
                    state_fingerprint=state_fp,
                )
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    error_code="argument_intent_violation",
                    outcome_metadata={"message": intent_schema_error or "Unable to constrain executor schema."},
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break

            self.system_instructions = self._current_system_instructions()
            executor_payload = {
                "original_user_goal": runtime.user_message,
                "plan": runtime.plan.to_dict(),
                "current_step": step.to_dict(),
                "active_problem_id": current_problem_id,
                "completed_step_outputs": runtime.completed_outputs,
            }
            try:
                executor_transport, executor_options = build_executor_request_options(
                    provider=self.llm_config.provider,
                    tool_schema=constrained_tool_schema,
                )
            except ExecutorTransportUnsupportedError:
                step.status = "failed"
                runtime.plan_status = "failed"
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id,
                    step.step_id,
                    error_code="executor_transport_unsupported",
                    state_fingerprint=state_fp,
                )
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    error_code="executor_transport_unsupported",
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break

            step.status = "running"
            try:
                response = self.client.responses.create(
                    model=self.llm_config.model,
                    instructions=self.system_instructions + _EXECUTOR_PROTOCOL_GROUNDING,
                    input=[{"role": "user", "content": json.dumps(executor_payload, ensure_ascii=False, sort_keys=True)}],
                    **executor_options,
                )
            except Exception:
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id,
                    step.step_id,
                    error_code="executor_provider_error",
                    state_fingerprint=state_fp,
                )
                step.status = "recoverable_failed"
                runtime.plan_status = "paused_recoverable"
                runtime.recoverable_step_id = step.step_id
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="recoverable_failed",
                    error_code="executor_provider_error",
                    outcome_metadata={"dispatch_attempted": False},
                )
                self._update_controlled_record(runtime)
                emit_control_debug(
                    {
                        "plan_id": runtime.plan.plan_id,
                        "step_id": step.step_id,
                        "approval_id": ledger.approval_id,
                        "approval_status": (
                            self.approval_store.get(ledger.approval_id).status if ledger.approval_id else None
                        ),
                        "approval_reason_codes": [],
                        "execution_ledger_state": "failed_before_dispatch",
                        "resume_allowed": True,
                        "state_fingerprint_match": True,
                    }
                )
                return AgentRunResult(
                    text=(
                        f"Plan {runtime.plan.plan_id} paused: executor_provider_error occurred before application dispatch. "
                        f"Use 'resume {runtime.plan.plan_id}' to retry only step {step.step_id}."
                    ),
                    tool_names=tuple(runtime.actual_tool_trace[getattr(runtime, "_result_trace_start", 0):]),
                )

            normalized = normalize_executor_response(response, transport=executor_transport)
            calls = normalized.tool_calls
            if len(calls) != 1:
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="execution_protocol_error", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", error_code="execution_protocol_error",
                    outcome_metadata={"attempted_call_count": len(calls)},
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break
            call = calls[0]
            actual_name = call.name
            if actual_name != expected_tool:
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="unplanned_tool_call", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", actual_tool=actual_name or None,
                    error_code="unplanned_tool_call", outcome_metadata={"attempted_tool": actual_name},
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="execution_protocol_error", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", actual_tool=actual_name,
                    error_code="execution_protocol_error", outcome_metadata={"arguments_parsed": False},
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break

            # CORE-FIX-01: restore the pre-approved canonical float literal, not a
            # new post-approval parameter value, before intent and schema validation.
            _restore_approved_float_literals(
                step=step, arguments=arguments, problem_adapter=runtime.state.problem_adapter
            )
            intent_result = _validate_argument_intent(
                step=step,
                tool_name=expected_tool,
                arguments=arguments,
                completed_outputs=runtime.completed_outputs,
                session_entity_context=self.session_entity_context,
                active_problem_id=current_problem_id,
            )
            if not intent_result.valid:
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="argument_intent_violation", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", actual_tool=actual_name,
                    error_code="argument_intent_violation",
                    outcome_metadata={
                        "message": intent_result.message,
                        "unexpected_argument_paths": list(intent_result.unexpected_argument_paths),
                    },
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break

            argument_error = _validate_executor_arguments_against_current_tool(
                tool_name=expected_tool,
                arguments=arguments,
                tool_schema=constrained_tool_schema,
                state=runtime.state,
            )
            if argument_error is not None:
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="invalid_tool_arguments", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", actual_tool=actual_name,
                    error_code="invalid_tool_arguments", outcome_metadata={"message": argument_error},
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break

            if expected_tool == "select_problem" and (
                not isinstance(arguments, Mapping) or arguments.get("problem_id") != step.target_problem_id
            ):
                self.execution_control_ledger.mark_failed_before_dispatch(
                    runtime.plan.plan_id, step.step_id,
                    error_code="selection_target_mismatch", state_fingerprint=state_fp,
                )
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", actual_tool=actual_name, error_code="selection_target_mismatch",
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break

            # This is the duplicate-execution boundary. Once marked started, any exception leaves
            # an unknown outcome and the action is never automatically retried.
            try:
                self.execution_control_ledger.mark_dispatch_started(
                    runtime.plan.plan_id, step.step_id, state_fingerprint=state_fp
                )
            except ExecutionControlError as exc:
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id, expected_application_tool=expected_tool,
                    status="failed", error_code=exc.code,
                )
                self._skip_remaining_controlled_steps(runtime, after_step_id=step.step_id, error_code="prior_step_failed")
                break

            emit_executor_protocol_debug(
                executor_protocol_debug_payload(
                    normalized,
                    plan_id=runtime.plan.plan_id,
                    step_id=step.step_id,
                    planned_tool=expected_tool,
                    active_problem_id=current_problem_id,
                    transport=executor_transport,
                    arguments_parsed=True,
                    arguments_schema_valid=True,
                    dispatch_attempted=True,
                    literal_binding_count=intent_result.literal_binding_count,
                    literal_bindings_preserved=intent_result.literal_bindings_preserved,
                    deferred_binding_count=intent_result.deferred_binding_count,
                    deferred_bindings_resolved=intent_result.deferred_bindings_resolved,
                    unexpected_argument_paths=intent_result.unexpected_argument_paths,
                    argument_intent_valid=True,
                )
            )
            try:
                tool_result = dispatch_tool(actual_name, arguments, self.context, runtime.state)
            except Exception:
                self.execution_control_ledger.mark_outcome_unknown(
                    runtime.plan.plan_id, step.step_id, error_code="post_dispatch_exception"
                )
                step.status = "outcome_unknown"
                runtime.plan_status = "manual_intervention_required"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="outcome_unknown",
                    actual_tool=actual_name,
                    error_code="outcome_unknown",
                    outcome_metadata={"dispatch_attempted": True},
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="manual_intervention_required"
                )
                self._finalize_controlled_turn(runtime)
                self._update_controlled_record(runtime)
                return AgentRunResult(
                    text=(
                        f"Manual intervention required for plan {runtime.plan.plan_id}, step {step.step_id}: "
                        "dispatch started, but the application cannot prove whether the action completed. "
                        "It will not retry the action automatically."
                    ),
                    tool_names=tuple(runtime.actual_tool_trace[getattr(runtime, "_result_trace_start", 0):]),
                )

            runtime.actual_tool_trace.append(actual_name)
            safe_result = _plan_safe_value(tool_result)
            safe_mapping = safe_result if isinstance(safe_result, Mapping) else {"result": safe_result}
            runtime.completed_outputs.append(
                {"step_id": step.step_id, "tool": actual_name, "result": dict(safe_mapping)}
            )
            self.execution_control_ledger.mark_dispatch_completed(
                runtime.plan.plan_id,
                step.step_id,
                actual_tool=actual_name,
                receipt_metadata={"tool": actual_name, "result": dict(safe_mapping)},
            )
            if not _tool_result_failed(tool_result):
                _record_recent_session_entities(
                    entity_context=self.session_entity_context,
                    plan_id=runtime.plan.plan_id,
                    step_id=step.step_id,
                    tool_name=actual_name,
                    arguments=arguments,
                    tool_result=tool_result,
                    state=runtime.state,
                )
            if _tool_result_failed(tool_result):
                step.status = "failed"
                runtime.plan_status = "failed"
                runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                    step_id=step.step_id,
                    expected_application_tool=expected_tool,
                    status="failed",
                    actual_tool=actual_name,
                    error_code=_tool_result_error_code(tool_result) or "application_tool_failed",
                    outcome_metadata=dict(safe_mapping),
                )
                self._skip_remaining_controlled_steps(
                    runtime, after_step_id=step.step_id, error_code="prior_step_failed"
                )
                break
            step.status = "completed"
            runtime.step_summaries[step.step_id] = PlanStepExecutionSummary(
                step_id=step.step_id,
                expected_application_tool=expected_tool,
                status="completed",
                actual_tool=actual_name,
                outcome_metadata=dict(safe_mapping),
            )

        if runtime.plan_status == "running":
            runtime.plan_status = "completed" if all(
                step.status in {"completed", "skipped"} for step in runtime.plan.steps
            ) else "failed"
        return self._controlled_terminal_result(runtime)

    def _run_planned_controlled(self, user_message: str) -> AgentRunResult:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message debe ser texto no vacío")
        if not self.is_multi_domain:
            raise RuntimeError("Controlled execution requires multi-domain mode.")

        self.session_entity_context.begin_turn()
        initial_problem_id = self.multi_domain_session.active_problem_id
        try:
            plan = self._request_execution_plan(user_message)
        except PlanValidationError as exc:
            plan_id = f"plan_{self._plan_counter:06d}" if self._plan_counter else self._next_plan_id()
            rejected = ExecutionPlan(
                plan_id=plan_id,
                user_goal=user_message,
                initial_active_problem_id=initial_problem_id,
                steps=[],
            )
            self._record_plan_execution(
                rejected,
                initial_problem_id=initial_problem_id,
                actual_tool_trace=[],
                step_summaries=[],
                validation_error=exc.code,
                plan_status="failed",
            )
            self.session_entity_context.finalize_turn()
            return AgentRunResult(text=f"Execution plan rejected: {exc.code}. {exc.message}", tool_names=())

        runtime = _ControlledPlanRuntime(
            plan=plan,
            user_message=user_message,
            initial_problem_id=initial_problem_id,
            state=self._new_run_state(),
            plan_fingerprint=stable_fingerprint(_plan_contract_payload(plan)),
        )
        self._controlled_plan_runtimes[plan.plan_id] = runtime
        for step in plan.steps:
            self.execution_control_ledger.ensure(plan.plan_id, step.step_id)
        self.last_execution_plan = plan

        # Step 20 safety boundary: the plan is already structurally/semantically validated, but
        # no ApprovalPolicy decision, executor-model call, application dispatch, or domain switch
        # has occurred.  Evaluate the original required condition against the problem targeted by
        # the validated mutating action(s), including simulated select_problem transitions.
        capability_decision = self.requirement_capability_guard.assess(
            user_text=user_message,
            plan=plan,
            registry=self.problem_registry,
            active_problem_id=initial_problem_id,
        )
        self.last_capability_guard_decision = capability_decision
        if capability_decision.blocked or capability_decision.informational:
            return self._apply_capability_guard_decision(runtime, capability_decision)

        return self._execute_controlled_runtime(runtime)

    def _run_planned(self, user_message: str) -> AgentRunResult:
        if self.controlled_execution:
            return self._run_planned_controlled(user_message)
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message debe ser texto no vacío")
        if not self.is_multi_domain:
            raise RuntimeError("Explicit planning is enabled only for multi-domain sessions.")

        self.session_entity_context.begin_turn()
        initial_problem_id = self.multi_domain_session.active_problem_id
        try:
            plan = self._request_execution_plan(user_message)
        except PlanValidationError as exc:
            # Invalid plans never execute an application tool and are not semantically repaired.
            plan_id = f"plan_{self._plan_counter:06d}" if self._plan_counter else self._next_plan_id()
            rejected = ExecutionPlan(
                plan_id=plan_id,
                user_goal=user_message,
                initial_active_problem_id=initial_problem_id,
                steps=[],
            )
            record = self._record_plan_execution(
                rejected,
                initial_problem_id=initial_problem_id,
                actual_tool_trace=[],
                step_summaries=[],
                validation_error=exc.code,
            )
            self.session_entity_context.finalize_turn()
            return AgentRunResult(
                text=f"Execution plan rejected: {exc.code}. {exc.message}",
                tool_names=(),
            )

        self.last_execution_plan = plan
        state = self._new_run_state()
        actual_tool_trace: list[str] = []
        step_summaries: list[PlanStepExecutionSummary] = []
        completed_outputs: list[Dict[str, object]] = []
        failed = False

        for index, step in enumerate(plan.steps):
            if failed:
                step.status = "skipped"
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=step.expected_application_tool,
                        status="skipped",
                        error_code="prior_step_failed",
                    )
                )
                continue

            step.status = "running"
            expected_tool = step.expected_application_tool
            if expected_tool is None:
                # No application action is required for this step. The final tool-free
                # synthesis answers it from application grounding and completed outcomes.
                step.status = "completed"
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=None,
                        status="completed",
                        outcome_metadata={"kind": "no_tool_step"},
                    )
                )
                continue

            current_problem_id = self.multi_domain_session.active_problem_id
            if (
                expected_tool != "select_problem"
                and step.target_problem_id is not None
                and step.target_problem_id != current_problem_id
            ):
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="execution_domain_mismatch",
                        outcome_metadata={
                            "active_problem_id": current_problem_id,
                            "target_problem_id": step.target_problem_id,
                        },
                    )
                )
                continue

            current_tools = self._current_llm_tools()
            tool_schema = next(
                (schema for schema in current_tools if schema.get("name") == expected_tool),
                None,
            )
            if tool_schema is None:
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="planned_tool_unavailable",
                    )
                )
                continue

            constrained_tool_schema, intent_schema_error = _constrain_executor_schema_to_plan(
                step=step,
                tool_schema=tool_schema,
                completed_outputs=completed_outputs,
                session_entity_context=self.session_entity_context,
                active_problem_id=current_problem_id,
            )
            if intent_schema_error is not None or constrained_tool_schema is None:
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        None,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport="not_started",
                        arguments_parsed=None,
                        arguments_schema_valid=None,
                        dispatch_attempted=False,
                        literal_binding_count=sum(
                            1 for binding in step.argument_bindings if binding.binding_type == "literal"
                        ),
                        deferred_binding_count=sum(
                            1 for binding in step.argument_bindings if binding.binding_type == "deferred"
                        ),
                        argument_intent_valid=False,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="argument_intent_violation",
                        outcome_metadata={"message": intent_schema_error or "Unable to constrain executor schema."},
                    )
                )
                continue

            self.system_instructions = self._current_system_instructions()
            executor_payload = {
                "original_user_goal": user_message,
                "plan": plan.to_dict(),
                "current_step": step.to_dict(),
                "active_problem_id": current_problem_id,
                "completed_step_outputs": completed_outputs,
            }
            try:
                executor_transport, executor_options = build_executor_request_options(
                    provider=self.llm_config.provider,
                    tool_schema=constrained_tool_schema,
                )
            except ExecutorTransportUnsupportedError:
                debug_payload = executor_protocol_debug_payload(
                    None,
                    plan_id=plan.plan_id,
                    step_id=step.step_id,
                    planned_tool=expected_tool,
                    active_problem_id=current_problem_id,
                    transport="unsupported",
                    arguments_parsed=None,
                    arguments_schema_valid=None,
                    dispatch_attempted=False,
                )
                emit_executor_protocol_debug(debug_payload)
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="executor_transport_unsupported",
                    )
                )
                continue

            try:
                response = self.client.responses.create(
                    model=self.llm_config.model,
                    instructions=self.system_instructions + _EXECUTOR_PROTOCOL_GROUNDING,
                    input=[
                        {
                            "role": "user",
                            "content": json.dumps(executor_payload, ensure_ascii=False, sort_keys=True),
                        }
                    ],
                    **executor_options,
                )
            except Exception:
                # No application tool has been dispatched for this step. Treat provider failure
                # as a normal failed Step 17 execution step; do not retry or replan.
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="executor_provider_error",
                        outcome_metadata={"dispatch_attempted": False},
                    )
                )
                continue
            normalized = normalize_executor_response(response, transport=executor_transport)
            calls = normalized.tool_calls
            if len(calls) != 1:
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        normalized,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport=executor_transport,
                        arguments_parsed=None,
                        arguments_schema_valid=None,
                        dispatch_attempted=False,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        error_code="execution_protocol_error",
                        outcome_metadata={"attempted_call_count": len(calls)},
                    )
                )
                continue

            call = calls[0]
            actual_name = call.name
            if actual_name != expected_tool:
                # Never dispatch an out-of-plan tool call, even if a provider violates named forcing.
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        normalized,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport=executor_transport,
                        arguments_parsed=None,
                        arguments_schema_valid=None,
                        dispatch_attempted=False,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        actual_tool=actual_name or None,
                        error_code="unplanned_tool_call",
                        outcome_metadata={"attempted_tool": actual_name},
                    )
                )
                continue

            try:
                arguments = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        normalized,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport=executor_transport,
                        arguments_parsed=False,
                        arguments_schema_valid=None,
                        dispatch_attempted=False,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        actual_tool=actual_name,
                        error_code="execution_protocol_error",
                        outcome_metadata={"arguments_parsed": False},
                    )
                )
                continue

            intent_result = _validate_argument_intent(
                step=step,
                tool_name=expected_tool,
                arguments=arguments,
                completed_outputs=completed_outputs,
                session_entity_context=self.session_entity_context,
                active_problem_id=current_problem_id,
            )
            if not intent_result.valid:
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        normalized,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport=executor_transport,
                        arguments_parsed=True,
                        arguments_schema_valid=None,
                        dispatch_attempted=False,
                        literal_binding_count=intent_result.literal_binding_count,
                        literal_bindings_preserved=intent_result.literal_bindings_preserved,
                        deferred_binding_count=intent_result.deferred_binding_count,
                        deferred_bindings_resolved=intent_result.deferred_bindings_resolved,
                        unexpected_argument_paths=intent_result.unexpected_argument_paths,
                        argument_intent_valid=False,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        actual_tool=actual_name,
                        error_code="argument_intent_violation",
                        outcome_metadata={
                            "message": intent_result.message,
                            "unexpected_argument_paths": list(intent_result.unexpected_argument_paths),
                        },
                    )
                )
                continue

            argument_error = _validate_executor_arguments_against_current_tool(
                tool_name=expected_tool,
                arguments=arguments,
                tool_schema=constrained_tool_schema,
                state=state,
            )
            if argument_error is not None:
                emit_executor_protocol_debug(
                    executor_protocol_debug_payload(
                        normalized,
                        plan_id=plan.plan_id,
                        step_id=step.step_id,
                        planned_tool=expected_tool,
                        active_problem_id=current_problem_id,
                        transport=executor_transport,
                        arguments_parsed=True,
                        arguments_schema_valid=False,
                        dispatch_attempted=False,
                        literal_binding_count=intent_result.literal_binding_count,
                        literal_bindings_preserved=intent_result.literal_bindings_preserved,
                        deferred_binding_count=intent_result.deferred_binding_count,
                        deferred_bindings_resolved=intent_result.deferred_bindings_resolved,
                        unexpected_argument_paths=intent_result.unexpected_argument_paths,
                        argument_intent_valid=True,
                    )
                )
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        actual_tool=actual_name,
                        error_code="invalid_tool_arguments",
                        outcome_metadata={"message": argument_error},
                    )
                )
                continue

            if expected_tool == "select_problem":
                if not isinstance(arguments, Mapping) or arguments.get("problem_id") != step.target_problem_id:
                    emit_executor_protocol_debug(
                        executor_protocol_debug_payload(
                            normalized,
                            plan_id=plan.plan_id,
                            step_id=step.step_id,
                            planned_tool=expected_tool,
                            active_problem_id=current_problem_id,
                            transport=executor_transport,
                            arguments_parsed=True,
                            arguments_schema_valid=True,
                            dispatch_attempted=False,
                        )
                    )
                    step.status = "failed"
                    failed = True
                    step_summaries.append(
                        PlanStepExecutionSummary(
                            step_id=step.step_id,
                            expected_application_tool=expected_tool,
                            status="failed",
                            actual_tool=actual_name,
                            error_code="selection_target_mismatch",
                            outcome_metadata={
                                "planned_target_problem_id": step.target_problem_id,
                                "attempted_problem_id": (
                                    arguments.get("problem_id") if isinstance(arguments, Mapping) else None
                                ),
                            },
                        )
                    )
                    continue

            emit_executor_protocol_debug(
                executor_protocol_debug_payload(
                    normalized,
                    plan_id=plan.plan_id,
                    step_id=step.step_id,
                    planned_tool=expected_tool,
                    active_problem_id=current_problem_id,
                    transport=executor_transport,
                    arguments_parsed=True,
                    arguments_schema_valid=True,
                    dispatch_attempted=True,
                    literal_binding_count=intent_result.literal_binding_count,
                    literal_bindings_preserved=intent_result.literal_bindings_preserved,
                    deferred_binding_count=intent_result.deferred_binding_count,
                    deferred_bindings_resolved=intent_result.deferred_bindings_resolved,
                    unexpected_argument_paths=intent_result.unexpected_argument_paths,
                    argument_intent_valid=True,
                )
            )
            tool_result = dispatch_tool(actual_name, arguments, self.context, state)
            actual_tool_trace.append(actual_name)
            safe_result = _plan_safe_value(tool_result)
            safe_mapping = safe_result if isinstance(safe_result, Mapping) else {"result": safe_result}
            completed_outputs.append(
                {
                    "step_id": step.step_id,
                    "tool": actual_name,
                    "result": dict(safe_mapping),
                }
            )
            if not _tool_result_failed(tool_result):
                _record_recent_session_entities(
                    entity_context=self.session_entity_context,
                    plan_id=plan.plan_id,
                    step_id=step.step_id,
                    tool_name=actual_name,
                    arguments=arguments,
                    tool_result=tool_result,
                    state=state,
                )
            if _tool_result_failed(tool_result):
                step.status = "failed"
                failed = True
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="failed",
                        actual_tool=actual_name,
                        error_code=_tool_result_error_code(tool_result) or "application_tool_failed",
                        outcome_metadata=dict(safe_mapping),
                    )
                )
            else:
                step.status = "completed"
                step_summaries.append(
                    PlanStepExecutionSummary(
                        step_id=step.step_id,
                        expected_application_tool=expected_tool,
                        status="completed",
                        actual_tool=actual_name,
                        outcome_metadata=dict(safe_mapping),
                    )
                )

        self.session_entity_context.finalize_turn()
        record = self._record_plan_execution(
            plan,
            initial_problem_id=initial_problem_id,
            actual_tool_trace=actual_tool_trace,
            step_summaries=step_summaries,
        )
        try:
            text = self._synthesize_plan_result(
                user_message=user_message,
                plan=plan,
                record=record,
            )
        except FinalSynthesisProviderError:
            # Application execution is already complete. Never retry or re-dispatch actions merely
            # because the provider failed during the tool-free synthesis phase. Preserve the
            # completed execution record and return deterministic safe execution metadata instead.
            record = self._mark_final_synthesis_error(record)
            text = self._final_synthesis_fallback(record)
        return AgentRunResult(text=text, tool_names=tuple(actual_tool_trace))

    def _run_reactive(self, user_message: str) -> AgentRunResult:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message debe ser texto no vacío")

        input_items: list[Any] = [{"role": "user", "content": user_message}]
        state = self._new_run_state()
        called_tools: list[str] = []
        tool_iterations = 0

        while True:
            # Multi-domain selection can change during this same request. Refresh both
            # the detailed active-domain grounding and parameter schemas every model turn.
            self.system_instructions = self._current_system_instructions()
            self.llm_tools = self._current_llm_tools()
            response = self.client.responses.create(
                model=self.llm_config.model,
                instructions=self.system_instructions,
                input=input_items,
                tools=self.llm_tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )

            response_output = list(getattr(response, "output", []) or [])
            input_items.extend(response_output)
            function_calls = [
                item
                for item in response_output
                if getattr(item, "type", None) == "function_call"
            ]

            if not function_calls:
                text = getattr(response, "output_text", "") or ""
                return AgentRunResult(text=str(text), tool_names=tuple(called_tools))

            if tool_iterations >= self.config.max_tool_iterations:
                raise RuntimeError(
                    "Maximum tool-call iterations exceeded "
                    f"({self.config.max_tool_iterations})."
                )
            tool_iterations += 1

            for tool_call in function_calls:
                name = str(getattr(tool_call, "name", ""))
                raw_arguments = getattr(tool_call, "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, json.JSONDecodeError):
                    tool_result: Dict[str, object] = {
                        "ok": False,
                        "error": {
                            "code": "invalid_json_arguments",
                            "message": "Tool arguments were not valid JSON.",
                        },
                    }
                else:
                    tool_result = dispatch_tool(name, arguments, self.context, state)

                called_tools.append(name)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(getattr(tool_call, "call_id", "")),
                        "output": json.dumps(tool_result, ensure_ascii=False),
                    }
                )


class BonsaiDecisionAgentV0(DecisionAgentV0):
    """Backward-compatible Bonsai construction path over the generic single-domain core."""

    def __init__(
        self,
        context: BonsaiAgentContext,
        *,
        config: AgentConfig | None = None,
        client: _ModelClient | None = None,
        solution_store: SolutionStore | None = None,
        scenario_store: ScenarioStore | None = None,
        run_history_store: RunHistoryStore | None = None,
        problem_adapter: OptimizationProblemAdapter | None = None,
    ) -> None:
        super().__init__(
            context,
            problem_adapter=problem_adapter or BonsaiProblemAdapter(context),
            config=config,
            client=client,
            solution_store=solution_store,
            scenario_store=scenario_store,
            run_history_store=run_history_store,
        )

__all__ = [
    "AgentConfig",
    "AgentRunResult",
    "LLMProviderConfig",
    "BonsaiAgentContext",
    "DecisionAgentV0",
    "BonsaiDecisionAgentV0",
    "OptimizationProblemAdapter",
    "BonsaiProblemAdapter",
    "LLM_TOOLS",
    "SYSTEM_INSTRUCTIONS",
    "build_system_instructions",
    "TOOL_REGISTRY",
    "NVIDIA_NIM_BASE_URL",
    "GOLDEN_REFERENCE",
    "LATEST_OPTIMIZER",
    "SolutionStore",
    "ScenarioStore",
    "RunHistoryStore",
    "create_llm_client",
    "create_openai_client",
    "dispatch_tool",
]
