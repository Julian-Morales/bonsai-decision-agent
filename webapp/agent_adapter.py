"""Thin UI adapter around the frozen Step 20 Decision Agent core."""
from __future__ import annotations

import threading
from typing import Callable, Dict, Mapping

from bonsai_agent import AgentConfig, AgentRunResult, LLMProviderConfig
from run_decision_agent import build_multi_domain_agent
from problem_adapter import business_contexts_compatible
from webapp.connections import core_provider_name, default_openai_client_factory, effective_base_url
from webapp.storage import ConnectionProfile, ConnectionProfileStore, StorageError
from webapp.ui_mapping import decision_view, developer_details, workflow_cards, rejected_domain_transition_card, uncertain_domain_transition_card
from webapp.presentation import presentation


class _UnconfiguredResponses:
    def create(self, **_kwargs):
        raise RuntimeError("No active provider connection is configured.")


class _UnconfiguredClient:
    responses = _UnconfiguredResponses()


def _safe_text(exc: BaseException, secrets: tuple[str, ...] = ()) -> str:
    text = str(exc)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class DecisionAgentUIAdapter:
    """Own one in-memory core session and expose presentation-safe projections."""

    def __init__(
        self,
        profile_store: ConnectionProfileStore,
        *,
        client_factory: Callable[[ConnectionProfile, str], object] | None = None,
        agent_factory: Callable[..., object] | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.client_factory = client_factory or default_openai_client_factory
        self._agent_factory = agent_factory or build_multi_domain_agent
        placeholder = AgentConfig(llm=LLMProviderConfig(provider="openai", model="ux1-unconfigured"))
        self.agent = self._agent_factory(config=placeholder, client=_UnconfiguredClient())
        self._lock = threading.RLock()

    def _configure_for_active_profile(self) -> ConnectionProfile:
        profile = self.profile_store.active_profile()
        if profile is None:
            raise StorageError("No active connection profile is configured")
        if not profile.model.strip():
            raise StorageError("The active connection profile does not have a selected model")
        api_key = self.profile_store.get_secret(profile.id)
        client = self.client_factory(profile, api_key)
        llm = LLMProviderConfig(
            provider=core_provider_name(profile),
            model=profile.model,
            base_url=effective_base_url(profile),
        )
        config = AgentConfig(llm=llm)
        # Application-level runtime configuration only. No frozen core source or behavior is changed.
        self.agent.client = client
        self.agent.llm_config = llm
        self.agent.config = config
        return profile

    def problems(self) -> list[Dict[str, object]]:
        registry = self.agent.problem_registry
        session = self.agent.multi_domain_session
        return [
            {
                **metadata.to_dict(),
                "active": metadata.problem_id == session.active_problem_id,
                "configurable_parameters": [
                    {
                        "name": name,
                        "category": definition.category,
                        "value_type": definition.value_type.__name__,
                        "default": definition.default,
                        "allowed_values": list(definition.allowed_values) if definition.allowed_values is not None else None,
                        "minimum": definition.minimum,
                        "maximum": definition.maximum,
                        "description": definition.description,
                    }
                    for name, definition in registry.get(metadata.problem_id).problem.parameters.items()
                    if definition.configurable
                ],
            }
            for metadata in registry.metadata()
        ]

    def select_problem(self, problem_id: str) -> Dict[str, object]:
        with self._lock:
            registry = self.agent.problem_registry
            # ProblemRegistry owns the allowlist check; MultiDomainSessionState owns selection.
            metadata = next((m for m in registry.metadata() if m.problem_id == problem_id), None)
            registry.get(problem_id)
            previous = self.agent.multi_domain_session.active_problem_id
            state = self.agent.multi_domain_session.select(problem_id)
            return {
                "success": True,
                "previous_problem_id": previous,
                "active_problem_id": state.adapter.problem_id,
                "problem": metadata.to_dict() if metadata is not None else {"problem_id": problem_id},
            }

    def scenarios(self) -> list[Dict[str, object]]:
        values: list[Dict[str, object]] = []
        for problem_id in self.agent.problem_registry.problem_ids:
            state = self.agent.multi_domain_session.state_for(problem_id)
            for record in state.scenario_store.list():
                values.append(
                    {
                        "problem_id": problem_id,
                        "handle": record.definition.handle,
                        "name": record.definition.name,
                        "parameter_overrides": dict(record.definition.parameter_overrides),
                        "solution_handle": record.solution_handle,
                        "optimized": record.solution_handle is not None,
                    }
                )
        return values

    def scenario_detail(self, problem_id: str, handle: str) -> Dict[str, object]:
        """Read-only scenario inspection; never resolves or executes an optimization."""
        with self._lock:
            state = self.agent.multi_domain_session.state_for(problem_id)
            record = state.scenario_store.get(handle)
            if record.problem_id != problem_id:
                raise ValueError("Scenario provenance does not match requested problem")
            # Association is authoritative only where the core stored the explicit handle.
            associated_runs = [
                {"run_id": run.run_id, "status": run.status, "solution_handle": run.solution_handle}
                for run in state.run_history_store.list()
                if run.scenario_handle == record.definition.handle
            ]
            return {"type": "scenario_detail", "title": "Scenario", "data": {
                "problem_id": record.problem_id,
                "handle": record.definition.handle,
                "name": record.definition.name,
                "parameter_overrides": dict(record.definition.parameter_overrides),
                "solution_handle": record.solution_handle,
                "associated_runs": associated_runs,
            }}

    def runs(self) -> list[Dict[str, object]]:
        values: list[Dict[str, object]] = []
        for problem_id in self.agent.problem_registry.problem_ids:
            state = self.agent.multi_domain_session.state_for(problem_id)
            for record in state.run_history_store.list():
                values.append(
                    {
                        "run_id": record.run_id,
                        "problem_id": record.problem_id,
                        "run_kind": record.run_kind,
                        "status": record.status,
                        "fast": record.fast,
                        "parameter_overrides": record.overrides,
                        "scenario_handle": record.scenario_handle,
                        "solution_handle": record.solution_handle,
                        "current_alias": record.current_alias,
                        "error_code": record.error_code,
                    }
                )
        return values


    def assert_run_exists(self, problem_id: str, run_id: str) -> None:
        """Membership check for presentation metadata only; no execution or mutation."""
        with self._lock:
            state = self.agent.multi_domain_session.state_for(problem_id)
            if run_id not in state.run_history_store:
                raise ValueError(f"Run is not registered for problem {problem_id}: {run_id}")

    def historical_run(self, problem_id: str, run_id: str) -> Dict[str, object]:
        """Return a read-only deterministic projection of one immutable historical run.

        This path deliberately does not call ``agent.run``, request a plan, dispatch an
        application tool, select a problem, or alter any session store. Complete solution
        rows are read from the existing immutable run_solution_* handle and passed directly
        to the domain's independent deterministic validator.
        """
        with self._lock:
            session = self.agent.multi_domain_session
            state = session.state_for(problem_id)
            try:
                record = state.run_history_store.get(run_id)
            except Exception as exc:
                raise ValueError(f"Run is not registered for problem {problem_id}: {run_id}") from exc

            metadata = next(
                (item for item in self.agent.problem_registry.metadata() if item.problem_id == problem_id),
                None,
            )
            data: Dict[str, object] = {
                "run_id": record.run_id,
                "problem_id": record.problem_id,
                "problem_display_name": metadata.display_name if metadata is not None else record.problem_id,
                "run_kind": record.run_kind,
                "status": record.status,
                "fast": record.fast,
                "parameter_overrides": record.overrides,
                "scenario_handle": record.scenario_handle,
                "solution_handle": record.solution_handle,
                "current_alias": record.current_alias,
                "error_code": record.error_code,
            }

            if record.status != "completed" or not record.solution_handle:
                return {
                    "type": "historical_run_result",
                    "title": "Historical run",
                    "historical": True,
                    "data": data,
                }

            rows = state.solution_store.get(record.solution_handle)
            stored_problem_id = state.solution_store.get_problem_id(record.solution_handle)
            if stored_problem_id != record.problem_id:
                raise ValueError("Historical run solution provenance does not match run metadata")
            business_context = state.solution_store.get_business_context(record.solution_handle)
            if business_context is None:
                raise ValueError("Historical run is missing its deterministic business context")

            report = state.adapter.validate_solution(rows, business_context=business_context)
            validated = dict(state.adapter.compact_validation_result(record.solution_handle, report))
            validated.pop("success", None)
            validated.pop("handle", None)
            validated["independently_validated"] = bool(validated.get("valid"))

            reference_context = state.adapter.problem.business_parameters
            reference_compatible = business_contexts_compatible(business_context, reference_context)
            baseline = state.adapter.get_baseline() if reference_compatible else None
            if not reference_compatible:
                # Match the live optimization projection: do not present default-context
                # savings beside a result evaluated under a different business context.
                if "saving" in validated:
                    validated["saving"] = None
                if "saving_pct" in validated:
                    validated["saving_pct"] = None

            data.update(
                {
                    "reference_context_compatible": reference_compatible,
                    "baseline": baseline,
                    "effective_business_context": dict(business_context),
                    "reference_business_context": dict(reference_context),
                    "validated_final": validated,
                }
            )
            return {
                "type": "historical_run_result",
                "title": "Historical run",
                "historical": True,
                "data": data,
            }

    def historical_run_by_id(self, run_id: str) -> Dict[str, object]:
        """Resolve an unscoped run id only when it is unique across problem stores."""
        with self._lock:
            matches = []
            for problem_id in self.agent.problem_registry.problem_ids:
                store = self.agent.multi_domain_session.state_for(problem_id).run_history_store
                if run_id in store:
                    matches.append(problem_id)
            if not matches:
                raise ValueError(f"Run is not registered: {run_id}")
            if len(matches) > 1:
                raise ValueError(
                    f"Run id {run_id} exists in more than one problem; use the problem-scoped run endpoint"
                )
            return self.historical_run(matches[0], run_id)

    def _response(self, result: AgentRunResult, *, prior_record: object | None = None, check_stale: bool = False) -> Dict[str, object]:
        record = getattr(self.agent, "last_plan_execution_record", None)
        approval = getattr(self.agent, "last_approval_record", None)
        capability = getattr(self.agent, "last_capability_guard_decision", None)
        if check_stale and record is prior_record:
            # An informational response may not update last_plan_execution_record;
            # do not replay stale workflow, result, approval, or capability metadata.
            record = None
            approval = None
            capability = None
        cards = workflow_cards(record=record, approval=approval, capability_decision=capability)
        transition = rejected_domain_transition_card(
            record=record,
            result_text=result.text,
            active_problem_id=self.agent.multi_domain_session.active_problem_id,
            registered_problem_ids=self.agent.problem_registry.problem_ids,
        )
        if transition is not None:
            cards.append(transition)
        else:
            uncertain = uncertain_domain_transition_card(
                record=record,
                active_problem_id=self.agent.multi_domain_session.active_problem_id,
            )
            if uncertain is not None:
                cards.append(uncertain)
        view = decision_view(record=record, capability_decision=capability, approval=approval, transition=transition)
        if view.get("type") in {"optimization", "scenario"} and record is not None:
            # Domain identity from the execution record, not current browser selection.
            view = {**view, "data": {**view.get("data", {}), "problem_id": record.final_problem_id}}
        record_data = record.to_dict() if record is not None else None
        if (record_data and record_data.get("succeeded") is True
                and record_data.get("final_synthesis_error") == "final_synthesis_provider_error"
                and record_data.get("actual_tool_trace")
                and view.get("type") in {"optimization", "scenario", "validation", "comparison", "baseline"}):
            view = {**view, "synthesis_warning": "Final explanation could not be generated."}
        return {
            "message": {"role": "assistant", "text": result.text},
            "presentation": presentation(view, cards, record=record_data),
            "tool_names": list(result.tool_names),
            "workflow_cards": cards,
            "decision_view": view,
            "developer_details": developer_details(self.agent) if record is not None else
                {"plan": None, "execution_record": None, "capability_guard": None, "approval": None, "execution_ledger": []},
            "workspace": {
                "problems": self.problems(),
                "scenarios": self.scenarios(),
                "runs": self.runs(),
            },
        }

    def chat(self, message: str) -> Dict[str, object]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty text")
        with self._lock:
            secret = ""
            try:
                active = self.profile_store.active_profile()
                if active is not None and self.profile_store.has_key(active.id):
                    secret = self.profile_store.get_secret(active.id)
                self._configure_for_active_profile()
                prior_record = getattr(self.agent, "last_plan_execution_record", None)
                result = self.agent.run(message.strip())
            except Exception as exc:
                return {
                    "message": {"role": "assistant", "text": "The agent could not complete this request."},
                    "tool_names": [],
                    "workflow_cards": [
                        {
                            "type": "error",
                            "title": "Request failed",
                            "error_code": "agent_request_failed",
                            "detail": _safe_text(exc, (secret,)),
                        }
                    ],
                    "decision_view": {"type": "execution_failure", "title": "Request issue", "data": {
                        "error_code": "agent_request_failed", "dispatch_state": "not_requested",
                        "safe_message": "The request could not be completed by the local agent."}},
                    "presentation": {"kind": "execution_failure", "summary": "The request could not be completed.", "show_full_response": True},
                    "developer_details": {"plan": None, "execution_record": None,
                                          "capability_guard": None, "approval": None, "execution_ledger": []},
                    "workspace": {
                        "problems": self.problems(),
                        "scenarios": self.scenarios(),
                        "runs": self.runs(),
                    },
                }
            return self._response(result, prior_record=prior_record, check_stale=True)

    def approve(self, approval_id: str | None) -> Dict[str, object]:
        with self._lock:
            result = self.agent.approve(approval_id)
            return self._response(result)

    def reject(self, approval_id: str | None) -> Dict[str, object]:
        with self._lock:
            result = self.agent.reject(approval_id)
            return self._response(result)
