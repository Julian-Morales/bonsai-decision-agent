"""Read-only provider diagnostics for locally configured OpenAI-compatible clients.

Diagnostics do not activate profiles, alter credentials, or enter the agent session.
The same client factory and endpoint defaults are used by DecisionAgentUIAdapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from webapp.storage import ConnectionProfile


NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DISCOVERY_UNAVAILABLE_STATUS = {404, 405, 501}


class ConnectionRuntimeError(RuntimeError):
    """Safe public connection-layer failure."""


@dataclass(frozen=True)
class ConnectionTestResult:
    ok: bool
    state: str
    code: str
    message: str
    model_count: int | None = None
    connected: bool | None = None
    authenticated: bool | None = None
    model_discovery: str = "not_run"
    model_test: str = "not_run"
    inference_verified: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "state": self.state,
            "code": self.code,
            "message": self.message,
            "model_count": self.model_count,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "model_discovery": self.model_discovery,
            "model_test": self.model_test,
            "inference_verified": self.inference_verified,
        }


def effective_base_url(profile: ConnectionProfile) -> str | None:
    if profile.provider == "nvidia":
        return profile.base_url or NVIDIA_NIM_BASE_URL
    return profile.base_url


def core_provider_name(profile: ConnectionProfile) -> str:
    # Custom OpenAI-compatible profiles reuse the frozen OpenAI transport.
    return "nvidia" if profile.provider == "nvidia" else "openai"


def default_openai_client_factory(profile: ConnectionProfile, api_key: str):
    try:
        from openai import OpenAI
    except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
        raise ConnectionRuntimeError(
            "OpenAI SDK is not installed. Install requirements-agent.txt before connecting."
        ) from exc
    kwargs: Dict[str, object] = {"api_key": api_key}
    base_url = effective_base_url(profile)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _iter_model_ids(response: object) -> Iterable[str]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if data is None:
        try:
            data = list(response)  # type: ignore[arg-type]
        except TypeError:
            data = []
    for item in data or []:
        model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if isinstance(model_id, str) and model_id:
            yield model_id


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if type(value) is int else None


def _model_error_code(exc: BaseException) -> str:
    # Consume only structured error identifiers; never inspect/return provider messages.
    code = getattr(exc, "code", None)
    if code is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                code = error.get("code")
    return str(code).casefold() if code is not None else ""


def _map_connection_exception(exc: BaseException, *, discovery_request: bool = False, inference_request: bool = False) -> ConnectionTestResult:
    name = type(exc).__name__.casefold()
    status_code = _status_code(exc)
    discovery = "failed" if discovery_request else "not_run"
    inference = "failed" if inference_request else "not_run"
    if status_code in {401, 403} or "authentication" in name or "permission" in name:
        return ConnectionTestResult(False, "authentication_failed", "authentication_failed", "Authentication failed.",
                                    connected=True, authenticated=False, model_discovery=discovery, model_test=inference)
    if discovery_request and status_code in _DISCOVERY_UNAVAILABLE_STATUS:
        return ConnectionTestResult(True, "discovery_unavailable", "model_discovery_unavailable",
                                    "Endpoint reachable; model discovery unavailable. Authentication not verified. Enter a model manually.",
                                    connected=True, authenticated=None, model_discovery="unavailable")
    if inference_request and _model_error_code(exc) in {"model_not_found", "model_not_available", "invalid_model", "model_unavailable"}:
        return ConnectionTestResult(False, "model_unavailable", "model_unavailable", "Selected model is unavailable.",
                                    connected=True, authenticated=None, model_test="failed")
    if "timeout" in name or "connection" in name or "connect" in name:
        return ConnectionTestResult(False, "endpoint_unreachable", "endpoint_failure",
                                    "Provider endpoint timed out." if "timeout" in name else "Could not reach the provider endpoint.",
                                    connected=False, authenticated=None, model_discovery=discovery, model_test=inference)
    if isinstance(exc, ConnectionRuntimeError):
        return ConnectionTestResult(False, "client_unavailable", "client_unavailable", str(exc),
                                    connected=False, authenticated=None, model_discovery=discovery, model_test=inference)
    if inference_request:
        # 404/405 from the inference route do not prove that the endpoint is unreachable,
        # nor that the model does not exist. Keep that failure separate from connectivity.
        return ConnectionTestResult(False, "inference_failed", "inference_test_failed",
                                    "Inference test failed; endpoint availability may still be valid.",
                                    connected=True if status_code is not None else None,
                                    authenticated=None, model_test="failed")
    if status_code is not None:
        return ConnectionTestResult(False, "endpoint_error", "endpoint_failure",
                                    "Provider endpoint returned an error.",
                                    connected=True, authenticated=None, model_discovery=discovery)
    return ConnectionTestResult(False, "provider_error", "provider_error", "Provider diagnostic failed.",
                                connected=None, authenticated=None, model_discovery=discovery, model_test=inference)


class ConnectionService:
    """Provider-only diagnostics, separate from persistent profile and active session state."""

    def __init__(self, client_factory: Callable[[ConnectionProfile, str], object] | None = None) -> None:
        self.client_factory = client_factory or default_openai_client_factory
        self._last_tests: Dict[str, ConnectionTestResult] = {}
        self._last_discoveries: Dict[str, ConnectionTestResult] = {}
        self._latest_diagnostics: Dict[str, ConnectionTestResult] = {}

    def state_for(self, profile_id: str, *, configured: bool) -> str:
        # Legacy diagnostic field; MUST NOT drive active-profile indicators.
        latest = self._latest_diagnostics.get(profile_id)
        return latest.state if latest else ("configured" if configured else "missing_key")

    def last_test_for(self, profile_id: str) -> dict[str, object] | None:
        result = self._last_tests.get(profile_id)
        return result.to_dict() if result else None

    def latest_diagnostic_for(self, profile_id: str) -> dict[str, object] | None:
        result = self._latest_diagnostics.get(profile_id)
        return result.to_dict() if result else None

    def _record(self, profile_id: str, result: ConnectionTestResult, *, test: bool) -> ConnectionTestResult:
        if test:
            self._last_tests[profile_id] = result
        else:
            self._last_discoveries[profile_id] = result
        self._latest_diagnostics[profile_id] = result
        return result

    def test(self, profile: ConnectionProfile, api_key: str) -> ConnectionTestResult:
        """Discover models without a selection; with a model, test the real Responses transport.

        No DecisionAgentV0, planner, or application tool is ever invoked. The discovery
        route is optional: its absence does not prevent a selected-model inference probe.
        """
        try:
            client = self.client_factory(profile, api_key)
        except Exception as exc:
            return self._record(profile.id, _map_connection_exception(exc), test=True)

        discovery_failure = None
        ids: tuple[str, ...] = ()
        try:
            models = getattr(client, "models")
            ids = tuple(_iter_model_ids(models.list()))
        except Exception as exc:
            discovery_failure = _map_connection_exception(exc, discovery_request=True)

        selected_model = profile.model.strip()
        if not selected_model:
            if discovery_failure is not None:
                return self._record(profile.id, discovery_failure, test=True)
            return self._record(profile.id, ConnectionTestResult(
                True, "connected", "connected_authenticated",
                "Endpoint reachable; authentication and model discovery verified. Inference not tested.",
                model_count=len(ids), connected=True, authenticated=True, model_discovery="available",
            ), test=True)

        # Use the same Responses transport the frozen DecisionAgentV0 uses. In
        # particular, do NOT require the optional /models/{id} metadata route:
        # NVIDIA NIM can support /models and /responses but not models.retrieve.
        responses = getattr(client, "responses", None)
        create = getattr(responses, "create", None)
        if callable(create):
            try:
                probe = create(model=selected_model,
                               input=[{"role": "user", "content": "Reply OK."}],
                               max_output_tokens=8)
                if probe is None or getattr(probe, "status", None) in {"failed", "cancelled"}:
                    raise RuntimeError("Provider did not return a successful inference result.")
            except Exception as exc:
                mapped = _map_connection_exception(exc, inference_request=True)
                # Successful discovery proves the host/auth were reachable. A separate
                # inference-route network failure cannot undo that evidence.
                if discovery_failure is None and mapped.code == "endpoint_failure":
                    mapped = ConnectionTestResult(False, "inference_failed", "inference_test_failed",
                                                  "Inference route unavailable; discovery endpoint remains reachable.",
                                                  connected=True, authenticated=True, model_test="failed")
                result = ConnectionTestResult(
                    False, mapped.state, mapped.code, mapped.message,
                    model_count=len(ids) if discovery_failure is None else None,
                    connected=True if discovery_failure is None else mapped.connected,
                    authenticated=True if discovery_failure is None and mapped.code != "authentication_failed" else mapped.authenticated,
                    model_discovery="available" if discovery_failure is None else discovery_failure.model_discovery,
                    model_test="failed",
                )
                return self._record(profile.id, result, test=True)
            return self._record(profile.id, ConnectionTestResult(
                True, "inference_verified", "model_test_successful",
                "Inference verified through the configured provider and model.",
                model_count=len(ids) if discovery_failure is None else None,
                connected=True, authenticated=True,
                model_discovery="available" if discovery_failure is None else discovery_failure.model_discovery,
                model_test="successful", inference_verified=True,
            ), test=True)

        # Compatibility with metadata-only test clients from UX-1.1: do not
        # present metadata retrieval as proof of inference on a real provider.
        if discovery_failure is not None:
            return self._record(profile.id, discovery_failure, test=True)
        retrieve = getattr(models, "retrieve", None)
        if callable(retrieve):
            try:
                retrieve(selected_model)
            except Exception as exc:
                mapped = _map_connection_exception(exc, inference_request=True)
                return self._record(profile.id, ConnectionTestResult(
                    False, mapped.state, mapped.code, mapped.message, model_count=len(ids),
                    connected=True, authenticated=mapped.authenticated if mapped.code == "authentication_failed" else True,
                    model_discovery="available", model_test="failed"), test=True)
            return self._record(profile.id, ConnectionTestResult(
                True, "connected", "model_test_successful", "Model metadata verified; inference not tested.",
                model_count=len(ids), connected=True, authenticated=True,
                model_discovery="available", model_test="successful", inference_verified=False), test=True)
        return self._record(profile.id, ConnectionTestResult(
            True, "connected", "connected_authenticated",
            "Endpoint reachable; authentication and discovery verified. Inference unavailable.",
            model_count=len(ids), connected=True, authenticated=True,
            model_discovery="available", model_test="unavailable"), test=True)

    def fetch_models(self, profile: ConnectionProfile, api_key: str) -> Dict[str, object]:
        """Discovery-only: model selection is never required, inference is never claimed."""
        try:
            client = self.client_factory(profile, api_key)
            response = getattr(client, "models").list()
            ids = sorted(dict.fromkeys(_iter_model_ids(response)))[:100]
        except Exception as exc:
            mapped = _map_connection_exception(exc, discovery_request=True)
            self._record(profile.id, mapped, test=False)
            return {**mapped.to_dict(), "models": []}
        result = ConnectionTestResult(
            True, "connected", "connected_authenticated",
            "Endpoint reachable; authentication and model discovery verified. Inference not tested.",
            model_count=len(ids), connected=True, authenticated=True,
            model_discovery="available", model_test="not_run",
        )
        self._record(profile.id, result, test=False)
        return {**result.to_dict(), "models": ids}
