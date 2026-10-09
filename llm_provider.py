"""Provider-facing helpers for structured private planning transport.

This module owns serialization details for the Step 17 private planner protocol so the
DecisionAgent does not need provider-specific request/response handling. Application tools
remain separate from this internal transport.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FORCED_FUNCTION_TRANSPORT = "forced_function"
STRICT_JSON_TRANSPORT = "strict_json_schema"


class PlannerTransportUnsupportedError(RuntimeError):
    """Raised when a provider has no supported structured planning transport."""


class ExecutorTransportUnsupportedError(RuntimeError):
    """Raised when a provider cannot force a planned application function call."""


@dataclass(frozen=True)
class PlannerProviderCapabilities:
    forced_named_function_calls: bool
    strict_json_schema_output: bool = False


@dataclass(frozen=True)
class NormalizedFunctionCall:
    name: str
    arguments: str
    call_id: str = ""


@dataclass(frozen=True)
class NormalizedPlannerResponse:
    finish_reason: str | None
    text_present: bool
    tool_calls: tuple[NormalizedFunctionCall, ...]
    transport: str


@dataclass(frozen=True)
class NormalizedExecutorResponse:
    finish_reason: str | None
    text_present: bool
    tool_calls: tuple[NormalizedFunctionCall, ...]
    transport: str


_PROVIDER_CAPABILITIES: dict[str, PlannerProviderCapabilities] = {
    # Both supported providers expose an OpenAI-compatible Responses endpoint with named
    # function selection. Keep the capability explicit so future providers fail closed.
    "openai": PlannerProviderCapabilities(forced_named_function_calls=True, strict_json_schema_output=True),
    "nvidia": PlannerProviderCapabilities(forced_named_function_calls=True, strict_json_schema_output=False),
}


def planner_provider_capabilities(provider: str) -> PlannerProviderCapabilities:
    capabilities = _PROVIDER_CAPABILITIES.get(str(provider).strip().lower())
    if capabilities is None:
        raise PlannerTransportUnsupportedError(
            f"No structured planner transport is configured for provider {provider!r}."
        )
    return capabilities


def build_planner_request_options(
    *,
    provider: str,
    submit_plan_schema: Mapping[str, object],
    capabilities: PlannerProviderCapabilities | None = None,
) -> tuple[str, dict[str, object]]:
    """Build provider-facing request options for exactly one structured planning action.

    Prefer a forced named function call. A strict JSON-schema transport is available as a
    serialization fallback for providers that explicitly advertise it. Both transports carry
    the same ``submit_plan`` payload schema and are normalized identically before PlanValidator.
    """

    resolved = capabilities or planner_provider_capabilities(provider)
    schema = dict(submit_plan_schema)
    name = str(schema.get("name", ""))
    if name != "submit_plan":
        raise ValueError("Planner schema must define the private submit_plan function.")

    if resolved.forced_named_function_calls:
        return FORCED_FUNCTION_TRANSPORT, {
            "tools": [schema],
            # Responses API named-function selection. This is stronger than "required",
            # which permits any supplied function and does not structurally select a name.
            "tool_choice": {"type": "function", "name": "submit_plan"},
            "parallel_tool_calls": False,
        }

    if resolved.strict_json_schema_output:
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("submit_plan schema requires a parameters object.")
        return STRICT_JSON_TRANSPORT, {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "submit_plan",
                    "schema": dict(parameters),
                    "strict": True,
                }
            }
        }

    raise PlannerTransportUnsupportedError(
        f"Provider {provider!r} supports neither forced named functions nor strict JSON planning output."
    )


def build_executor_request_options(
    *,
    provider: str,
    tool_schema: Mapping[str, object],
    capabilities: PlannerProviderCapabilities | None = None,
) -> tuple[str, dict[str, object]]:
    """Build provider-facing options for one already-planned application action.

    Step 17 execution knows exactly one legal application function for the current step.
    The provider transport therefore must force that named function rather than use auto/required
    selection. Providers without reliable named-function forcing fail closed; application code
    must never recover by parsing executor prose.
    """

    if capabilities is not None:
        resolved = capabilities
    else:
        try:
            resolved = planner_provider_capabilities(provider)
        except PlannerTransportUnsupportedError as exc:
            raise ExecutorTransportUnsupportedError(
                f"No structured executor transport is configured for provider {provider!r}."
            ) from exc
    schema = dict(tool_schema)
    name = schema.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Executor tool schema must define a non-empty function name.")

    if resolved.forced_named_function_calls:
        return FORCED_FUNCTION_TRANSPORT, {
            "tools": [schema],
            "tool_choice": {"type": "function", "name": name},
            "parallel_tool_calls": False,
        }

    raise ExecutorTransportUnsupportedError(
        f"Provider {provider!r} does not support reliable forced named application function calls."
    )


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_sequence(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return value
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _response_text_present(response: object) -> bool:
    output_text = _get(response, "output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return True

    for item in _as_sequence(_get(response, "output", ())):
        if _get(item, "type") != "message":
            continue
        for part in _as_sequence(_get(item, "content", ())):
            text = _get(part, "text", "")
            if isinstance(text, str) and text.strip():
                return True

    choices = _as_sequence(_get(response, "choices", ()))
    if choices:
        message = _get(choices[0], "message")
        content = _get(message, "content", "") if message is not None else ""
        if isinstance(content, str) and content.strip():
            return True
    return False


def _finish_reason(response: object) -> str | None:
    choices = _as_sequence(_get(response, "choices", ()))
    if choices:
        reason = _get(choices[0], "finish_reason")
        if reason is not None:
            return str(reason)

    incomplete = _get(response, "incomplete_details")
    if incomplete is not None:
        reason = _get(incomplete, "reason")
        if reason is not None:
            return str(reason)

    status = _get(response, "status")
    return str(status) if status is not None else None


def _responses_function_calls(response: object) -> list[NormalizedFunctionCall]:
    calls: list[NormalizedFunctionCall] = []
    for item in _as_sequence(_get(response, "output", ())):
        if _get(item, "type") != "function_call":
            continue
        calls.append(
            NormalizedFunctionCall(
                name=str(_get(item, "name", "") or ""),
                arguments=str(_get(item, "arguments", "") or ""),
                call_id=str(_get(item, "call_id", "") or _get(item, "id", "") or ""),
            )
        )
    return calls


def _chat_completion_function_calls(response: object) -> list[NormalizedFunctionCall]:
    """Preserve tool calls if an OpenAI-compatible backend returns Chat-style fields.

    The unified runner uses ``responses.create``. Some compatible providers/proxies nevertheless
    surface tool-call fields in the Chat Completions shape; preserving them here prevents a valid
    structured planner call from being misclassified as a text-only protocol failure.
    """

    choices = _as_sequence(_get(response, "choices", ()))
    if not choices:
        return []
    message = _get(choices[0], "message")
    if message is None:
        return []
    calls: list[NormalizedFunctionCall] = []
    for item in _as_sequence(_get(message, "tool_calls", ())):
        function = _get(item, "function")
        if function is None:
            continue
        calls.append(
            NormalizedFunctionCall(
                name=str(_get(function, "name", "") or ""),
                arguments=str(_get(function, "arguments", "") or ""),
                call_id=str(_get(item, "id", "") or ""),
            )
        )
    return calls


def _normalized_forced_function_calls(response: object) -> tuple[NormalizedFunctionCall, ...]:
    calls: list[NormalizedFunctionCall] = []
    calls.extend(_responses_function_calls(response))
    calls.extend(_chat_completion_function_calls(response))
    # De-duplicate if a compatibility wrapper exposes the same call in both shapes.
    deduped: list[NormalizedFunctionCall] = []
    seen: set[tuple[str, str, str]] = set()
    for call in calls:
        key = (call.call_id, call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return tuple(deduped)


def normalize_planner_response(response: object, *, transport: str) -> NormalizedPlannerResponse:
    """Normalize structured planner output without reading reasoning or arbitrary prose."""

    calls: tuple[NormalizedFunctionCall, ...] = ()
    if transport == FORCED_FUNCTION_TRANSPORT:
        calls = _normalized_forced_function_calls(response)
    elif transport == STRICT_JSON_TRANSPORT:
        output_text = _get(response, "output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            calls = (
                NormalizedFunctionCall(
                    name="submit_plan",
                    arguments=output_text,
                    call_id="structured_json",
                ),
            )
    else:
        raise PlannerTransportUnsupportedError(f"Unknown planner transport: {transport!r}")

    return NormalizedPlannerResponse(
        finish_reason=_finish_reason(response),
        text_present=_response_text_present(response),
        tool_calls=calls,
        transport=transport,
    )


def normalize_executor_response(response: object, *, transport: str) -> NormalizedExecutorResponse:
    """Normalize one planned application function response across supported API shapes.

    Only structured function calls are preserved. Text is observed only as a boolean diagnostic;
    it is never parsed into application arguments.
    """

    if transport != FORCED_FUNCTION_TRANSPORT:
        raise ExecutorTransportUnsupportedError(f"Unknown executor transport: {transport!r}")
    return NormalizedExecutorResponse(
        finish_reason=_finish_reason(response),
        text_present=_response_text_present(response),
        tool_calls=_normalized_forced_function_calls(response),
        transport=transport,
    )


def planner_protocol_debug_enabled() -> bool:
    value = (os.getenv("BONSAI_AGENT_SHOW_PLANNER_PROTOCOL") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def planner_protocol_debug_payload(
    normalized: NormalizedPlannerResponse,
    *,
    expected_name: str = "submit_plan",
) -> dict[str, object]:
    parsed: bool | None = None
    if len(normalized.tool_calls) == 1 and normalized.tool_calls[0].name == expected_name:
        try:
            json.loads(normalized.tool_calls[0].arguments)
        except (TypeError, json.JSONDecodeError):
            parsed = False
        else:
            parsed = True
    return {
        "finish_reason": normalized.finish_reason,
        "text_returned": normalized.text_present,
        "tool_call_count": len(normalized.tool_calls),
        "tool_call_names": [call.name for call in normalized.tool_calls],
        "submit_plan_arguments_parsed": parsed,
        "transport": normalized.transport,
    }


def emit_planner_protocol_debug(normalized: NormalizedPlannerResponse) -> None:
    if not planner_protocol_debug_enabled():
        return
    payload = planner_protocol_debug_payload(normalized)
    print(
        "[planner_protocol] " + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )


def executor_protocol_debug_enabled() -> bool:
    value = (os.getenv("BONSAI_AGENT_SHOW_EXECUTOR_PROTOCOL") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def executor_protocol_debug_payload(
    normalized: NormalizedExecutorResponse | None,
    *,
    plan_id: str,
    step_id: str,
    planned_tool: str,
    active_problem_id: str,
    transport: str,
    arguments_parsed: bool | None,
    arguments_schema_valid: bool | None,
    dispatch_attempted: bool,
    literal_binding_count: int = 0,
    literal_bindings_preserved: bool | None = None,
    deferred_binding_count: int = 0,
    deferred_bindings_resolved: bool | None = None,
    unexpected_argument_paths: Sequence[str] = (),
    argument_intent_valid: bool | None = None,
) -> dict[str, object]:
    return {
        "plan_id": plan_id,
        "step_id": step_id,
        "planned_tool": planned_tool,
        "active_problem_id": active_problem_id,
        "transport": transport,
        "finish_reason": normalized.finish_reason if normalized is not None else None,
        "text_returned": normalized.text_present if normalized is not None else False,
        "tool_call_count": len(normalized.tool_calls) if normalized is not None else 0,
        "tool_call_names": [call.name for call in normalized.tool_calls] if normalized is not None else [],
        "arguments_parsed": arguments_parsed,
        "arguments_schema_valid": arguments_schema_valid,
        "literal_binding_count": int(literal_binding_count),
        "literal_bindings_preserved": literal_bindings_preserved,
        "deferred_binding_count": int(deferred_binding_count),
        "deferred_bindings_resolved": deferred_bindings_resolved,
        "unexpected_argument_paths": list(unexpected_argument_paths),
        "argument_intent_valid": argument_intent_valid,
        "dispatch_attempted": bool(dispatch_attempted),
    }


def emit_executor_protocol_debug(payload: Mapping[str, object]) -> None:
    if not executor_protocol_debug_enabled():
        return
    print(
        "[executor_protocol] " + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )


__all__ = [
    "FORCED_FUNCTION_TRANSPORT",
    "STRICT_JSON_TRANSPORT",
    "ExecutorTransportUnsupportedError",
    "NormalizedFunctionCall",
    "NormalizedPlannerResponse",
    "NormalizedExecutorResponse",
    "PlannerProviderCapabilities",
    "PlannerTransportUnsupportedError",
    "build_planner_request_options",
    "build_executor_request_options",
    "emit_planner_protocol_debug",
    "emit_executor_protocol_debug",
    "normalize_planner_response",
    "normalize_executor_response",
    "planner_protocol_debug_enabled",
    "executor_protocol_debug_enabled",
    "executor_protocol_debug_payload",
    "planner_protocol_debug_payload",
    "planner_provider_capabilities",
]
