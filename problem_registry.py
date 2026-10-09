"""Explicit allowlisted optimization-problem registry for multi-domain sessions.

The registry owns only preconstructed adapter instances supplied by application code.
It performs no imports, discovery, dynamic construction, or user-driven loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Tuple

from problem_adapter import OptimizationProblemAdapter


class ProblemRegistryError(Exception):
    """Base class for deterministic registry failures."""


class ProblemAlreadyRegisteredError(ProblemRegistryError):
    """Raised when the application tries to register a duplicate problem id."""


class ProblemNotRegisteredError(ProblemRegistryError):
    """Raised when a requested problem id is not in the explicit allowlist."""


@dataclass(frozen=True)
class RegisteredProblemMetadata:
    problem_id: str
    display_name: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "problem_id": self.problem_id,
            "display_name": self.display_name,
            "description": self.description,
        }


class ProblemRegistry:
    """Read-only allowlist of application-constructed problem adapters."""

    def __init__(self, adapters: Iterable[OptimizationProblemAdapter]) -> None:
        registered: Dict[str, OptimizationProblemAdapter] = {}
        metadata: Dict[str, RegisteredProblemMetadata] = {}
        for adapter in adapters:
            problem_id = adapter.problem_id
            if not isinstance(problem_id, str) or not problem_id.strip():
                raise ValueError("Registered adapters require a non-empty problem_id.")
            problem_id = problem_id.strip()
            if problem_id in registered:
                raise ProblemAlreadyRegisteredError(
                    f"Problem is already registered: {problem_id}"
                )
            if adapter.spec.problem_id != problem_id:
                raise ValueError(
                    "Adapter problem_id must match OptimizationProblemSpec.problem_id."
                )
            registered[problem_id] = adapter
            metadata[problem_id] = RegisteredProblemMetadata(
                problem_id=problem_id,
                display_name=adapter.spec.display_name,
                description=adapter.spec.description,
            )
        if not registered:
            raise ValueError("ProblemRegistry requires at least one explicit adapter.")
        self._adapters: Mapping[str, OptimizationProblemAdapter] = MappingProxyType(registered)
        self._metadata: Mapping[str, RegisteredProblemMetadata] = MappingProxyType(metadata)

    @property
    def problem_ids(self) -> Tuple[str, ...]:
        return tuple(self._adapters.keys())

    def get(self, problem_id: str) -> OptimizationProblemAdapter:
        if not isinstance(problem_id, str):
            raise ProblemNotRegisteredError("Problem id must be text.")
        key = problem_id.strip()
        try:
            return self._adapters[key]
        except KeyError as exc:
            raise ProblemNotRegisteredError(
                f"Problem is not registered: {problem_id}"
            ) from exc

    def metadata(self) -> Tuple[RegisteredProblemMetadata, ...]:
        return tuple(self._metadata[problem_id] for problem_id in self.problem_ids)

    def __contains__(self, problem_id: object) -> bool:
        return isinstance(problem_id, str) and problem_id in self._adapters


__all__ = [
    "ProblemAlreadyRegisteredError",
    "ProblemNotRegisteredError",
    "ProblemRegistry",
    "ProblemRegistryError",
    "RegisteredProblemMetadata",
]
