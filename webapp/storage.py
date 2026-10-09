"""Server-side JSON persistence for UX-1 connection profiles and secrets.

API keys are stored only in ``secrets.json`` and are never returned by the normal
profile-loading interface. Storage is plaintext by design for this local UX-1 shell;
filesystem access therefore implies access to the stored secrets.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping


PROVIDER_TYPES = ("nvidia", "openai", "custom_openai")


class StorageError(RuntimeError):
    """Predictable UX storage/configuration failure."""


class ProfileNotFoundError(StorageError):
    """Raised for an unknown connection profile id."""


@dataclass(frozen=True)
class ConnectionProfile:
    id: str
    display_name: str
    provider: str
    base_url: str | None
    model: str
    secret_ref: str

    def public_dict(self, *, api_key_configured: bool) -> Dict[str, object]:
        # ``secret_ref`` is intentionally server-side metadata only.
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_configured": bool(api_key_configured),
        }

    def stored_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "secret_ref": self.secret_ref,
        }


def _atomic_json_write(path: Path, payload: Mapping[str, object], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_path, mode)
        except OSError:
            pass
        os.replace(tmp_path, path)
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_json(path: Path, default: Mapping[str, object]) -> Dict[str, object]:
    if not path.exists():
        return deepcopy(dict(default))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Could not read application data file: {path.name}") from exc
    if not isinstance(value, dict):
        raise StorageError(f"Application data file must contain a JSON object: {path.name}")
    return value


def _clean_text(value: object, field: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise StorageError(f"{field} is required")
        return None
    text = str(value).strip()
    if required and not text:
        raise StorageError(f"{field} is required")
    return text or None


class ConnectionProfileStore:
    """Thread-safe local persistence for profile metadata, secrets, and UI config."""

    def __init__(
        self,
        data_dir: Path,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.profile_path = self.data_dir / "connection_profiles.json"
        self.secret_path = self.data_dir / "secrets.json"
        self.ui_config_path = self.data_dir / "ui_config.json"
        self._id_factory = id_factory or (lambda: f"profile_{uuid.uuid4().hex[:12]}")
        self._lock = threading.RLock()
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.profile_path.exists():
                _atomic_json_write(self.profile_path, {"version": 1, "profiles": []})
            if not self.secret_path.exists():
                _atomic_json_write(self.secret_path, {"version": 1, "secrets": {}})
            if not self.ui_config_path.exists():
                _atomic_json_write(
                    self.ui_config_path,
                    {
                        "version": 1,
                        "active_connection_profile_id": None,
                        "allowKeyExposure": False,
                    },
                )

    def _profile_payload(self) -> Dict[str, object]:
        payload = _load_json(self.profile_path, {"version": 1, "profiles": []})
        if not isinstance(payload.get("profiles"), list):
            raise StorageError("connection_profiles.json has an invalid profiles collection")
        return payload

    def _secret_payload(self) -> Dict[str, object]:
        payload = _load_json(self.secret_path, {"version": 1, "secrets": {}})
        if not isinstance(payload.get("secrets"), dict):
            raise StorageError("secrets.json has an invalid secrets collection")
        return payload

    def _ui_payload(self) -> Dict[str, object]:
        payload = _load_json(
            self.ui_config_path,
            {"version": 1, "active_connection_profile_id": None, "allowKeyExposure": False},
        )
        # UX-1 has no key-exposure mechanism. Fail closed even if the file is hand-edited.
        payload["allowKeyExposure"] = False
        return payload

    @staticmethod
    def _from_stored(value: Mapping[str, object]) -> ConnectionProfile:
        return ConnectionProfile(
            id=str(value["id"]),
            display_name=str(value["display_name"]),
            provider=str(value["provider"]),
            base_url=str(value["base_url"]) if value.get("base_url") is not None else None,
            model=str(value["model"]),
            secret_ref=str(value["secret_ref"]),
        )

    def list_profiles(self) -> tuple[ConnectionProfile, ...]:
        with self._lock:
            payload = self._profile_payload()
            return tuple(self._from_stored(item) for item in payload["profiles"] if isinstance(item, dict))

    def get(self, profile_id: str) -> ConnectionProfile:
        key = str(profile_id).strip()
        for profile in self.list_profiles():
            if profile.id == key:
                return profile
        raise ProfileNotFoundError(f"Connection profile is not registered: {profile_id}")

    def _validate_profile_fields(self, values: Mapping[str, object]) -> tuple[str, str, str | None, str]:
        display_name = _clean_text(values.get("display_name"), "display_name")
        provider = _clean_text(values.get("provider"), "provider")
        if provider not in PROVIDER_TYPES:
            raise StorageError(f"Unsupported provider type: {provider}")
        base_url = _clean_text(values.get("base_url"), "base_url", required=False)
        if provider == "custom_openai" and not base_url:
            raise StorageError("base_url is required for Custom OpenAI-compatible profiles")
        # UX-1.1: a model is selected after endpoint/auth setup and model discovery.
        # Keep manual model entry as an optional fallback, not a profile-creation prerequisite.
        model = _clean_text(values.get("model"), "model", required=False) or ""
        return str(display_name), str(provider), base_url, str(model)

    def create(self, values: Mapping[str, object]) -> ConnectionProfile:
        display_name, provider, base_url, model = self._validate_profile_fields(values)
        with self._lock:
            payload = self._profile_payload()
            profile_id = self._id_factory()
            existing_ids = {str(item.get("id")) for item in payload["profiles"] if isinstance(item, dict)}
            while profile_id in existing_ids:
                profile_id = self._id_factory()
            profile = ConnectionProfile(
                id=profile_id,
                display_name=display_name,
                provider=provider,
                base_url=base_url,
                model=model,
                secret_ref=f"secret_{profile_id}",
            )
            payload["profiles"].append(profile.stored_dict())
            _atomic_json_write(self.profile_path, payload)
            ui = self._ui_payload()
            if not ui.get("active_connection_profile_id") and profile.model.strip():
                ui["active_connection_profile_id"] = profile.id
                _atomic_json_write(self.ui_config_path, ui)
            return profile

    def update(self, profile_id: str, values: Mapping[str, object]) -> ConnectionProfile:
        current = self.get(profile_id)
        merged = {
            "display_name": values.get("display_name", current.display_name),
            "provider": values.get("provider", current.provider),
            "base_url": values.get("base_url", current.base_url),
            "model": values.get("model", current.model),
        }
        display_name, provider, base_url, model = self._validate_profile_fields(merged)
        updated = ConnectionProfile(
            id=current.id,
            display_name=display_name,
            provider=provider,
            base_url=base_url,
            model=model,
            secret_ref=current.secret_ref,
        )
        with self._lock:
            payload = self._profile_payload()
            found = False
            for index, item in enumerate(payload["profiles"]):
                if isinstance(item, dict) and item.get("id") == current.id:
                    payload["profiles"][index] = updated.stored_dict()
                    found = True
                    break
            if not found:
                raise ProfileNotFoundError(f"Connection profile is not registered: {profile_id}")
            _atomic_json_write(self.profile_path, payload)
            # A model-less profile is valid setup state but cannot remain the active
            # agent runtime profile. Clearing an active model therefore deactivates it.
            if not updated.model.strip():
                ui = self._ui_payload()
                if ui.get("active_connection_profile_id") == updated.id:
                    ui["active_connection_profile_id"] = None
                    ui["allowKeyExposure"] = False
                    _atomic_json_write(self.ui_config_path, ui)
        return updated

    def duplicate(self, profile_id: str, *, display_name: str | None = None) -> ConnectionProfile:
        source = self.get(profile_id)
        # Deliberately do not copy the secret. The duplicate receives an independent secret ref.
        return self.create(
            {
                "display_name": display_name or f"{source.display_name} copy",
                "provider": source.provider,
                "base_url": source.base_url,
                "model": source.model,
            }
        )

    def delete(self, profile_id: str) -> None:
        profile = self.get(profile_id)
        with self._lock:
            payload = self._profile_payload()
            payload["profiles"] = [
                item for item in payload["profiles"]
                if not (isinstance(item, dict) and item.get("id") == profile.id)
            ]
            _atomic_json_write(self.profile_path, payload)
            secrets = self._secret_payload()
            secrets["secrets"].pop(profile.secret_ref, None)
            _atomic_json_write(self.secret_path, secrets)
            ui = self._ui_payload()
            if ui.get("active_connection_profile_id") == profile.id:
                remaining = [self._from_stored(item) for item in payload["profiles"] if isinstance(item, dict)]
                usable = next((item for item in remaining if item.model.strip()), None)
                ui["active_connection_profile_id"] = usable.id if usable is not None else None
                ui["allowKeyExposure"] = False
                _atomic_json_write(self.ui_config_path, ui)

    def set_key(self, profile_id: str, api_key: str) -> None:
        profile = self.get(profile_id)
        if not isinstance(api_key, str) or not api_key.strip():
            raise StorageError("api_key must be non-empty text")
        with self._lock:
            payload = self._secret_payload()
            payload["secrets"][profile.secret_ref] = api_key.strip()
            _atomic_json_write(self.secret_path, payload)

    def delete_key(self, profile_id: str) -> None:
        profile = self.get(profile_id)
        with self._lock:
            payload = self._secret_payload()
            payload["secrets"].pop(profile.secret_ref, None)
            _atomic_json_write(self.secret_path, payload)

    def has_key(self, profile_id: str) -> bool:
        profile = self.get(profile_id)
        with self._lock:
            payload = self._secret_payload()
            value = payload["secrets"].get(profile.secret_ref)
            return isinstance(value, str) and bool(value)

    def get_secret(self, profile_id: str) -> str:
        """Server-side only. Never expose this value through normal frontend API payloads."""
        profile = self.get(profile_id)
        with self._lock:
            payload = self._secret_payload()
            value = payload["secrets"].get(profile.secret_ref)
            if not isinstance(value, str) or not value:
                raise StorageError("API key is not configured for this profile")
            return value

    def public_profiles(self) -> tuple[Dict[str, object], ...]:
        return tuple(
            profile.public_dict(api_key_configured=self.has_key(profile.id))
            for profile in self.list_profiles()
        )

    @property
    def active_profile_id(self) -> str | None:
        with self._lock:
            value = self._ui_payload().get("active_connection_profile_id")
            return str(value) if isinstance(value, str) and value else None

    def set_active(self, profile_id: str) -> ConnectionProfile:
        profile = self.get(profile_id)
        if not profile.model.strip():
            raise StorageError("A model must be selected before this profile can be used by the agent")
        with self._lock:
            payload = self._ui_payload()
            payload["active_connection_profile_id"] = profile.id
            payload["allowKeyExposure"] = False
            _atomic_json_write(self.ui_config_path, payload)
        return profile

    def active_profile(self) -> ConnectionProfile | None:
        profile_id = self.active_profile_id
        if profile_id is None:
            return None
        try:
            return self.get(profile_id)
        except ProfileNotFoundError:
            return None

    def public_config(self) -> Dict[str, object]:
        return {
            "active_connection_profile_id": self.active_profile_id,
            "allowKeyExposure": False,
        }
