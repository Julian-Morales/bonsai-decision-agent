"""UX-1.2 presentation-only labels and sidebar visibility for immutable runs.

Keys are (problem_id, run_id); the store never touches optimization history,
solution handles, or active session state. Hidden rows can always be restored.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict

from webapp.storage import StorageError, _atomic_json_write, _load_json


class RunUIMetadataStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "run_ui_metadata.json"
        self._lock = threading.RLock()

    @staticmethod
    def _key(problem_id: str, run_id: str) -> str:
        if not problem_id or not run_id:
            raise StorageError("A problem_id and run_id are required")
        return f"{problem_id}:{run_id}"

    def _read(self) -> Dict[str, object]:
        payload = _load_json(self.path, {"version": 1, "runs": {}})
        if payload.get("version") != 1 or not isinstance(payload.get("runs"), dict):
            raise StorageError("Invalid run UI metadata format")
        return payload

    def list(self) -> list[Dict[str, object]]:
        with self._lock:
            values = self._read()["runs"]
            return [dict(value) for value in values.values() if isinstance(value, dict)]

    def update(self, problem_id: str, run_id: str, *, action: str, label: object = None) -> Dict[str, object]:
        key = self._key(problem_id, run_id)
        if action not in {"rename", "archive", "restore"}:
            raise StorageError("Unsupported run UI action")
        with self._lock:
            payload = self._read()
            rows = payload["runs"]
            original = rows.get(key)
            row = dict(original) if isinstance(original, dict) else {
                "problem_id": problem_id, "run_id": run_id, "label": None, "archived": False,
            }
            if action == "rename":
                if not isinstance(label, str):
                    raise StorageError("Run label must be text")
                cleaned = label.strip()
                if not cleaned or len(cleaned) > 100 or any(ord(c) < 32 for c in cleaned):
                    raise StorageError("Run label must contain 1–100 printable characters")
                row["label"] = cleaned
            elif action == "archive":
                row["archived"] = True
            elif action == "restore":
                row["archived"] = False
            rows[key] = row
            _atomic_json_write(self.path, payload)
            return dict(row)
