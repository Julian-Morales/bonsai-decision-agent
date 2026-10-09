"""Local-only stdlib HTTP server for PRODUCTIZATION UX-1.2."""
from __future__ import annotations

import json
import logging
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Mapping
from urllib.parse import unquote, urlsplit

from problem_registry import ProblemNotRegisteredError
from webapp.agent_adapter import DecisionAgentUIAdapter
from webapp.connections import ConnectionService
from webapp.storage import ConnectionProfileStore, ProfileNotFoundError, StorageError
from webapp.run_ui_metadata import RunUIMetadataStore


LOGGER = logging.getLogger("bonsai.web")
MAX_JSON_BODY = 1_000_000


class BonsaiWebApplication:
    """Small JSON API boundary; all core decisions stay inside the frozen agent."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_dir: Path | None = None,
        profile_store: ConnectionProfileStore | None = None,
        connection_service: ConnectionService | None = None,
        agent_adapter: DecisionAgentUIAdapter | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.static_dir = self.project_root / "webapp" / "static"
        self.profile_store = profile_store or ConnectionProfileStore(data_dir or (self.project_root / "data"))
        self.run_ui_store = RunUIMetadataStore(data_dir or (self.project_root / "data"))
        self.connection_service = connection_service or ConnectionService()
        self.agent_adapter = agent_adapter or DecisionAgentUIAdapter(self.profile_store)

    def connections_payload(self) -> Dict[str, object]:
        profiles = []
        for profile in self.profile_store.public_profiles():
            item = dict(profile)
            item["connection_state"] = self.connection_service.state_for(
                str(item["id"]), configured=bool(item["api_key_configured"])
            )
            # Diagnostics are ephemeral and do not determine active-profile identity.
            item["last_connection_test"] = self.connection_service.last_test_for(str(item["id"]))
            item["latest_diagnostic"] = self.connection_service.latest_diagnostic_for(str(item["id"]))
            profiles.append(item)
        config = self.profile_store.public_config()
        return {
            "profiles": profiles,
            "active_profile_id": config["active_connection_profile_id"],
            "allow_key_exposure": False,
        }

    def workspace_payload(self) -> Dict[str, object]:
        return {
            "problems": self.agent_adapter.problems(),
            "scenarios": self.agent_adapter.scenarios(),
            "runs": self.agent_adapter.runs(),
            "run_ui_metadata": self.run_ui_store.list(),
        }

    def handle_json(self, method: str, path: str, body: Mapping[str, object] | None) -> tuple[int, Dict[str, object]]:
        payload = dict(body or {})
        try:
            if method == "GET" and path == "/api/problems":
                return 200, {"problems": self.agent_adapter.problems()}
            if method == "GET" and path == "/api/scenarios":
                return 200, {"scenarios": self.agent_adapter.scenarios()}
            match = re.fullmatch(r"/api/problems/([^/]+)/scenarios/([^/]+)", path)
            if method == "GET" and match:
                return 200, self.agent_adapter.scenario_detail(unquote(match.group(1)), unquote(match.group(2)))
            if method == "GET" and path == "/api/runs":
                return 200, {"runs": self.agent_adapter.runs()}
            if method == "GET" and path == "/api/run-ui":
                return 200, {"run_ui_metadata": self.run_ui_store.list()}
            match = re.fullmatch(r"/api/run-ui/([^/]+)/([^/]+)/(rename|archive|restore)", path)
            if method == "POST" and match:
                problem_id, run_id, action = (unquote(part) for part in match.groups())
                # Read-only registry/history membership check. No run, solution or alias mutation.
                self.agent_adapter.assert_run_exists(problem_id, run_id)
                row = self.run_ui_store.update(problem_id, run_id, action=action, label=payload.get("label"))
                return 200, {"run_ui": row}

            # UX-1.1 read-only historical navigation. The problem-scoped form is
            # unambiguous because run ids are sequential within each problem store.
            match = re.fullmatch(r"/api/problems/([^/]+)/runs/([^/]+)", path)
            if method == "GET" and match:
                return 200, self.agent_adapter.historical_run(
                    unquote(match.group(1)), unquote(match.group(2))
                )
            match = re.fullmatch(r"/api/runs/([^/]+)", path)
            if method == "GET" and match:
                return 200, self.agent_adapter.historical_run_by_id(unquote(match.group(1)))

            if method == "GET" and path == "/api/workspace":
                return 200, self.workspace_payload()
            if method == "POST" and path == "/api/chat":
                return 200, self.agent_adapter.chat(str(payload.get("message", "")))

            match = re.fullmatch(r"/api/problems/([^/]+)/select", path)
            if method == "POST" and match:
                return 200, self.agent_adapter.select_problem(unquote(match.group(1)))

            match = re.fullmatch(r"/api/approvals/([^/]+)/(approve|reject)", path)
            if method == "POST" and match:
                approval_id = unquote(match.group(1))
                action = match.group(2)
                return 200, self.agent_adapter.approve(approval_id) if action == "approve" else self.agent_adapter.reject(approval_id)

            if method == "GET" and path == "/api/connections":
                return 200, self.connections_payload()
            if method == "POST" and path == "/api/connections":
                profile = self.profile_store.create(payload)
                return 201, {"profile": profile.public_dict(api_key_configured=False), **self.connections_payload()}

            match = re.fullmatch(r"/api/connections/([^/]+)", path)
            if match and method == "PUT":
                profile = self.profile_store.update(unquote(match.group(1)), payload)
                return 200, {"profile": profile.public_dict(api_key_configured=self.profile_store.has_key(profile.id)), **self.connections_payload()}
            if match and method == "DELETE":
                self.profile_store.delete(unquote(match.group(1)))
                return 200, {"ok": True, **self.connections_payload()}

            match = re.fullmatch(r"/api/connections/([^/]+)/duplicate", path)
            if method == "POST" and match:
                profile = self.profile_store.duplicate(
                    unquote(match.group(1)),
                    display_name=str(payload.get("display_name", "")).strip() or None,
                )
                return 201, {"profile": profile.public_dict(api_key_configured=False), **self.connections_payload()}

            match = re.fullmatch(r"/api/connections/([^/]+)/activate", path)
            if method == "POST" and match:
                profile = self.profile_store.set_active(unquote(match.group(1)))
                return 200, {"ok": True, "active_profile_id": profile.id, **self.connections_payload()}

            match = re.fullmatch(r"/api/connections/([^/]+)/key", path)
            if method == "POST" and match:
                self.profile_store.set_key(unquote(match.group(1)), str(payload.get("api_key", "")))
                return 200, {"ok": True, **self.connections_payload()}
            if method == "DELETE" and match:
                self.profile_store.delete_key(unquote(match.group(1)))
                return 200, {"ok": True, **self.connections_payload()}

            match = re.fullmatch(r"/api/connections/([^/]+)/test", path)
            if method == "POST" and match:
                profile = self.profile_store.get(unquote(match.group(1)))
                secret = self.profile_store.get_secret(profile.id)
                result = self.connection_service.test(profile, secret)
                return (200 if result.ok else 502), result.to_dict()

            match = re.fullmatch(r"/api/connections/([^/]+)/models", path)
            if method == "GET" and match:
                profile = self.profile_store.get(unquote(match.group(1)))
                secret = self.profile_store.get_secret(profile.id)
                result = self.connection_service.fetch_models(profile, secret)
                return (200 if result.get("ok") else 502), result

            return 404, {"error": {"code": "not_found", "message": "Endpoint not found."}}
        except (StorageError, ProfileNotFoundError, ProblemNotRegisteredError, ValueError) as exc:
            # These validation messages are configuration metadata only; request bodies are never logged.
            return 400, {"error": {"code": "invalid_request", "message": str(exc)}}
        except Exception:
            # Never serialize arbitrary exception text: provider exceptions can contain request metadata.
            return 500, {"error": {"code": "internal_error", "message": "The local application encountered an error."}}


class BonsaiRequestHandler(BaseHTTPRequestHandler):
    server_version = "BonsaiUX2/2.0"

    @property
    def app(self) -> BonsaiWebApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Do not use BaseHTTPRequestHandler's raw target log; normalized path only, no body/query.
        path = urlsplit(self.path).path
        LOGGER.info("%s %s", self.command, path)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'")

    def _send_json(self, status: int, payload: Mapping[str, object]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        try:
            raw = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _json_body(self) -> Dict[str, object]:
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > MAX_JSON_BODY:
            raise ValueError("Request body is too large")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def _api(self, method: str) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._json_body() if method in {"POST", "PUT"} else {}
            status, payload = self.app.handle_json(method, path, body)
        except ValueError as exc:
            status, payload = 400, {"error": {"code": "invalid_json", "message": str(exc)}}
        self._send_json(status, payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self._api("GET")
            return
        if path in {"/", "/index.html"}:
            self._send_file(self.app.static_dir / "index.html")
            return
        if path == "/static/styles.css":
            self._send_file(self.app.static_dir / "styles.css")
            return
        if path == "/static/app.js":
            self._send_file(self.app.static_dir / "app.js")
            return
        if path == "/static/markdown.js":
            self._send_file(self.app.static_dir / "markdown.js")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        self._api("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._api("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._api("DELETE")


def create_server(app: BonsaiWebApplication, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), BonsaiRequestHandler)
    server.app = app  # type: ignore[attr-defined]
    server.daemon_threads = True
    return server
