"""HTTP del motor. Solo POST /context; no abre el socket de la app."""

from __future__ import annotations

import errno
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from agent.contract import InvalidContext, Place, clean_text, parse_context
from agent.graph import build_guide
from agent.retrieve import load_index
from agent.settings import Settings, load_settings

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 65_536
JSON_ERROR = "El cuerpo tiene que ser un JSON"
FAILURE_MESSAGE = "No pude armar el relato en este momento."


def serve(env_file: Path) -> None:
    settings = load_settings(env_file)
    Handler.settings = settings
    host = read_host()
    port = read_port()
    try:
        server = listen(host, port)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            sys.stderr.write(f"El puerto {port} ya está en uso.\n")
            raise SystemExit(1) from None
        raise

    print(f"Escuchando en http://{host}:{port}", flush=True)
    print(settings.startup_line, flush=True)
    print(settings.jev_line, flush=True)
    counts = ", ".join(f"{city} {total}" for city, total in load_index().counts().items())
    print(f"Contexto histórico listo ({counts or 'sin ciudades'}). Si falta, se busca en Wikipedia.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("Servidor detenido.\n")
    finally:
        server.server_close()


def read_host() -> str:
    host = os.environ.get("HOST", DEFAULT_HOST).strip()
    return host or DEFAULT_HOST


def read_port() -> int:
    raw = os.environ.get("PORT", str(DEFAULT_PORT)).strip() or str(DEFAULT_PORT)
    try:
        port = int(raw)
    except ValueError:
        sys.stderr.write("PORT debe ser un entero entre 1 y 65535.\n")
        raise SystemExit(1) from None
    if not 1 <= port <= 65535:
        sys.stderr.write("PORT debe ser un entero entre 1 y 65535.\n")
        raise SystemExit(1)
    return port


def listen(host: str, port: int) -> ThreadingHTTPServer:
    class GuideServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 32

        def handle_error(self, request, client_address) -> None:
            exc_type = sys.exc_info()[0]
            name = exc_type.__name__ if exc_type is not None else "Error"
            sys.stderr.write(f"error al atender un pedido ({name})\n")

    return GuideServer((host, port), Handler)


class Handler(BaseHTTPRequestHandler):
    settings: Settings
    timeout = 20
    protocol_version = "HTTP/1.1"

    def version_string(self) -> str:
        return "tour-ai-agent"

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        sys.stderr.write(f"{self.command} {code}\n")

    def setup(self) -> None:
        super().setup()
        self._sent = False

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def _dispatch(self, method: str) -> None:
        self.close_connection = True
        try:
            self._route(method)
        except Exception:
            sys.stderr.write("no se pudo responder\n")
            self._json(200, {"message": FAILURE_MESSAGE})

    def _route(self, method: str) -> None:
        if urlsplit(self.path).path != "/context":
            self._json(404, {"message": "No encontrado"})
            return
        if method != "POST":
            self._json(405, {"message": "Método no permitido"})
            return
        raw = self._read_body()
        if raw is None:
            return
        status, body = response_for(raw, self.settings)
        self._json(status, body)

    def _read_body(self) -> bytes | None:
        lengths = self.headers.get_all("Content-Length") or []
        problem = body_length_error(
            lengths[0] if len(lengths) == 1 else None,
            self.headers.get("Transfer-Encoding"),
            length_count=len(lengths),
        )
        if problem is not None:
            self._json(400, {"message": problem})
            return None
        length = int(lengths[0].strip())
        try:
            raw = self.rfile.read(length)
        except (TimeoutError, OSError):
            self._json(400, {"message": JSON_ERROR})
            return None
        if len(raw) != length:
            self._json(400, {"message": JSON_ERROR})
            return None
        return raw

    def _json(self, status: int, body: dict[str, object]) -> None:
        if self._sent:
            return
        self._sent = True
        try:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except (UnicodeEncodeError, ValueError):
            status = 200
            raw = json.dumps({"message": FAILURE_MESSAGE}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if status == 405:
            self.send_header("Allow", "POST")
        self.end_headers()
        self.wfile.write(raw)


def response_for(raw: bytes, settings: Settings) -> tuple[int, dict[str, object]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, {"message": JSON_ERROR}
    try:
        context = parse_context(payload)
    except InvalidContext as error:
        return 400, {"message": error.message}
    try:
        guide = build_guide(
            context,
            settings,
            user_id=optional_user_id(payload) if isinstance(payload, dict) else "",
            speak=not (isinstance(payload, dict) and payload.get("guideLive") is False),
        )
    except Exception:
        sys.stderr.write("no se pudo armar el relato\n")
        return 200, {"message": FAILURE_MESSAGE}
    if not isinstance(guide.reply, str):
        return 200, {"message": FAILURE_MESSAGE}
    reply = guide.reply.strip()
    body: dict[str, object] = {
        "reply": reply,
        "interests": [interest_payload(place) for place in guide.places],
    }
    if guide.place is not None and reply:
        body["interest"] = interest_payload(guide.place)
    return 200, body


def interest_payload(place: Place) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": place.name,
        "latitude": place.latitude,
        "longitude": place.longitude,
    }
    if place.category:
        payload["category"] = place.category
    return payload


def optional_user_id(payload: dict[str, object]) -> str:
    value = payload.get("userId")
    if not isinstance(value, str):
        return ""
    return clean_text(value, 80)


def body_length_error(
    content_length: str | None,
    transfer_encoding: str | None,
    length_count: int = 1,
) -> str | None:
    if transfer_encoding or length_count != 1:
        return JSON_ERROR
    if content_length is None:
        return JSON_ERROR
    header = content_length.strip()
    if len(header) > 8 or not header.isdigit():
        return JSON_ERROR
    length = int(header)
    if length == 0:
        return JSON_ERROR
    if length > MAX_BODY_BYTES:
        return "El cuerpo es demasiado grande"
    return None
