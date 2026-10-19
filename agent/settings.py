"""Carga la clave del modelo desde .env. El pedido HTTP no puede cambiar la URL."""

from __future__ import annotations

import math
import os
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_KEYS = ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME", "JEV_API_KEY", "JEV_CONFIDENCE_THRESHOLD")
DEFAULT_MODEL_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_ENV_BYTES = 16_384
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class Settings:
    model_key: str
    model_name: str
    model_base_url: str | None
    startup_line: str
    jev_key: str
    jev_threshold: float
    jev_line: str

    @property
    def model_enabled(self) -> bool:
        return bool(self.model_key and self.model_name and self.model_base_url)


def load_settings(env_file: Path) -> Settings:
    values = {key: "" for key in ENV_KEYS}
    values.update(read_env_file(env_file))
    for key in ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return settings_from(values)


def settings_from(values: Mapping[str, str]) -> Settings:
    raw_key = values.get("MODEL_API_KEY", "")
    key = usable_key(raw_key)
    name = usable_model_name(values.get("MODEL_NAME", ""))
    raw_url = values.get("MODEL_BASE_URL", "").strip() or DEFAULT_MODEL_BASE_URL
    base_url = parse_model_base(raw_url)
    host = model_host(base_url)

    if not key and raw_key.strip():
        line = "La clave del modelo no es válida: el relato lo arma la voz local."
        base_url = None
    elif not key:
        line = "Sin clave de modelo: el relato lo arma la voz local."
        base_url = None
    elif base_url is None:
        line = "La URL del modelo no es válida: el relato lo arma la voz local."
    elif not name:
        line = "El nombre del modelo no es válido: el relato lo arma la voz local."
        base_url = None
    else:
        line = f"Modelo configurado ({host})."

    raw_jev = values.get("JEV_API_KEY", "")
    jev_key = usable_key(raw_jev)
    if jev_key:
        jev_line = "Evaluador Jev configurado."
    elif raw_jev.strip():
        jev_line = "La clave de Jev no es válida: el evaluador se simula en local."
    else:
        jev_line = "Sin clave de Jev: el evaluador se simula en local."

    return Settings(
        model_key=key,
        model_name=name,
        model_base_url=base_url,
        startup_line=line,
        jev_key=jev_key,
        jev_threshold=parse_threshold(values.get("JEV_CONFIDENCE_THRESHOLD", "")),
        jev_line=jev_line,
    )


def parse_threshold(value: str) -> float:
    raw = value.strip()
    if not raw:
        return 0.85
    try:
        number = float(raw)
    except ValueError:
        return 0.85
    if not math.isfinite(number) or not 0.5 <= number <= 0.99:
        return 0.85
    return number


def completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return trimmed + "/chat/completions"


def read_env_file(path: Path) -> dict[str, str]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_ENV_BYTES:
            return {}
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}

    found: dict[str, str] = {}
    for line in text.splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in ENV_KEYS:
            found[key] = value
    return found


def parse_env_line(line: str) -> tuple[str, str] | None:
    if len(line) > 1024:
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    key, separator, value = stripped.partition("=")
    if not separator:
        return None
    key = key.strip()
    if not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value


def usable_key(value: str) -> str:
    key = value.strip()
    if not key or len(key) > 512:
        return ""
    if any(ord(char) < 33 or ord(char) == 127 for char in key):
        return ""
    return key


def usable_model_name(value: str) -> str:
    name = value.strip() or DEFAULT_MODEL_NAME
    if len(name) > 128 or not name[0].isalnum():
        return ""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/-")
    if any(char not in allowed for char in name):
        return ""
    return name


def parse_model_base(value: str) -> str | None:
    candidate = value.strip()
    if not candidate or any(ord(char) < 33 or ord(char) == 127 for char in candidate):
        return None
    try:
        parsed = urllib.parse.urlparse(candidate)
        port = parsed.port
    except ValueError:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    host = parsed.hostname
    if host is None:
        return None
    loopback = host.lower() in LOOPBACK_HOSTS
    if parsed.scheme == "http":
        if not loopback:
            return None
    elif parsed.scheme != "https":
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return candidate.rstrip("/")


def model_host(base_url: str | None) -> str:
    if not base_url:
        return ""
    hostname = urllib.parse.urlparse(base_url).hostname
    return hostname or ""
