"""Nodo guía: redacta el texto para TTS. Solo corre si Jev decidió hablar."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from agent.contract import clean_text
from agent.settings import Settings, completions_url

GUIDE_TIMEOUT_SECONDS = 4
MAX_MODEL_BYTES = 32_768
MAX_REPLY_CHARS = 500
MAX_LINES = 4

EMPTY_REPLY = (
    "Por acá no aparece ningún lugar destacado. "
    "Seguí caminando y te aviso en cuanto tenga algo cerca para contarte."
)
UNREADABLE_REPLY = (
    "Me llegaron lugares, pero no pude leer ninguno. "
    "Seguí un poco y probamos de nuevo."
)

GUIDE_SYSTEM_PROMPT = (
    "Eres un guía turístico local que acompaña al usuario. "
    "Redacta la respuesta para ser leída por un motor Text-to-Speech (TTS).\n"
    "Reglas estrictas:\n"
    "1. Escribe oraciones cortas, directas y conversacionales.\n"
    "2. Ve directo al dato interesante en la primera línea.\n"
    "3. No uses Markdown, emojis ni viñetas.\n"
    "4. Extensión máxima: 3 o 4 líneas breves."
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def write_script(
    name: str,
    category: str,
    description: str,
    settings: Settings,
    timeout: float | None = GUIDE_TIMEOUT_SECONDS,
) -> str:
    name = name.strip()
    description = description.strip()
    if settings.model_enabled and settings.model_base_url and timeout is not None:
        generated = request_guide(name, category, description, settings, settings.model_base_url, timeout)
        if generated:
            return generated
    return local_script(name, description)


def request_guide(
    name: str,
    category: str,
    description: str,
    settings: Settings,
    base_url: str,
    timeout: float = GUIDE_TIMEOUT_SECONDS,
) -> str | None:
    user = json.dumps(
        {"place": name, "category": category, "historicalContext": description},
        ensure_ascii=False,
    )
    if description:
        user += "\nUsá solo ese dato histórico."
    else:
        user += "\nNo hay dato histórico. No inventes fechas ni episodios."
    payload = {
        "model": settings.model_name,
        "temperature": 0.4,
        "max_tokens": 180,
        "messages": [
            {"role": "system", "content": GUIDE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.model_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tour-ai-agent",
        },
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_MODEL_BYTES + 1)
    except urllib.error.HTTPError as error:
        error.close()
        code = error.code if isinstance(error.code, int) else 0
        sys.stderr.write(f"modelo respondió {code}\n")
        return None
    except OSError:
        sys.stderr.write("modelo no disponible\n")
        return None
    if len(raw) > MAX_MODEL_BYTES:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    return read_model_text(parsed)


def read_model_text(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        content = "\n".join(parts)
    if not isinstance(content, str):
        return None
    text = for_speech(content)
    if len(text) < 12:
        return None
    return text


def for_speech(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        if not line or line.startswith(("```", "#")):
            continue
        lines.append(line)
        if len(lines) == MAX_LINES:
            break
    return clean_text(" ".join(lines), MAX_REPLY_CHARS)


def local_script(name: str, description: str) -> str:
    if description:
        return f"{name}. {description}"
    if name:
        return f"{name}. Está acá al lado."
    return "Hay algo cerca para mirar."
