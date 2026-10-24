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
MAX_REPLY_CHARS = 480
MAX_LINES = 4

EMPTY_REPLY = (
    "Por acá no veo nada especial para contarte ahora. "
    "Si seguís un poco, te aviso apenas aparezca algo lindo."
)
UNREADABLE_REPLY = (
    "Me llegaron lugares, pero no pude leer bien ninguno. "
    "Demos unos pasos más y lo intentamos de nuevo."
)

GUIDE_SYSTEM_PROMPT = (
    "Sos un amigo o un familiar cercano caminando al lado de la persona. "
    "Hablás en español rioplatense, de vos, con calidez tranquila. "
    "El texto se lee en voz alta.\n"
    "\n"
    "Tono:\n"
    "- Como si le mostraras el lugar a alguien querido, sin discurso ni dramatismo.\n"
    "- Un dato verdadero y una sensación breve alcanzan: por qué vale mirarlo, "
    "qué se siente estar ahí.\n"
    "- Cercano, nunca empalagoso. Nada de 'mi amor', 'tesoro', preguntas íntimas, "
    "reflexiones profundas ni consejos de vida.\n"
    "- Nada de folleto, Wikipedia leída en voz alta ni robot. "
    "No empieces siempre con 'Estás frente a', 'Acá tenés', 'Este es' o 'Mirá'.\n"
    "- Variá el arranque. A veces el nombre, a veces el dato, a veces la sensación.\n"
    "\n"
    "Reglas:\n"
    "1. Oraciones cortas y fáciles de escuchar caminando.\n"
    "2. Máximo 3 oraciones. Mejor corto que largo.\n"
    "3. Sin Markdown, emojis, viñetas ni signos de exclamación de más.\n"
    "4. Usá solo el dato histórico del mensaje. Si no hay, no inventes fechas ni episodios.\n"
    "5. Si hay zones, son partes del mismo sitio. Contá el lugar y después "
    "mencioná esas zonas en una sola oración. No digas que son lugares distintos.\n"
    "6. No saludes, no te presentes y no digas que sos una IA o un guía."
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
    zones: tuple[str, ...] | list[str] = (),
) -> str:
    name = name.strip()
    description = description.strip()
    zone_list = [zone.strip() for zone in zones if isinstance(zone, str) and zone.strip()]
    if settings.model_enabled and settings.model_base_url and timeout is not None:
        generated = request_guide(
            name,
            category,
            description,
            settings,
            settings.model_base_url,
            timeout,
            zone_list,
        )
        if generated:
            return generated
    return local_script(name, description, zone_list)


def request_guide(
    name: str,
    category: str,
    description: str,
    settings: Settings,
    base_url: str,
    timeout: float = GUIDE_TIMEOUT_SECONDS,
    zones: list[str] | None = None,
) -> str | None:
    payload_user = {
        "place": name,
        "category": category,
        "historicalContext": description,
    }
    if zones:
        payload_user["zones"] = zones
    user = json.dumps(payload_user, ensure_ascii=False)
    user += (
        "\nContalo con cercanía y brevedad, como si estuvieras caminando al lado. "
        "Un dato verdadero y un toque humano alcanzan. Nada solemne."
    )
    if description:
        user += "\nUsá solo ese dato histórico."
    else:
        user += "\nNo hay dato histórico. No inventes fechas ni episodios; invitá a mirar el lugar con calidez."
    if zones:
        user += "\nLas zones son partes del mismo lugar. Mencionalas juntas al final, con naturalidad."
    payload = {
        "model": settings.model_name,
        "temperature": 0.7,
        "max_tokens": 220,
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


def local_script(name: str, description: str, zones: list[str] | None = None) -> str:
    if description:
        base = f"Si mirás para allá, está {name}. {description} Vale la pena parar un segundo."
    elif name:
        base = f"Mirá, acá cerca tenés {name}. Si querés, demos una vuelta y lo miramos con calma."
    else:
        base = "Hay algo cerca que vale la pena mirar. Si querés, nos acercamos."
    zone_list = [zone for zone in (zones or []) if zone]
    if not zone_list:
        return base
    if len(zone_list) == 1:
        return f"{base} También tiene {zone_list[0]}, por si te dan ganas de quedarte un rato."
    if len(zone_list) == 2:
        return (
            f"{base} Adentro vas a encontrar {zone_list[0]} y {zone_list[1]}, "
            "todo en el mismo lugar."
        )
    head = ", ".join(zone_list[:-1])
    return (
        f"{base} Tiene varias zonas para disfrutar, como {head} y {zone_list[-1]}, "
        "sin salir del mismo sitio."
    )
