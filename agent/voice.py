"""Nodo guía: redacta el texto para TTS. Solo corre si Jev decidió hablar."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from agent.contract import clean_text
from agent.settings import Settings, completions_url
from agent.situation import fold

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
    "- Contale qué es el lugar, como si se lo señalaras a un amigo: nombre, qué es y un dato real.\n"
    "- Cercano y natural, sin dramatismo ni discurso.\n"
    "- Nada de energías, vibras, aura, magia del lugar, 'se siente la historia', "
    "'respira historia', positivismo extremo ni frases motivacionales.\n"
    "- No digas cómo 'se siente' el lugar. Preferí hechos: qué es, cuándo, para qué sirve, qué mirar.\n"
    "- Nada de folleto, Wikipedia leída en voz alta ni robot.\n"
    "- Variá el arranque, pero siempre con el nombre exacto del lugar.\n"
    "\n"
    "Reglas:\n"
    "1. El nombre exacto del campo place tiene que aparecer tal cual en el relato. "
    "No lo cambies, no lo alargues y no uses otro nombre aunque el contexto histórico "
    "diga distinto. La persona puede ir con el teléfono guardado.\n"
    "2. Oraciones cortas y fáciles de escuchar caminando.\n"
    "3. Máximo 3 oraciones. Mejor corto que largo.\n"
    "4. Sin Markdown, emojis, viñetas ni signos de exclamación de más.\n"
    "5. Usá solo el dato histórico del mensaje. Si no hay, no inventes fechas ni episodios.\n"
    "6. Si hay zones, son partes del mismo sitio. Contá el lugar y después "
    "mencioná esas zonas en una sola oración. No digas que son lugares distintos.\n"
    "7. No saludes, no te presentes y no digas que sos una IA o un guía."
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
            return ensure_place_name(generated, name)
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
        "\nContalo con cercanía y brevedad, como un amigo que señala el lugar. "
        "Hechos concretos, sin energías ni sensaciones místicas. "
        f"Usá exactamente este nombre y ningún otro: {name}."
    )
    if description:
        user += (
            "\nUsá solo ese dato histórico. Si ahí aparece otro nombre del lugar, "
            "ignorá ese nombre y usá solo el de place."
        )
    else:
        user += "\nNo hay dato histórico. No inventes fechas ni episodios; explicá qué es el lugar con naturalidad."
    if zones:
        user += "\nLas zones son partes del mismo lugar. Mencionalas juntas al final, con naturalidad."
    payload = {
        "model": settings.model_name,
        "temperature": 0.45,
        "max_tokens": 200,
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


def mentions_place(text: str, name: str) -> bool:
    """El nombre del título tiene que aparecer tal cual (sin exigir acentos)."""
    cleaned_name = name.strip()
    if not cleaned_name:
        return True
    folded_text = fold(text)
    folded_name = fold(cleaned_name)
    return bool(folded_name) and folded_name in folded_text


def ensure_place_name(text: str, name: str) -> str:
    cleaned = text.strip()
    cleaned_name = name.strip()
    if not cleaned or not cleaned_name:
        return cleaned
    if mentions_place(cleaned, cleaned_name):
        return cleaned
    return clean_text(f"{cleaned_name}. {cleaned}", MAX_REPLY_CHARS)


def local_script(name: str, description: str, zones: list[str] | None = None) -> str:
    if description:
        base = f"{name}. {description}"
    elif name:
        base = f"{name}. Está acá cerca, por si querés acercarte a mirarlo."
    else:
        base = "Hay algo cerca que vale la pena mirar."
    zone_list = [zone for zone in (zones or []) if zone]
    if not zone_list:
        return ensure_place_name(base, name)
    if len(zone_list) == 1:
        return ensure_place_name(f"{base} También tiene {zone_list[0]}.", name)
    if len(zone_list) == 2:
        return ensure_place_name(
            f"{base} Adentro están {zone_list[0]} y {zone_list[1]}, todo en el mismo lugar.",
            name,
        )
    head = ", ".join(zone_list[:-1])
    return ensure_place_name(
        f"{base} Tiene varias zonas, como {head} y {zone_list[-1]}.",
        name,
    )
