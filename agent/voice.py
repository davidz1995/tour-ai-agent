"""Nodo guía: redacta el texto para TTS. Solo corre si Jev decidió hablar."""

from __future__ import annotations

import json
import re
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
    "- Cercano y natural, de vos, sin dramatismo ni discurso.\n"
    "- Nada de energías, vibras, aura, magia del lugar, 'se siente la historia', "
    "'respira historia', positivismo extremo ni frases motivacionales.\n"
    "- No digas cómo 'se siente' el lugar. Preferí hechos: qué es, cuándo, para qué sirve, qué mirar.\n"
    "- Nada de folleto, Wikipedia leída en voz alta ni robot.\n"
    "- Variá el arranque. A veces 'Mirá'; otras 'Fijate', 'Ahí tenés', 'Acá está', "
    "'Sabés qué', 'Justo acá' o 'De este lado'.\n"
    "- No arranques siempre con la misma palabra. Nada poético ni rebuscado.\n"
    "- Siempre con el nombre exacto del lugar.\n"
    "\n"
    "Reglas:\n"
    "1. El nombre exacto del campo place tiene que aparecer tal cual. "
    "No lo cambies ni lo alargues. Si el dato histórico nombra otro lugar, "
    "ese otro nombre se queda en esos hechos: no se lo pases a place ni reemplaces uno por el otro. "
    "Una tumba, una bóveda o una persona no es el cementerio, la iglesia o el museo que la contiene.\n"
    "2. Oraciones cortas y fáciles de escuchar caminando.\n"
    "3. Máximo 3 oraciones. Mejor corto que largo.\n"
    "4. Sin Markdown, emojis, viñetas ni signos de exclamación de más.\n"
    "5. Usá solo un hecho que el dato histórico diga de ese mismo lugar. "
    "Si no hay dato sobre place, no inventes fechas, episodios ni qué tipo de lugar es. "
    "La categoría del mapa no alcanza para afirmarlo.\n"
    "6. No digas el barrio, el barrio vecino ni la ciudad donde queda, "
    "salvo que el dato histórico lo diga. La persona ya está ahí. "
    "No ubiques Plaza Francia en Belgrano ni El Ateneo en San Telmo.\n"
    "7. Si hay zones, están en el mismo predio. place es el lugar principal y cada zona "
    "conserva su propio nombre. No conviertas una zona en el lugar ni el lugar en la zona.\n"
    "8. No saludes, no te presentes y no digas que sos una IA o un guía."
)

# Arranques cortos, de amigo. Se rota por lugar para no repetir siempre "Mirá".
FRIEND_OPENERS = (
    "Mirá",
    "Fijate",
    "Ahí tenés",
    "Acá está",
    "Sabés qué",
    "Justo acá",
    "De este lado",
)


def friend_opener(name: str) -> str:
    folded = fold(name)
    if not folded:
        return FRIEND_OPENERS[0]
    index = sum(ord(char) for char in folded) % len(FRIEND_OPENERS)
    return FRIEND_OPENERS[index]


# Barrios que el modelo suele inventar. El más largo va primero para no cortar "san telmo" en "telmo".
_BARRIOS = (
    "villa general mitre",
    "parque avellaneda",
    "parque chacabuco",
    "parque patricios",
    "villa del parque",
    "villa pueyrredon",
    "villa santa rita",
    "villa riachuelo",
    "puerto madero",
    "nueva pompeya",
    "villa urquiza",
    "villa ortuzar",
    "villa devoto",
    "villa soldati",
    "villa lugano",
    "villa crespo",
    "villa martelli",
    "villa adelina",
    "vicente lopez",
    "monte castro",
    "la paternal",
    "barrio norte",
    "san cristobal",
    "san nicolas",
    "velez sarsfield",
    "san telmo",
    "la lucila",
    "villa luro",
    "villa real",
    "la boca",
    "agronomia",
    "almagro",
    "balvanera",
    "barracas",
    "belgrano",
    "boedo",
    "caballito",
    "carapachay",
    "chacarita",
    "coghlan",
    "colegiales",
    "constitucion",
    "floresta",
    "florida",
    "flores",
    "liniers",
    "mataderos",
    "monserrat",
    "montserrat",
    "munro",
    "nunez",
    "olivos",
    "palermo",
    "parque chas",
    "recoleta",
    "retiro",
    "saavedra",
    "versalles",
)
_BARRIO_CUE = re.compile(
    r"(?:barrio(?:\s+de)?|(?:en|de|del))\s+(?:(?:la|el)\s+)?("
    + "|".join(re.escape(name) for name in sorted(_BARRIOS, key=len, reverse=True))
    + r")(?![a-z0-9])"
)


def located_barrios(text: str) -> set[str]:
    """Barrios que el texto afirma, como 'de Belgrano' o 'en San Telmo'."""
    return {match.group(1) for match in _BARRIO_CUE.finditer(fold(text))}


def location_is_grounded(text: str, name: str, description: str) -> bool:
    """El relato no puede sumar un barrio que el dato y el nombre no dicen."""
    return located_barrios(text) <= located_barrios(f"{name}. {description}")


# La categoría del mapa a veces nombra el predio, no el pin.
_CATEGORY_WORDS = {
    "cemetery": frozenset({"cementerio"}),
    "church": frozenset({"iglesia", "catedral", "basilica", "parroquia", "capilla"}),
    "place_of_worship": frozenset({"iglesia", "catedral", "basilica", "parroquia", "templo", "mezquita", "sinagoga"}),
    "museum": frozenset({"museo", "museum"}),
    "park": frozenset({"parque"}),
    "plaza": frozenset({"plaza"}),
}


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
    category = category_for_speech(name, category)
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
            spoken = ensure_place_name(generated, name)
            if location_is_grounded(spoken, name, description):
                return spoken
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
    payload_user: dict[str, object] = {"place": name, "historicalContext": description}
    if category:
        payload_user["category"] = category
    if zones:
        payload_user["zones"] = zones
    user = json.dumps(payload_user, ensure_ascii=False)
    opener = friend_opener(name)
    user += (
        "\nContalo corto, como un amigo que camina al lado. "
        f"Esta vez arrancá cerca de «{opener}», o con otra frase igual de simple. "
        "No uses siempre «mirá». Nada poético ni rebuscado. "
        "Hechos concretos, sin energías ni sensaciones místicas. "
        "No digas el barrio ni la ciudad si el dato histórico no lo dice. "
        f"El nombre de place tiene que aparecer tal cual: {name}. No lo reemplaces por otro."
    )
    if description:
        user += (
            "\nEse dato se queda con el sitio del que habla. "
            "Si nombra otro lugar, no digas que place es ese lugar."
        )
    else:
        user += (
            "\nNo hay dato histórico. No inventes fechas, episodios ni qué tipo de lugar es. "
            "No afirmes que place es un cementerio, una iglesia o un museo si el nombre no lo dice."
        )
    if zones:
        user += (
            "\nCada zona conserva su nombre. No la conviertas en el lugar principal ni al revés."
        )
    payload = {
        "model": settings.model_name,
        "temperature": 0.2,
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


def category_for_speech(name: str, category: str) -> str:
    """No le decimos al relato que una tumba es un cementerio porque el mapa marcó el predio."""
    cleaned = category.strip()
    words = _CATEGORY_WORDS.get(cleaned.lower().replace(" ", "_"))
    if not words:
        return cleaned
    tokens = set(fold(name).replace("-", " ").split())
    if tokens & words:
        return cleaned
    return ""


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
