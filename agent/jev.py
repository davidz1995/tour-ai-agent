"""Cliente de Jev (System One). No usa system prompt: manda state y questions noul."""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from typing import Literal, TypedDict

from agent.retrieve import looks_like_person_memorial, tokens
from agent.settings import Settings
from agent.situation import fold

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_TIMEOUT_SECONDS = 1.5
QUESTION_ID = "should_speak"
MAX_RESPONSE_BYTES = 32_768

INSTRUCTIONS = (
    "¿El pin es el lugar en sí, un sitio que la persona puede ver o visitar, y todavía no lo escuchó? "
    "No alcanza con que el mapa lo marque como interés turístico ni con que el nombre suene a un paisaje o a una actividad. "
    "Si el pin es una persona, una tumba o una bóveda, no alcanza con que se pueda mirar: "
    "tiene que ser alguien que casi cualquiera reconocería. "
    "Si `place_kind` es `museum` o `worship`, seguí esa regla y no la categoría del mapa. "
    "Si `is_duplicate` es verdadero, el mapa repitió el mismo lugar y no se vuelve a marcar."
)
CRITERIA_TRUE = (
    "El pin es el sitio: se señala, se mira o se recorre. "
    "Monumento, museo, plaza, parque, mirador o reserva que es el lugar, "
    "teatro, estadio, universidad, arquitectura notable, sitio histórico, biblioteca, "
    "cementerio histórico o similar. El nombre nombra ese sitio, no una empresa. "
    "Un lugar chico o poco famoso sigue siendo true si es el sitio, no es una persona "
    "y `place_kind` no es `worship`. "
    "Si `place_kind` es `museum`, es true: el nombre dice museo o la categoría es museum. "
    "Si `place_kind` es `worship`, es true solo cuando el edificio tiene belleza física "
    "para quien pasa: una catedral, una basílica, una fachada o un interior que la gente visita por verlo. "
    "Una iglesia católica linda cuenta. "
    "Una tumba, un sepulcro, un mausoleo, una bóveda o una persona es true solo si esa persona "
    "es una figura pública que casi cualquiera reconocería en su país. Eva Perón, Evita, es true. "
    "La bóveda de la familia Duarte también es true, porque ahí está Eva Perón. "
    "También true si `in_curated_base` es verdadero, o si `poi_category` es `custom-area`: "
    "la guía ya lo eligió para contar. En esos casos, false si ya está en recent_history "
    "o si `is_duplicate` es verdadero."
)
CRITERIA_FALSE = (
    "El pin es el local, la oficina, la casa o la sede de un negocio, aunque el mapa lo marque "
    "como interés turístico (`tourist_attraction`, `point_of_interest` u otra categoría parecida) "
    "y aunque el nombre describa una actividad, un paisaje o un animal. "
    "Agencia de viajes, operador de tours, vendedor de excursiones o experiencias: "
    "el nombre puede decir 'vista de pájaros', 'safari' o 'aventura' y el pin igual es el negocio, "
    "no el lugar donde eso ocurre. "
    "También false si es un negocio de uso cotidiano (farmacia, cajero, kiosco, supermercado, banco, "
    "estacionamiento, lavandería, consultorio, hotel genérico, bar o restaurante común) "
    "y `in_curated_base` es falso, o si ya aparece en recent_history. "
    "Si el nombre puede ser tanto un sitio público como la marca de quien vende esa experiencia, "
    "y no nombra un lugar (mirador, reserva, parque, museo, plaza, monumento), respondé false. "
    "Si `place_kind` es `worship`, false cuando es un templo o una iglesia cualquiera: "
    "un salón, una casa, un local o un edificio común de una congregación cristiana o de otra religión, "
    "aunque el mapa diga church, place_of_worship o point_of_interest. "
    "Que el nombre diga iglesia no lo vuelve turístico. "
    "False también si es la tumba, el sepulcro, el mausoleo o la bóveda de una familia o de alguien "
    "que un visitante no reconocería, aunque el mapa lo marque como sitio histórico, y `in_curated_base` es falso. "
    "False si `is_duplicate` es verdadero: es el mismo lugar en otro pin o con un nombre casi igual, "
    "aunque esté en otro punto del predio y aunque `in_curated_base` sea verdadero. "
    "Cementerio de la Recoleta y Cementerio Recoleta son el mismo sitio."
)

TOURIST_CATEGORIES = frozenset({
    "amusement_park",
    "aquarium",
    "art_gallery",
    "cemetery",
    "city_hall",
    "courthouse",
    "hindu_temple",
    "historical_landmark",
    "library",
    "monument",
    "museum",
    "park",
    "plaza",
    "stadium",
    "custom-area",
    "university",
    "zoo",
})
COMMON_CATEGORIES = frozenset({
    "atm",
    "bakery",
    "bank",
    "bar",
    "beauty_salon",
    "cafe",
    "convenience_store",
    "dentist",
    "doctor",
    "gas_station",
    "gym",
    "hospital",
    "laundry",
    "lodging",
    "night_club",
    "parking",
    "pharmacy",
    "restaurant",
    "shopping_mall",
    "store",
    "supermarket",
    "tour_agency",
    "travel_agency",
})
COMMON_TOKENS = frozenset({"farmacia", "pharmacy", "cajero", "kiosco", "kiosk", "atm"})
BUSINESS_TOKENS = frozenset({
    "agencia",
    "excursion",
    "excursiones",
    "oficina",
    "oficinas",
    "operador",
    "operadores",
    "tour",
    "tours",
    "viajes",
})
MUSEUM_TOKENS = frozenset({"museo", "museum"})
WORSHIP_CATEGORIES = frozenset({
    "church",
    "place_of_worship",
    "mosque",
    "synagogue",
    "hindu_temple",
})
WORSHIP_TOKENS = frozenset({
    "iglesia",
    "templo",
    "parroquia",
    "catedral",
    "basilica",
    "mezquita",
    "sinagoga",
    "abadia",
    "convento",
    "monasterio",
    "capilla",
})
NOTABLE_WORSHIP_TOKENS = frozenset({"catedral", "basilica", "abadia", "monasterio", "convento"})
TOURIST_TOKENS = frozenset({
    "museo",
    "catedral",
    "basilica",
    "monumento",
    "cabildo",
    "obelisco",
    "plaza",
    "teatro",
    "parque",
    "estadio",
    "universidad",
    "biblioteca",
    "cementerio",
    "palacio",
    "castillo",
    "torre",
    "puente",
    "galeria",
    "mirador",
    "reserva",
})


class JevState(TypedDict):
    poi_name: str
    poi_category: str
    place_kind: str
    in_curated_base: bool
    is_duplicate: bool
    recent_history: list[str]


class NoulCriteria(TypedDict):
    true: str
    false: str


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: str
    criteria: NoulCriteria


class JevRequest(TypedDict):
    model: str
    state: JevState
    questions: dict[str, NoulQuestion]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def name_tokens(name: str) -> set[str]:
    return set(fold(name).replace("-", " ").split())


def place_kind_for(name: str, category: str) -> str:
    """museum, worship o site. Jev usa este campo para no fiarse solo de la categoría del mapa."""
    tokens = name_tokens(name)
    category_key = category.strip().lower().replace(" ", "_")
    if category_key == "museum" or tokens & MUSEUM_TOKENS:
        return "museum"
    if category_key in WORSHIP_CATEGORIES or tokens & WORSHIP_TOKENS:
        return "worship"
    return "site"


def build_request(state: JevState) -> JevRequest:
    """La API identifica cada pregunta por su clave. No es un array."""
    return {
        "model": JEV_MODEL,
        "state": {
            "poi_name": state["poi_name"],
            "poi_category": state["poi_category"],
            "place_kind": place_kind_for(state["poi_name"], state["poi_category"]),
            "in_curated_base": state["in_curated_base"],
            "is_duplicate": bool(state.get("is_duplicate", False)),
            "recent_history": list(state["recent_history"]),
        },
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": INSTRUCTIONS,
                "criteria": {"true": CRITERIA_TRUE, "false": CRITERIA_FALSE},
            }
        },
    }


def probability_of_true(state: JevState, settings: Settings, timeout: float | None = JEV_TIMEOUT_SECONDS) -> float:
    if state.get("is_duplicate"):
        return 0.04
    if settings.jev_key and timeout is not None:
        remote = ask_remote(state, settings, timeout)
        if remote is not None:
            return remote
        sys.stderr.write("jev no disponible\n")
    return simulate_noul(state)


def ask_remote(state: JevState, settings: Settings, timeout: float = JEV_TIMEOUT_SECONDS) -> float | None:
    payload = json.dumps(build_request(state)).encode("utf-8")
    request = urllib.request.Request(
        JEV_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.jev_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tour-ai-agent",
        },
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        error.close()
        code = error.code if isinstance(error.code, int) else 0
        sys.stderr.write(f"jev respondió {code}\n")
        return None
    except OSError:
        return None
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    return read_noul(parsed)


def read_noul(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, dict) or answer.get("type") not in (None, "noul"):
        return None
    value = answer.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def simulate_noul(state: JevState) -> float:
    """Misma decisión que el criterio noul, para cuando no hay clave de Jev.

    `tourist_attraction` y `point_of_interest` no alcanzan: el mapa marca así
    también oficinas y agencias con un nombre que suena a paisaje.
    """
    if state.get("is_duplicate") or already_heard(state["poi_name"], state["recent_history"]):
        return 0.04
    if state["in_curated_base"]:
        return 0.95
    kind = place_kind_for(state["poi_name"], state["poi_category"])
    if kind == "museum":
        return 0.93
    if looks_like_person_memorial(state["poi_name"]):
        return 0.12
    if kind == "worship":
        if name_tokens(state["poi_name"]) & NOTABLE_WORSHIP_TOKENS:
            return 0.93
        return 0.18
    tokens = name_tokens(state["poi_name"])
    if tokens & (COMMON_TOKENS | BUSINESS_TOKENS):
        return 0.05
    category = state["poi_category"].strip().lower().replace(" ", "_")
    if category in COMMON_CATEGORIES:
        return 0.07
    if category in TOURIST_CATEGORIES or tokens & TOURIST_TOKENS:
        return 0.93
    return 0.32


def already_heard(name: str, history: list[str]) -> bool:
    folded = fold(name).strip()
    if not folded:
        return False
    return any(fold(item).strip() == folded for item in history)


def repeats_place(name: str, others: list[str]) -> bool:
    """El mismo sitio aunque el mapa cambie una palabra vacía o lo ponga en otro punto."""
    if already_heard(name, others):
        return True
    own = set(tokens(name))
    if len(own) < 2:
        return False
    return any(set(tokens(other)) == own for other in others)


def clean_jev_state(
    name: str,
    category: str,
    history: list[str],
    in_curated_base: bool = False,
    is_duplicate: bool = False,
) -> JevState:
    names = [item.strip() for item in history if isinstance(item, str) and item.strip()]
    cleaned_name = name.strip()
    cleaned_category = category.strip()
    return {
        "poi_name": cleaned_name,
        "poi_category": cleaned_category,
        "place_kind": place_kind_for(cleaned_name, cleaned_category),
        "in_curated_base": in_curated_base,
        "is_duplicate": is_duplicate,
        "recent_history": names[-5:],
    }
