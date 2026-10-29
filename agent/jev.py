"""Cliente de Jev (System One). No usa system prompt: manda state y questions noul."""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from typing import Literal, TypedDict

from agent.settings import Settings
from agent.situation import fold

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_TIMEOUT_SECONDS = 1.5
QUESTION_ID = "should_speak"
MAX_RESPONSE_BYTES = 32_768

INSTRUCTIONS = (
    "¿Es este un punto de interés turístico o cultural sobre el cual el usuario aún no ha escuchado? "
    "Preferí aceptar lugares visitables aunque no sean famosos."
)
CRITERIA_TRUE = (
    "El lugar es o parece un sitio para visitar o conocer: monumento, museo, plaza, iglesia, templo, "
    "parque, teatro, estadio, universidad, atracción turística, arquitectura notable, "
    "sitio histórico, biblioteca, cementerio histórico o similar, "
    "o `in_curated_base` es verdadero porque la guía lo eligió como sitio para contar, "
    "Y NO aparece en la lista de recent_history. "
    "Si la categoría o el nombre sugieren un lugar cultural o turístico, respondé true aunque no sea icónico. "
    "Si `poi_category` es `custom-area`, es una zona elegida para contar: respondé true salvo que ya esté en recent_history."
)
CRITERIA_FALSE = (
    "El lugar es claramente un negocio de uso cotidiano: farmacia, cajero, kiosco, supermercado, "
    "banco, estacionamiento, lavandería, consultorio, hotel genérico, bar o restaurante común "
    "sin valor turístico, Y `in_curated_base` es falso; O ya aparece en el recent_history. "
    "Ante la duda entre un negocio común y un sitio cultural, preferí true."
)

TOURIST_CATEGORIES = frozenset({
    "amusement_park",
    "aquarium",
    "art_gallery",
    "cemetery",
    "church",
    "city_hall",
    "courthouse",
    "hindu_temple",
    "historical_landmark",
    "library",
    "monument",
    "mosque",
    "museum",
    "park",
    "place_of_worship",
    "plaza",
    "stadium",
    "custom-area",
    "synagogue",
    "tourist_attraction",
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
})
COMMON_TOKENS = frozenset({"farmacia", "pharmacy", "cajero", "kiosco", "kiosk", "atm"})
TOURIST_TOKENS = frozenset({
    "museo",
    "catedral",
    "iglesia",
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
    "reserva",
})


class JevState(TypedDict):
    poi_name: str
    poi_category: str
    in_curated_base: bool
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


def build_request(state: JevState) -> JevRequest:
    """La API identifica cada pregunta por su clave. No es un array."""
    return {
        "model": JEV_MODEL,
        "state": {
            "poi_name": state["poi_name"],
            "poi_category": state["poi_category"],
            "in_curated_base": state["in_curated_base"],
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
    """Misma decisión que el criterio noul, para cuando no hay clave de Jev."""
    if already_heard(state["poi_name"], state["recent_history"]):
        return 0.04
    if state["in_curated_base"]:
        return 0.95
    tokens = set(fold(state["poi_name"]).replace("-", " ").split())
    if tokens & COMMON_TOKENS:
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


def clean_jev_state(name: str, category: str, history: list[str], in_curated_base: bool = False) -> JevState:
    names = [item.strip() for item in history if isinstance(item, str) and item.strip()]
    return {
        "poi_name": name.strip(),
        "poi_category": category.strip(),
        "in_curated_base": in_curated_base,
        "recent_history": names[-5:],
    }
