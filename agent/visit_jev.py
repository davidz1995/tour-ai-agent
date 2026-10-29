"""Jev con la foto de la visita. No decide el relato ni simula una probabilidad."""

from __future__ import annotations

import json
import math
import secrets
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

from agent.jev import JEV_MODEL, JEV_URL
from agent.settings import Settings

JEV_VISIT_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 32_768
MATCHES_PLACE = "matches_place"
LOOKS_LIKE_SCREEN = "looks_like_screen"
IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"

MATCHES_INSTRUCTIONS = (
    "¿La foto en `photo` es de ese lugar? "
    "Usá el nombre `place_name`, la categoría `place_category` y la ciudad `city`."
)
MATCHES_TRUE = "La foto muestra ese lugar: se reconoce el sitio nombrado, de esa categoría y en esa ciudad."
MATCHES_FALSE = "La foto es de otro sitio, de otra ciudad, o no se puede afirmar que sea ese lugar."
SCREEN_INSTRUCTIONS = "¿La foto en `photo` parece una foto de una pantalla o de una impresión?"
SCREEN_TRUE = (
    "Se ve una pantalla o una foto impresa: el marco de un monitor, un teléfono, un televisor, "
    "moiré, píxeles, brillo de vidrio o el borde del papel."
)
SCREEN_FALSE = "Es una foto tomada en el lugar mismo, no de una pantalla ni de una impresión."


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def build_visit_request(place_id: str, name: str, category: str, city: str, country: str) -> dict[str, Any]:
    reserved = "\n".join(
        (
            place_id,
            name,
            category,
            city,
            country,
            MATCHES_INSTRUCTIONS,
            MATCHES_TRUE,
            MATCHES_FALSE,
            SCREEN_INSTRUCTIONS,
            SCREEN_TRUE,
            SCREEN_FALSE,
        )
    )
    ref = _photo_reference(reserved)
    return {
        "model": JEV_MODEL,
        "state": {
            "place_id": place_id,
            "place_name": name,
            "place_category": category,
            "city": city,
            "country": country,
            "photo": ref,
        },
        "assets": {ref: {"type": "image", "part": "visit"}},
        "questions": {
            MATCHES_PLACE: {
                "type": "noul",
                "instructions": MATCHES_INSTRUCTIONS,
                "criteria": {"true": MATCHES_TRUE, "false": MATCHES_FALSE},
            },
            LOOKS_LIKE_SCREEN: {
                "type": "noul",
                "instructions": SCREEN_INSTRUCTIONS,
                "criteria": {"true": SCREEN_TRUE, "false": SCREEN_FALSE},
            },
        },
    }


def read_visit_answers(payload: object) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    matches = _probability(answers.get(MATCHES_PLACE))
    screen = _probability(answers.get(LOOKS_LIKE_SCREEN))
    if matches is None or screen is None:
        return None
    return matches, screen


def ask_visit(
    photo: bytes,
    *,
    place_id: str,
    name: str,
    category: str,
    city: str,
    country: str,
    settings: Settings,
    timeout: float,
    opener: Any = None,
) -> tuple[float, float] | None:
    if not settings.jev_key or timeout <= 0 or not photo:
        return None
    try:
        encoded = json.dumps(
            build_visit_request(place_id, name, category, city, country),
            ensure_ascii=False,
        ).encode("utf-8")
    except (UnicodeError, ValueError):
        return None
    boundary = _boundary(photo, encoded)
    if boundary is None:
        return None
    request = urllib.request.Request(
        JEV_URL,
        data=_multipart(boundary, encoded, photo),
        headers={
            "Authorization": f"Bearer {settings.jev_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "tour-ai-agent",
        },
        method="POST",
    )
    # add_header pasa el nombre por capitalize y pierde la mayúscula del medio.
    request.headers[IMAGE_HEADER] = "true"
    client = _OPENER if opener is None else opener
    try:
        with client.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        code = error.code if isinstance(error.code, int) else 0
        error.close()
        sys.stderr.write(f"jev visita respondió {code}\n")
        return None
    except OSError:
        return None
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    return read_visit_answers(parsed)


def _probability(answer: object) -> float | None:
    if not isinstance(answer, dict):
        return None
    if answer.get("type") not in (None, "noul"):
        return None
    value = answer.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _photo_reference(reserved: str) -> str:
    for _ in range(4):
        ref = uuid.uuid4().hex
        if ref not in reserved:
            return ref
    return uuid.uuid4().hex


def _boundary(photo: bytes, request: bytes) -> str | None:
    for _ in range(5):
        boundary = "visit" + secrets.token_hex(16)
        raw = boundary.encode("ascii")
        if raw not in photo and raw not in request:
            return boundary
    return None


def _multipart(boundary: str, request: bytes, photo: bytes) -> bytes:
    token = boundary.encode("ascii")
    return b"".join(
        (
            b"--" + token + b"\r\n",
            b'Content-Disposition: form-data; name="request"\r\n',
            b"Content-Type: application/json\r\n\r\n",
            request,
            b"\r\n--" + token + b"\r\n",
            b'Content-Disposition: form-data; name="visit"; filename="visit.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            photo,
            b"\r\n--" + token + b"--\r\n",
        )
    )
