"""Verificación visual de una visita. No arma el relato ni entrega sello o XP."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from agent.contract import MAX_CATEGORY_CHARS, MAX_NAME_CHARS, clean_text
from agent.deadline import Deadline
from agent.settings import Settings
from agent.visit_embedding import ReferenceCatalog, embed_visit, embedding_signal, load_catalog
from agent.visit_jev import JEV_VISIT_TIMEOUT_SECONDS, ask_visit

VISIT_BUDGET_SECONDS = 12.0
MIN_SIDE = 280
MAX_SIDE = 1568
MAX_REQUEST_PART_BYTES = 16_384
MULTIPART_ERROR = "El cuerpo tiene que ser multipart"
JPEG_ERROR = "La foto tiene que ser un JPEG"
SIZE_ERROR = "La foto tiene que medir entre 280 y 1568 px por lado"

_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_STANDALONE = frozenset({0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8})
_FIELDS = (
    ("placeId", "place_id", MAX_NAME_CHARS, "Falta el lugar"),
    ("name", "name", MAX_NAME_CHARS, "Falta el nombre"),
    ("category", "category", MAX_CATEGORY_CHARS, "Falta la categoría"),
    ("city", "city", MAX_NAME_CHARS, "Falta la ciudad"),
    ("country", "country", MAX_NAME_CHARS, "Falta el país"),
)


class InvalidVisit(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class VisitUpload:
    request: bytes
    photo: bytes


@dataclass(frozen=True)
class VisitPlace:
    place_id: str
    name: str
    category: str
    city: str
    country: str


@dataclass(frozen=True)
class _Part:
    name: str
    content_type: str | None
    body: bytes


def verify_response(
    content_type: str | None,
    body: bytes,
    settings: Settings,
    catalog: ReferenceCatalog | None = None,
    embed: Callable[[bytes], tuple[float, ...] | None] = embed_visit,
    ask: Callable[..., tuple[float, float] | None] = ask_visit,
) -> tuple[int, dict[str, object]]:
    try:
        upload = parse_visit_multipart(content_type, body)
        place = parse_visit_fields(upload.request)
    except InvalidVisit as error:
        return 400, {"message": error.message}
    photo_error = validate_visit_jpeg(upload.photo)
    if photo_error is not None:
        return 400, {"message": photo_error}

    views = load_catalog() if catalog is None else catalog
    deadline = Deadline(VISIT_BUDGET_SECONDS)
    embedding = embedding_signal(place.place_id, upload.photo, views, embed)
    probs = None
    if settings.jev_key:
        timeout = deadline.timeout_for(JEV_VISIT_TIMEOUT_SECONDS)
        if timeout is not None:
            probs = ask(
                upload.photo,
                place_id=place.place_id,
                name=place.name,
                category=place.category,
                city=place.city,
                country=place.country,
                settings=settings,
                timeout=timeout,
            )
    return 200, {"embedding": embedding, "jev": _jev_signal(probs)}


def parse_visit_multipart(content_type: str | None, body: bytes) -> VisitUpload:
    parsed = _split_header(content_type or "")
    if parsed is None:
        raise InvalidVisit(MULTIPART_ERROR)
    mime, params = parsed
    boundary = params.get("boundary")
    if mime != "multipart/form-data" or not boundary or len(boundary) > 70:
        raise InvalidVisit(MULTIPART_ERROR)
    if any(ord(char) < 33 or ord(char) > 126 for char in boundary):
        raise InvalidVisit(MULTIPART_ERROR)
    parts = _parts(body, boundary)
    by_name: dict[str, _Part] = {}
    for part in parts:
        if part.name in by_name:
            raise InvalidVisit("El pedido tiene partes de más")
        by_name[part.name] = part
    if "request" not in by_name:
        raise InvalidVisit("Falta el pedido")
    if "visit" not in by_name:
        raise InvalidVisit("Falta la foto")
    if len(by_name) != 2:
        raise InvalidVisit("El pedido tiene partes de más")
    request = by_name["request"]
    photo = by_name["visit"]
    if len(request.body) > MAX_REQUEST_PART_BYTES or not request.body:
        raise InvalidVisit("El pedido es demasiado grande" if request.body else "Falta el pedido")
    if not _is_json_type(request.content_type):
        raise InvalidVisit("La parte request tiene que ser JSON")
    if photo.content_type is not None and not _is_jpeg_type(photo.content_type):
        raise InvalidVisit(JPEG_ERROR)
    return VisitUpload(request.body, photo.body)


def parse_visit_fields(raw: bytes) -> VisitPlace:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise InvalidVisit("El pedido tiene que ser un objeto JSON") from None
    if not isinstance(payload, dict):
        raise InvalidVisit("El pedido tiene que ser un objeto JSON")
    values: dict[str, str] = {}
    problems: list[str] = []
    for key, field, limit, missing in _FIELDS:
        text = _required_text(payload.get(key), limit)
        if text is None:
            problems.append(missing)
        else:
            values[field] = text
    if problems:
        raise InvalidVisit(". ".join(problems))
    return VisitPlace(
        place_id=values["place_id"],
        name=values["name"],
        category=values["category"],
        city=values["city"],
        country=values["country"],
    )


def validate_visit_jpeg(photo: bytes) -> str | None:
    size = jpeg_size(photo)
    if size is None:
        return JPEG_ERROR
    width, height = size
    if width < MIN_SIDE or height < MIN_SIDE or width > MAX_SIDE or height > MAX_SIDE:
        return SIZE_ERROR
    return None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    limit = len(data)
    while index < limit:
        if data[index] != 0xFF:
            return None
        while index < limit and data[index] == 0xFF:
            index += 1
        if index >= limit:
            return None
        marker = data[index]
        index += 1
        if marker == 0xD9:
            return None
        if marker in _STANDALONE or marker == 0x00:
            continue
        if index + 2 > limit:
            return None
        length = int.from_bytes(data[index : index + 2], "big")
        if length < 2 or index + length > limit:
            return None
        if marker in _SOF:
            if length < 7:
                return None
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if width <= 0 or height <= 0:
                return None
            return width, height
        if marker == 0xDA:
            return None
        index += length
    return None


def _required_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    return clean_text(value, limit) or None


def _jev_signal(probs: tuple[float, float] | None) -> dict[str, object]:
    if probs is None:
        return {"available": False}
    matches, screen = probs
    return {"available": True, "matchesPlace": matches, "looksLikeScreen": screen}


def _parts(body: bytes, boundary: str) -> list[_Part]:
    token = b"--" + boundary.encode("ascii")
    if body.startswith(token):
        rest = body
    else:
        marker = b"\r\n" + token
        at = body.find(marker)
        if at < 0:
            raise InvalidVisit(MULTIPART_ERROR)
        rest = body[at + 2 :]
    found: list[_Part] = []
    while len(found) <= 2:
        if rest.startswith(token + b"--"):
            return found
        if not rest.startswith(token + b"\r\n"):
            raise InvalidVisit(MULTIPART_ERROR)
        rest = rest[len(token) + 2 :]
        end = rest.find(b"\r\n\r\n")
        if end < 0 or end > 8_192:
            raise InvalidVisit(MULTIPART_ERROR)
        part = _part(rest[:end], rest[end + 4 :], token)
        if part is None:
            raise InvalidVisit(MULTIPART_ERROR)
        payload, rest = part
        found.append(payload)
    raise InvalidVisit("El pedido tiene partes de más")


def _part(header_blob: bytes, rest: bytes, token: bytes) -> tuple[_Part, bytes] | None:
    try:
        text = header_blob.decode("utf-8")
    except UnicodeError:
        return None
    name: str | None = None
    content_type: str | None = None
    seen: set[str] = set()
    for line in text.split("\r\n"):
        if ":" not in line:
            return None
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not key or key in seen or len(value) > 1024:
            return None
        seen.add(key)
        if key == "content-disposition":
            name = _disposition_name(value)
            if name is None:
                return None
        elif key == "content-type":
            content_type = value or None
    if name is None:
        return None
    marker = b"\r\n" + token
    token_at = rest.find(marker)
    if token_at < 0:
        return None
    return _Part(name, content_type, rest[:token_at]), rest[token_at + 2 :]


def _disposition_name(value: str) -> str | None:
    parsed = _split_header(value)
    if parsed is None:
        return None
    mime, params = parsed
    name = params.get("name", "")
    if mime != "form-data" or not name or len(name) > 64:
        return None
    return name


def _split_header(value: str) -> tuple[str, dict[str, str]] | None:
    if not value or "\r" in value or "\n" in value or len(value) > 1500:
        return None
    pieces: list[str] = []
    current: list[str] = []
    quoted = False
    for char in value:
        if char == '"':
            quoted = not quoted
            current.append(char)
        elif char == ";" and not quoted:
            pieces.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quoted:
        return None
    pieces.append("".join(current).strip())
    if not pieces or not pieces[0]:
        return None
    params: dict[str, str] = {}
    for piece in pieces[1:]:
        if not piece:
            continue
        key, separator, raw = piece.partition("=")
        if not separator:
            return None
        key = key.strip().lower()
        item = _unquote(raw)
        if not key or item is None or key in params:
            return None
        params[key] = item
    return pieces[0].lower(), params


def _unquote(raw: str) -> str | None:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if value is None or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def _is_json_type(value: str | None) -> bool:
    if not value:
        return False
    parsed = _split_header(value)
    if parsed is None:
        return False
    mime, params = parsed
    if mime != "application/json":
        return False
    charset = params.get("charset", "utf-8").lower().replace("_", "").replace("-", "")
    return charset == "utf8"


def _is_jpeg_type(value: str) -> bool:
    parsed = _split_header(value)
    if parsed is None:
        return False
    mime, _params = parsed
    return mime in {"image/jpeg", "image/jpg"}
