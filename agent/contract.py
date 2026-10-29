"""Valida el cuerpo de POST /context. No persiste usuarios ni lugares."""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_NAME_CHARS = 80
MAX_CATEGORY_CHARS = 40
MAX_AREA_NAME_CHARS = 200
MAX_AREA_DESCRIPTION_CHARS = 500
MAX_PLACES = 20
MAX_AREAS = 8
MAX_DISTANCE_METERS = 100_000


@dataclass(frozen=True)
class Coordinates:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Place:
    name: str
    category: str | None
    distance_meters: float
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Area:
    name: str
    category: str
    description: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Context:
    location: Coordinates
    places: list[Place]
    received_places: int
    areas: list[Area]


class InvalidContext(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def parse_context(payload: object) -> Context:
    if not isinstance(payload, dict):
        raise InvalidContext("El cuerpo tiene que ser un objeto JSON")

    problems: list[str] = []
    location, location_problem = read_location(payload.get("location", _MISSING))
    if location_problem:
        problems.append(location_problem)

    raw_places = payload.get("pointsOfInterest", _MISSING)
    if not isinstance(raw_places, list):
        problems.append("Los lugares tienen que ser una lista")
        places: list[Place] = []
        received = 0
    else:
        places = read_places(raw_places)
        received = len(raw_places)

    if problems or location is None:
        raise InvalidContext(". ".join(problems) or "Falta la ubicación")
    return Context(
        location=location,
        places=places,
        received_places=received,
        areas=read_areas(payload.get("areas", [])),
    )


def read_location(value: object) -> tuple[Coordinates | None, str | None]:
    if value is _MISSING or value is None or not isinstance(value, dict):
        return None, "Falta la ubicación"
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    if not _is_coordinate(latitude, -90, 90) or not _is_coordinate(longitude, -180, 180):
        return None, "La ubicación no es válida"
    return Coordinates(float(latitude), float(longitude)), None


def read_places(items: list[object]) -> list[Place]:
    places: list[Place] = []
    for item in items:
        if len(places) >= MAX_PLACES:
            break
        place = read_place(item)
        if place is not None:
            places.append(place)
    return places


def read_areas(value: object) -> list[Area]:
    if not isinstance(value, list):
        return []
    areas: list[Area] = []
    for item in value:
        if len(areas) >= MAX_AREAS:
            break
        area = read_area(item)
        if area is not None:
            areas.append(area)
    return areas


def read_area(value: object) -> Area | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    category = value.get("category")
    description = value.get("description")
    if not isinstance(name, str) or not isinstance(category, str) or not isinstance(description, str):
        return None
    cleaned_name = clean_text(name, MAX_AREA_NAME_CHARS)
    cleaned_category = clean_text(category, MAX_CATEGORY_CHARS)
    cleaned_description = clean_text(description, MAX_AREA_DESCRIPTION_CHARS)
    if not cleaned_name or not cleaned_category or not cleaned_description:
        return None
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    if not _is_coordinate(latitude, -90, 90) or not _is_coordinate(longitude, -180, 180):
        return None
    return Area(
        name=cleaned_name,
        category=cleaned_category,
        description=cleaned_description,
        latitude=float(latitude),
        longitude=float(longitude),
    )


def read_place(value: object) -> Place | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str):
        return None
    cleaned_name = clean_text(name, MAX_NAME_CHARS)
    if not cleaned_name:
        return None
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    distance = value.get("distanceMeters")
    if not _is_coordinate(latitude, -90, 90) or not _is_coordinate(longitude, -180, 180):
        return None
    if not _is_number(distance) or not 0 <= float(distance) <= MAX_DISTANCE_METERS:
        return None
    category = value.get("category", None)
    cleaned_category = clean_text(category, MAX_CATEGORY_CHARS) if isinstance(category, str) else None
    return Place(
        name=cleaned_name,
        category=cleaned_category or None,
        distance_meters=float(distance),
        latitude=float(latitude),
        longitude=float(longitude),
    )


def clean_text(value: str, limit: int) -> str:
    chars: list[str] = []
    for char in value:
        if char.isprintable() and char != "\x7f":
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())[:limit].strip()


def _is_coordinate(value: object, low: float, high: float) -> bool:
    return _is_number(value) and low <= float(value) <= high


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class _Missing:
    pass


_MISSING = _Missing()
