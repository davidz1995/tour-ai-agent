"""Paso 1: qué tiene el usuario cerca, en qué dirección y a qué distancia."""

from __future__ import annotations

import math
import unicodedata

from agent.contract import Context, Coordinates, Place

SPOKEN_PLACES = 3
KNOWN_PLACES = 5


class Mention:
    def __init__(self, place: Place, direction: str | None) -> None:
        self.place = place
        self.direction = direction


class Situation:
    def __init__(self, origin: Coordinates, mentions: list[Mention], unreadable: bool) -> None:
        self.origin = origin
        self.mentions = mentions
        self.unreadable = unreadable

    @property
    def spoken(self) -> list[Mention]:
        return self.mentions[:SPOKEN_PLACES]


def analyze_situation(context: Context) -> Situation:
    if not context.places:
        return Situation(context.location, [], unreadable=context.received_places > 0)
    ordered = sorted(context.places, key=lambda place: (place.distance_meters, place.name))[:KNOWN_PLACES]
    mentions = [Mention(place, cardinal(context.location, place)) for place in ordered]
    return Situation(context.location, mentions, unreadable=False)


def cardinal(origin: Coordinates, place: Place) -> str | None:
    north_m = (place.latitude - origin.latitude) * 111_320
    east_m = (place.longitude - origin.longitude) * 111_320 * math.cos(math.radians(origin.latitude))
    if math.hypot(north_m, east_m) < 15:
        return None
    angle = math.degrees(math.atan2(east_m, north_m))
    if angle < 0:
        angle += 360
    names = ("norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste")
    return names[int((angle + 22.5) // 45) % 8]


def distance_phrase(meters: float) -> str:
    rounded = int(round(meters))
    if rounded < 20:
        return "unos pocos metros"
    if rounded < 1000:
        return f"{rounded} metros"
    kilometers = rounded / 1000
    shown = f"{kilometers:.1f}".replace(".", ",")
    if shown.endswith(",0"):
        shown = shown[:-2]
    unit = "kilómetro" if shown == "1" else "kilómetros"
    return f"{shown} {unit}"


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()
