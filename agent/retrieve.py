"""Base histórica local, separada por ciudad. El texto recuperado es el único contexto que se puede contar."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from agent.contract import Place
from agent.situation import Situation, fold

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MAX_MATCH_METERS = 450
CITY_MARGIN_DEGREES = 0.01
MAX_PLACE_FILE_BYTES = 65_536
EARTH_RADIUS_METERS = 6_371_000
STOPWORDS = frozenset({"de", "del", "la", "las", "el", "los", "y", "en", "the", "of"})
_TOKEN = re.compile(r"[a-z0-9]+")
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOCK = threading.Lock()
_INDEX: HistoryIndex | None = None


@dataclass(frozen=True)
class Passage:
    place_name: str
    title: str
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    city: str
    title: str
    aliases: tuple[str, ...]
    latitude: float
    longitude: float
    text: str


@dataclass(frozen=True)
class City:
    id: str
    name: str
    south: float
    west: float
    north: float
    east: float

    def covers(self, latitude: float, longitude: float, margin: float = CITY_MARGIN_DEGREES) -> bool:
        return (
            self.south - margin <= latitude <= self.north + margin
            and self.west - margin <= longitude <= self.east + margin
        )


class HistoryIndex:
    def __init__(self, cities: list[City], documents: list[Document]) -> None:
        self.cities = cities
        self.by_city: dict[str, list[Document]] = {city.id: [] for city in cities}
        for document in documents:
            self.by_city.setdefault(document.city, []).append(document)

    @property
    def size(self) -> int:
        return sum(len(documents) for documents in self.by_city.values())

    def counts(self) -> dict[str, int]:
        return {city.id: len(self.by_city.get(city.id, [])) for city in self.cities}

    def cities_near(self, latitude: float, longitude: float) -> list[str]:
        return [city.id for city in self.cities if city.covers(latitude, longitude)]

    def search(self, place: Place) -> Document | None:
        best: Document | None = None
        best_rank: tuple[int, float] | None = None
        for city_id in self.cities_near(place.latitude, place.longitude):
            for document in self.by_city.get(city_id, []):
                score = name_score(place.name, document.aliases)
                if score <= 0:
                    continue
                distance = distance_meters(place.latitude, place.longitude, document.latitude, document.longitude)
                if distance > MAX_MATCH_METERS:
                    continue
                rank = (score, -distance)
                if best_rank is None or rank > best_rank:
                    best = document
                    best_rank = rank
        return best


def search_history(situation: Situation) -> list[Passage]:
    index = load_index()
    passages: list[Passage] = []
    used: set[str] = set()
    for mention in situation.mentions:
        document = index.search(mention.place)
        if document is None or document.id in used:
            continue
        used.add(document.id)
        passages.append(Passage(mention.place.name, document.title, document.text))
    return passages


def load_index() -> HistoryIndex:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = read_knowledge(KNOWLEDGE_DIR)
        return _INDEX


def read_knowledge(base: Path) -> HistoryIndex:
    cities = read_cities(base / "cities.json")
    documents: list[Document] = []
    places_dir = base / "places"
    for city in cities:
        documents.extend(read_city_places(places_dir, city.id))
    return HistoryIndex(cities, documents)


def read_cities(path: Path) -> list[City]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        sys.stderr.write("no se pudo leer cities.json\n")
        return []
    if not isinstance(payload, list):
        return []
    cities: list[City] = []
    seen: set[str] = set()
    for item in payload:
        city = read_city(item)
        if city is None or city.id in seen:
            continue
        seen.add(city.id)
        cities.append(city)
    return cities


def read_city(value: object) -> City | None:
    if not isinstance(value, dict):
        return None
    identifier = value.get("id")
    name = value.get("name")
    bbox = value.get("bbox")
    if not isinstance(identifier, str) or not _SLUG.match(identifier):
        return None
    if not isinstance(name, str) or not name.strip() or not isinstance(bbox, dict):
        return None
    edges = [bbox.get(key) for key in ("south", "west", "north", "east")]
    if not all(_is_number(edge) for edge in edges):
        return None
    south, west, north, east = (float(edge) for edge in edges)
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        return None
    return City(identifier, name.strip(), south, west, north, east)


def read_city_places(places_dir: Path, city_id: str) -> list[Document]:
    folder = places_dir / city_id
    try:
        if not folder.is_dir() or folder.is_symlink():
            return []
        files = sorted(folder.glob("*.json"))
    except OSError:
        return []
    documents: list[Document] = []
    seen: set[str] = set()
    for path in files:
        document = read_place_file(path, city_id)
        if document is None:
            sys.stderr.write(f"lugar inválido: {city_id}/{path.name}\n")
            continue
        if document.id in seen:
            continue
        seen.add(document.id)
        documents.append(document)
    return documents


def read_place_file(path: Path, city_id: str) -> Document | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PLACE_FILE_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return read_document(payload, city_id)


def read_document(value: object, city_id: str) -> Document | None:
    if not isinstance(value, dict):
        return None
    title = value.get("title")
    text = value.get("text")
    identifier = value.get("id")
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    aliases = value.get("aliases")
    if not all(isinstance(item, str) and item.strip() for item in (title, text, identifier)):
        return None
    if not _is_number(latitude) or not _is_number(longitude):
        return None
    if not -90 <= float(latitude) <= 90 or not -180 <= float(longitude) <= 180:
        return None
    if not isinstance(aliases, list) or not aliases:
        return None
    names = [alias.strip() for alias in aliases if isinstance(alias, str) and alias.strip()]
    if not names:
        return None
    return Document(
        id=f"{city_id}/{identifier.strip()}",
        city=city_id,
        title=title.strip(),
        aliases=tuple(names),
        latitude=float(latitude),
        longitude=float(longitude),
        text=" ".join(text.split()),
    )


def name_score(place_name: str, aliases: tuple[str, ...]) -> int:
    place_tokens = set(tokens(place_name))
    if not place_tokens:
        return 0
    best = 0
    for alias in aliases:
        alias_tokens = set(tokens(alias))
        if alias_tokens and alias_tokens <= place_tokens:
            best = max(best, len(alias_tokens))
    return best


def tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(fold(text)) if token not in STOPWORDS and len(token) >= 3]


def distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    haversine = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(haversine))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
