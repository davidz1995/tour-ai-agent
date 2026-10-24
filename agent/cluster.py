"""Agrupa pines cercanos que son el mismo sitio (plaza + zonas) en un solo interés."""

from __future__ import annotations

from dataclasses import dataclass

from agent.contract import Place
from agent.retrieve import distance_meters, tokens
from agent.situation import fold

# Distancia máxima entre pines que pueden ser el mismo sitio.
CLUSTER_METERS = 140

AMENITY_TOKENS = frozenset({
    "juegos",
    "infantil",
    "infantiles",
    "mascotas",
    "mascota",
    "perros",
    "perro",
    "entrenamiento",
    "ejercicio",
    "fitness",
    "calistenia",
    "cancha",
    "skate",
    "picnic",
    "sector",
    "zona",
    "area",
    "playground",
    "dogs",
    "dog",
    "workout",
    "gym",
    "gimnasio",
    "deportiva",
    "deportivo",
    "recreativa",
    "recreativo",
})

SITE_CATEGORIES = frozenset({
    "park",
    "plaza",
    "tourist_attraction",
    "museum",
    "monument",
    "historical_landmark",
    "church",
    "place_of_worship",
    "stadium",
    "zoo",
    "aquarium",
    "art_gallery",
    "cemetery",
    "library",
    "university",
    "natural_feature",
})

ZONE_LABELS = (
    (("juegos", "infantil", "infantiles", "playground"), "juegos"),
    (("mascotas", "mascota", "perros", "perro", "dogs", "dog"), "mascotas"),
    (("entrenamiento", "ejercicio", "fitness", "calistenia", "workout", "gym", "gimnasio"), "entrenamiento"),
    (("cancha", "deportiva", "deportivo"), "canchas"),
    (("skate",), "skate"),
    (("picnic",), "picnic"),
)


@dataclass(frozen=True)
class SiteCluster:
    primary: Place
    zones: tuple[Place, ...]
    description: str
    curated: bool

    @property
    def zone_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        seen: set[str] = set()
        for zone in self.zones:
            label = zone_label(zone.name)
            if label in seen or fold(label) == fold(self.primary.name):
                continue
            seen.add(label)
            labels.append(label)
        return tuple(labels)

    def names(self) -> tuple[str, ...]:
        return (self.primary.name, *(zone.name for zone in self.zones))


@dataclass(frozen=True)
class RankedPlace:
    place: Place
    description: str
    curated: bool
    score: float


def cluster_interests(ranked: list[RankedPlace]) -> list[SiteCluster]:
    """Elige un pin por sitio: el más relevante, y el resto cercano queda como zona."""
    if not ranked:
        return []
    ordered = sorted(ranked, key=lambda item: (-item.score, item.place.distance_meters, item.place.name))
    clusters: list[SiteCluster] = []
    used: set[int] = set()
    for index, candidate in enumerate(ordered):
        if index in used:
            continue
        zones: list[Place] = []
        for other_index, other in enumerate(ordered):
            if other_index == index or other_index in used:
                continue
            if same_site(candidate.place, other.place):
                used.add(other_index)
                zones.append(other.place)
        used.add(index)
        clusters.append(
            SiteCluster(
                primary=candidate.place,
                zones=tuple(zones),
                description=candidate.description,
                curated=candidate.curated,
            )
        )
    return clusters


def same_site(left: Place, right: Place) -> bool:
    distance = distance_meters(left.latitude, left.longitude, right.latitude, right.longitude)
    if distance > CLUSTER_METERS:
        return False
    left_tokens = set(tokens(left.name))
    right_tokens = set(tokens(right.name))
    shared = left_tokens & right_tokens
    shared_meaningful = shared - AMENITY_TOKENS
    if shared_meaningful:
        return True
    if looks_like_amenity(left.name) or looks_like_amenity(right.name):
        return True
    if category_of(left) in SITE_CATEGORIES and category_of(right) in SITE_CATEGORIES:
        return True
    return False


def rank_place(place: Place, description: str, curated: bool) -> RankedPlace:
    score = 0.0
    if curated:
        score += 1000
    if description:
        score += 80
    category = category_of(place)
    if category in {"museum", "monument", "historical_landmark", "church", "place_of_worship"}:
        score += 120
    elif category in {"plaza", "park", "tourist_attraction"}:
        score += 100
    elif category in SITE_CATEGORIES:
        score += 60
    if looks_like_amenity(place.name):
        score -= 200
    # Nombres cortos y propios de un sitio suelen ser el pin principal.
    score += max(0, 40 - len(place.name))
    score -= min(place.distance_meters, 500) / 50
    return RankedPlace(place=place, description=description, curated=curated, score=score)


def looks_like_amenity(name: str) -> bool:
    return bool(set(tokens(name)) & AMENITY_TOKENS)


def category_of(place: Place) -> str:
    return (place.category or "").strip().lower().replace(" ", "_")


def zone_label(name: str) -> str:
    folded = fold(name)
    for keys, label in ZONE_LABELS:
        if any(key in folded for key in keys):
            return label
    cleaned = " ".join(part for part in name.split() if fold(part) not in {"zona", "area", "área", "sector", "de", "del", "la", "las", "el", "los"})
    return cleaned.strip() or name.strip()
