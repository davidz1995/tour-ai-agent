"""Señal visual por reconocimiento de lugar. No usa CLIP ni promedia vistas."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# DINOv2 ViT-B/14 con agregación GeM (generalized mean), el descriptor de un lugar.
# No es un embedding genérico. Un vector guardado con otro modelId no se compara.
MODEL_ID = "dinov2-vitb14-gem"
VECTOR_DIM = 768
PLACE_VIEWS_DIR = Path(__file__).resolve().parent / "knowledge" / "place-views"
MAX_VIEW_FILE_BYTES = 1_048_576
MAX_VIEWS = 512


@dataclass(frozen=True)
class StoredView:
    place_id: str
    model_id: str
    view_id: str
    vector: tuple[float, ...]


class ReferenceCatalog:
    def __init__(self, views: tuple[StoredView, ...] = ()) -> None:
        self.views = views

    def for_place(self, place_id: str, model_id: str) -> tuple[StoredView, ...]:
        return tuple(view for view in self.views if view.place_id == place_id and view.model_id == model_id)


def load_catalog(directory: Path | None = None) -> ReferenceCatalog:
    root = PLACE_VIEWS_DIR if directory is None else directory
    try:
        if not root.is_dir():
            return ReferenceCatalog()
        paths = sorted(path for path in root.glob("*.json") if path.is_file())
    except OSError:
        sys.stderr.write("no se pudieron leer las vistas de lugar\n")
        return ReferenceCatalog()
    if len(paths) > MAX_VIEWS:
        sys.stderr.write("hay demasiadas vistas de lugar\n")
        paths = paths[:MAX_VIEWS]
    views: list[StoredView] = []
    for path in paths:
        view = read_view(path)
        if view is not None:
            views.append(view)
    return ReferenceCatalog(tuple(views))


def read_view(path: Path) -> StoredView | None:
    try:
        if path.stat().st_size > MAX_VIEW_FILE_BYTES:
            sys.stderr.write(f"vista de lugar inválida: {path.name}\n")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        sys.stderr.write(f"vista de lugar inválida: {path.name}\n")
        return None
    view = parse_view(payload)
    if view is None:
        sys.stderr.write(f"vista de lugar inválida: {path.name}\n")
    return view


def parse_view(payload: object) -> StoredView | None:
    if not isinstance(payload, dict):
        return None
    place_id = _label(payload.get("placeId"))
    model_id = _label(payload.get("modelId"))
    view_id = _label(payload.get("viewId"))
    vector = _vector(payload.get("vector"))
    if place_id is None or model_id is None or view_id is None or vector is None:
        return None
    return StoredView(place_id, model_id, view_id, vector)


def nearest_similarity(model_id: str, query: Sequence[float], views: Sequence[StoredView]) -> float | None:
    """Similitud con la vista más cercana. Las demás vistas no se promedian."""
    scores: list[float] = []
    for view in views:
        if view.model_id != model_id or len(view.vector) != len(query):
            continue
        score = _cosine(query, view.vector)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    best = max(scores)
    if not math.isfinite(best) or best < -1.000001 or best > 1.000001:
        return None
    return max(-1.0, min(1.0, best))


def embedding_signal(
    place_id: str,
    photo: bytes,
    catalog: ReferenceCatalog,
    embed=None,
) -> dict[str, object]:
    views = catalog.for_place(place_id, MODEL_ID)
    if not views:
        return {"available": False}
    vector = embed_visit(photo) if embed is None else embed(photo)
    if vector is None:
        return {"available": False}
    score = nearest_similarity(MODEL_ID, vector, views)
    if score is None:
        return {"available": False}
    return {"available": True, "nearest": score}


def embed_visit(photo: bytes) -> tuple[float, ...] | None:
    """Este corte no calcula el descriptor DINOv2 de la foto.

    Sin ese vector no hay similitud, aunque el lugar ya tenga vistas guardadas.
    """
    del photo
    return None


def _label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = " ".join(value.split())
    if not label or len(label) > 80:
        return None
    return label


def _vector(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != VECTOR_DIM:
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        numbers.append(float(item))
    return tuple(numbers)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        if isinstance(a, bool) or isinstance(b, bool) or not math.isfinite(a) or not math.isfinite(b):
            return None
        dot += float(a) * float(b)
        left_norm += float(a) * float(a)
        right_norm += float(b) * float(b)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)
