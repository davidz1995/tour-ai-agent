"""Contexto de respaldo: Wikipedia en español, solo si places.json no tiene el lugar."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass

from agent.contract import clean_text
from agent.retrieve import distance_meters, tokens

WIKI_API = "https://es.wikipedia.org/w/api.php"
WIKI_TIMEOUT_SECONDS = 2.5
MIN_SECOND_CALL_SECONDS = 0.5
SEARCH_RADIUS_METERS = 500
MAX_MATCH_METERS = 450
SINGLE_WORD_MAX_METERS = 150
MAX_RESPONSE_BYTES = 262_144
MAX_SNIPPET_CHARS = 320
CACHE_SIZE = 256
USER_AGENT = "tour-ai-agent/0.1 (guía turística local)"
_PARENTHESES = re.compile(r"\s*\([^()]*\)")


@dataclass(frozen=True)
class WebPassage:
    title: str
    text: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)
_CACHE: OrderedDict[tuple[str, float, float], WebPassage | None] = OrderedDict()
_LOCK = threading.Lock()


def lookup_web(name: str, latitude: float, longitude: float, timeout: float) -> WebPassage | None:
    key = (" ".join(tokens(name)), round(latitude, 4), round(longitude, 4))
    if not key[0]:
        return None
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    started = time.monotonic()
    nearby = fetch_nearby_pages(latitude, longitude, timeout)
    passage = pick_page(nearby, name, latitude, longitude) if nearby is not None else None
    searched: object | None = None
    if passage is None:
        left = timeout - (time.monotonic() - started)
        if left >= MIN_SECOND_CALL_SECONDS:
            searched = fetch_pages_by_name(name, left)
            if searched is not None:
                passage = pick_page(searched, name, latitude, longitude)
    if nearby is None and searched is None:
        return None
    with _LOCK:
        _CACHE[key] = passage
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return passage


def fetch_nearby_pages(latitude: float, longitude: float, timeout: float) -> object | None:
    return fetch_pages({
        "generator": "geosearch",
        "ggscoord": f"{latitude:.6f}|{longitude:.6f}",
        "ggsradius": str(SEARCH_RADIUS_METERS),
        "ggslimit": "20",
    }, timeout)


def fetch_pages_by_name(name: str, timeout: float) -> object | None:
    query = clean_text(name, 80)
    if not query:
        return None
    return fetch_pages({
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": "10",
        "gsrnamespace": "0",
    }, timeout)


def fetch_pages(generator: dict[str, str], timeout: float) -> object | None:
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "formatversion": "2",
        **generator,
        "prop": "extracts|coordinates",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "2",
        "exlimit": "20",
        "colimit": "20",
    })
    request = urllib.request.Request(
        f"{WIKI_API}?{params}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        error.close()
        code = error.code if isinstance(error.code, int) else 0
        sys.stderr.write(f"wikipedia respondió {code}\n")
        return None
    except OSError:
        sys.stderr.write("wikipedia no disponible\n")
        return None
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None


def pick_page(payload: object, name: str, latitude: float, longitude: float) -> WebPassage | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), dict):
        return None
    pages = payload["query"].get("pages")
    if not isinstance(pages, list):
        return None
    best: WebPassage | None = None
    best_rank: tuple[int, float] | None = None
    for page in pages:
        candidate = read_page(page)
        if candidate is None:
            continue
        title, extract, page_lat, page_lon = candidate
        distance = distance_meters(latitude, longitude, page_lat, page_lon)
        score = match_score(name, title, distance)
        if score <= 0:
            continue
        rank = (score, -distance)
        if best_rank is None or rank > best_rank:
            best = WebPassage(title=title, text=extract)
            best_rank = rank
    return best


def read_page(page: object) -> tuple[str, str, float, float] | None:
    if not isinstance(page, dict):
        return None
    title = page.get("title")
    extract = page.get("extract")
    coordinates = page.get("coordinates")
    if not isinstance(title, str) or not isinstance(extract, str):
        return None
    if not isinstance(coordinates, list) or not coordinates or not isinstance(coordinates[0], dict):
        return None
    lat = coordinates[0].get("lat")
    lon = coordinates[0].get("lon")
    if isinstance(lat, bool) or isinstance(lon, bool):
        return None
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    text = speakable(extract)
    if len(text) < 20:
        return None
    return title, text, float(lat), float(lon)


def match_score(name: str, title: str, distance: float) -> int:
    if distance > MAX_MATCH_METERS:
        return 0
    place_tokens = set(tokens(name))
    title_tokens = set(tokens(_PARENTHESES.sub("", title)))
    if not place_tokens or not title_tokens:
        return 0
    if place_tokens == title_tokens:
        return 100
    smaller = place_tokens if len(place_tokens) <= len(title_tokens) else title_tokens
    larger = title_tokens if smaller is place_tokens else place_tokens
    if not smaller <= larger:
        return 0
    if len(smaller) >= 2:
        return len(smaller)
    if distance <= SINGLE_WORD_MAX_METERS:
        return 1
    return 0


def speakable(extract: str) -> str:
    text = _PARENTHESES.sub("", extract)
    text = clean_text(text, 2000)
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    cut = text[:MAX_SNIPPET_CHARS]
    period = cut.rfind(". ")
    if period > 80:
        return cut[: period + 1]
    return cut.rsplit(" ", 1)[0] + "."
