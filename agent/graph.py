"""Grafo de dos nodos: Jev decide si hablar y el guía redacta solo en ese caso."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.cluster import SiteCluster, cluster_interests, rank_place
from agent.contract import Area, Context, Place
from agent.deadline import Deadline
from agent.history import SHARED_MEMORY, MentionMemory
from agent.jev import JEV_TIMEOUT_SECONDS, already_heard, clean_jev_state, probability_of_true, repeats_place
from agent.retrieve import HistoryIndex, distance_meters, load_index, looks_like_person_memorial
from agent.settings import Settings
from agent.situation import KNOWN_PLACES, Mention, analyze_situation, cardinal
from agent.voice import EMPTY_REPLY, GUIDE_TIMEOUT_SECONDS, UNREADABLE_REPLY, write_script
from agent.web_context import WIKI_TIMEOUT_SECONDS, lookup_web

GUIDE_RESERVE_SECONDS = 1.0
MAX_PERSON_CHECKS = 4


class PoiData(TypedDict):
    name: str
    category: str
    description: str
    source: str
    latitude: float
    longitude: float


class GuideState(TypedDict):
    user_location: dict[str, float]
    poi_data: PoiData
    recent_history: list[str]
    jev_confidence: float
    should_trigger: bool
    final_audio_script: str


@dataclass(frozen=True)
class GuideResult:
    reply: str
    place: Place | None = None
    places: tuple[Place, ...] = ()
    remembered: str = ""
    remembered_place: Place | None = None


def build_reply(
    context: Context,
    settings: Settings,
    user_id: str = "",
    memory: MentionMemory | None = None,
    deadline: Deadline | None = None,
) -> str:
    return build_guide(context, settings, user_id, memory, deadline).reply


def build_guide(
    context: Context,
    settings: Settings,
    user_id: str = "",
    memory: MentionMemory | None = None,
    deadline: Deadline | None = None,
    speak: bool = True,
) -> GuideResult:
    situation = analyze_situation(context)
    store = memory if memory is not None else SHARED_MEMORY
    history = store.recent(user_id)
    custom = next_custom_area(context, history) if speak else None
    if custom is not None:
        return spoken_custom_area(store, user_id, custom)
    budget = deadline if deadline is not None else Deadline()
    if not situation.mentions:
        reply = UNREADABLE_REPLY if situation.unreadable else EMPTY_REPLY
        return GuideResult(reply)
    index = load_index()
    mentions = guide_mentions(context, index)
    ranked = []
    person_checks = 0
    accepted_names: list[str] = []
    accepted_ids: set[str] = set()
    remembered = ""
    remembered_place: Place | None = None
    for mention in mentions:
        if budget.remaining() < GUIDE_RESERVE_SECONDS:
            break
        figure = index.figure_match(mention.place)
        memorial = figure is None and looks_like_person_memorial(mention.place.name)
        if memorial:
            if person_checks >= MAX_PERSON_CHECKS:
                continue
            person_checks += 1
        if figure is not None:
            description = figure.text
            curated = True
            identity = f"figure:{figure.name}"
        else:
            document = index.search(mention.place)
            description = document.text if document is not None else ""
            curated = document is not None
            identity = f"doc:{document.id}" if document is not None else ""
        duplicate = _is_repeated(mention.place.name, history, accepted_names, identity, accepted_ids)
        if duplicate and speak and repeats_place(mention.place.name, history) and not remembered:
            cached = store.script_for(user_id, [mention.place.name, *history])
            if cached:
                remembered = cached
                remembered_place = mention.place
        state = initial_state(
            latitude=situation.origin.latitude,
            longitude=situation.origin.longitude,
            name=mention.place.name,
            category=mention.place.category or "",
            description=description,
            source="local" if curated else "",
            place_latitude=mention.place.latitude,
            place_longitude=mention.place.longitude,
            recent_history=[*history, *accepted_names],
            is_duplicate=duplicate,
        )
        if not evaluate(state, settings, budget)["should_trigger"]:
            continue
        ranked.append(rank_place(mention.place, description, curated))
        accepted_names.append(mention.place.name)
        if identity:
            accepted_ids.add(identity)

    clusters = cluster_interests(ranked)
    interests = tuple(cluster.primary for cluster in clusters)
    if not clusters:
        return GuideResult("", None, (), remembered, remembered_place)

    spoken: Place | None = None
    reply = ""
    if speak:
        for cluster in clusters:
            if site_already_heard(cluster, history):
                cached = store.script_for(user_id, list(cluster.names()))
                if cached and not remembered:
                    remembered = cached
                    remembered_place = cluster.primary
                if cached:
                    continue
            if budget.remaining() < GUIDE_RESERVE_SECONDS:
                break
            script = narrate_cluster(cluster, settings, budget)
            if not script:
                continue
            for name in cluster.names():
                store.remember(user_id, name, script)
            spoken = cluster.primary
            reply = script
            break
    if speak:
        known = store.recent(user_id)
        for name in accepted_names:
            if not already_heard(name, known):
                store.remember(user_id, name, "")
    return GuideResult(reply, spoken, interests, remembered, remembered_place)


def guide_mentions(context: Context, index: HistoryIndex) -> list[Mention]:
    """El lugar entra aunque no sea de los más cercanos. Una figura curada también, aunque el mapa no la mande."""
    places = sorted(context.places, key=lambda place: (place.distance_meters, place.name))
    chosen: list[Place] = places[:KNOWN_PLACES]
    seen = {id(place) for place in chosen}
    for place in places:
        if id(place) in seen or looks_like_person_memorial(place.name):
            continue
        chosen.append(place)
        seen.add(id(place))
    origin = context.location
    mentions = [Mention(place, cardinal(origin, place)) for place in chosen]
    covered = {
        figure.name
        for mention in mentions
        if (figure := index.figure_match(mention.place)) is not None
    }
    for figure in index.figures_near(origin.latitude, origin.longitude):
        if figure.name in covered:
            continue
        place = Place(
            name=figure.name,
            category="historical_landmark",
            distance_meters=distance_meters(
                origin.latitude, origin.longitude, figure.latitude, figure.longitude
            ),
            latitude=figure.latitude,
            longitude=figure.longitude,
        )
        mentions.append(Mention(place, cardinal(origin, place)))
        covered.add(figure.name)
    mentions.sort(key=lambda mention: mention_rank(mention, index))
    return mentions


def mention_rank(mention: Mention, index: HistoryIndex) -> tuple[int, float, str]:
    if index.figure_match(mention.place) is not None:
        group = 0
    elif looks_like_person_memorial(mention.place.name):
        group = 2
    else:
        group = 1
    return (group, mention.place.distance_meters, mention.place.name)


CUSTOM_AREA = "custom-area"


def next_custom_area(context: Context, history: list[str]) -> Area | None:
    for area in context.areas:
        if area.category != CUSTOM_AREA or already_heard(area.name, history):
            continue
        if area.description.strip():
            return area
    return None


def place_for_area(area: Area) -> Place:
    return Place(
        name=area.name,
        category=area.category,
        distance_meters=0,
        latitude=area.latitude,
        longitude=area.longitude,
    )


def spoken_custom_area(store: MentionMemory, user_id: str, area: Area) -> GuideResult:
    place = place_for_area(area)
    store.remember(user_id, area.name, area.description)
    return GuideResult(area.description, place, (place,))


def _is_repeated(
    name: str,
    history: list[str],
    accepted_names: list[str],
    identity: str,
    accepted_ids: set[str],
) -> bool:
    if repeats_place(name, [*history, *accepted_names]):
        return True
    return bool(identity) and identity in accepted_ids


def site_already_heard(cluster: SiteCluster, history: list[str]) -> bool:
    return any(already_heard(name, history) for name in cluster.names())


def narrate_cluster(cluster: SiteCluster, settings: Settings, deadline: Deadline) -> str:
    state = initial_state(
        latitude=cluster.primary.latitude,
        longitude=cluster.primary.longitude,
        name=cluster.primary.name,
        category=cluster.primary.category or "",
        description=cluster.description,
        source="local" if cluster.curated else "",
        place_latitude=cluster.primary.latitude,
        place_longitude=cluster.primary.longitude,
        recent_history=[],
    )
    return str(
        narrate(state, settings, deadline, zones=cluster.zone_labels).get("final_audio_script", "")
    ).strip()


def compile_graph(settings: Settings, deadline: Deadline | None = None):
    budget = deadline if deadline is not None else Deadline()

    def evaluator(state: GuideState) -> dict[str, float | bool]:
        return evaluate(state, settings, budget)

    def guide(state: GuideState) -> dict[str, object]:
        return narrate(state, settings, budget)

    graph = StateGraph(GuideState)
    graph.add_node("evaluator", evaluator)
    graph.add_node("guide", guide)
    graph.add_edge(START, "evaluator")
    graph.add_conditional_edges("evaluator", route_after_evaluator)
    graph.add_edge("guide", END)
    return graph.compile()


def evaluate(state: GuideState, settings: Settings, deadline: Deadline) -> dict[str, float | bool]:
    poi = state["poi_data"]
    jev_state = clean_jev_state(
        poi["name"],
        poi["category"],
        state["recent_history"],
        in_curated_base=poi["source"] == "local",
        is_duplicate=bool(state.get("is_duplicate", False)),
    )
    timeout = deadline.timeout_for(JEV_TIMEOUT_SECONDS, reserve=GUIDE_RESERVE_SECONDS)
    confidence = probability_of_true(jev_state, settings, timeout)
    return {
        "jev_confidence": confidence,
        "should_trigger": confidence > settings.jev_threshold,
    }


def narrate(
    state: GuideState,
    settings: Settings,
    deadline: Deadline,
    zones: tuple[str, ...] | list[str] = (),
) -> dict[str, object]:
    poi = dict(state["poi_data"])
    if not poi["description"]:
        timeout = deadline.timeout_for(WIKI_TIMEOUT_SECONDS, reserve=GUIDE_RESERVE_SECONDS)
        if timeout is not None:
            found = lookup_web(poi["name"], poi["latitude"], poi["longitude"], timeout)
            if found is not None:
                poi["description"] = found.text
                poi["source"] = "wikipedia"
    script = write_script(
        poi["name"],
        poi["category"],
        poi["description"],
        settings,
        deadline.timeout_for(GUIDE_TIMEOUT_SECONDS, reserve=0.2),
        zones=zones,
    )
    return {"poi_data": poi, "final_audio_script": script}


def route_after_evaluator(state: GuideState) -> str:
    if state["should_trigger"]:
        return "guide"
    return END


def initial_state(
    *,
    latitude: float,
    longitude: float,
    name: str,
    category: str,
    description: str,
    source: str,
    place_latitude: float,
    place_longitude: float,
    recent_history: list[str],
    is_duplicate: bool = False,
) -> GuideState:
    return {
        "user_location": {"latitude": latitude, "longitude": longitude},
        "poi_data": {
            "name": name,
            "category": category,
            "description": description,
            "source": source,
            "latitude": place_latitude,
            "longitude": place_longitude,
        },
        "recent_history": list(recent_history),
        "is_duplicate": is_duplicate,
        "jev_confidence": 0.0,
        "should_trigger": False,
        "final_audio_script": "",
    }
