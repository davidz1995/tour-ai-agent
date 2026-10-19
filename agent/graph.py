"""Grafo de dos nodos: Jev decide si hablar y el guía redacta solo en ese caso."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.contract import Context, Place
from agent.deadline import Deadline
from agent.history import SHARED_MEMORY, MentionMemory
from agent.jev import JEV_TIMEOUT_SECONDS, already_heard, clean_jev_state, probability_of_true
from agent.retrieve import load_index
from agent.settings import Settings
from agent.situation import analyze_situation
from agent.voice import EMPTY_REPLY, GUIDE_TIMEOUT_SECONDS, UNREADABLE_REPLY, write_script
from agent.web_context import WIKI_TIMEOUT_SECONDS, lookup_web

GUIDE_RESERVE_SECONDS = 1.0


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
    if not situation.mentions:
        reply = UNREADABLE_REPLY if situation.unreadable else EMPTY_REPLY
        return GuideResult(reply)

    budget = deadline if deadline is not None else Deadline()
    store = memory if memory is not None else SHARED_MEMORY
    history = store.recent(user_id)
    interests: list[Place] = []
    spoken: Place | None = None
    reply = ""
    for mention in situation.mentions:
        if budget.remaining() < GUIDE_RESERVE_SECONDS:
            break
        document = load_index().search(mention.place)
        state = initial_state(
            latitude=situation.origin.latitude,
            longitude=situation.origin.longitude,
            name=mention.place.name,
            category=mention.place.category or "",
            description=document.text if document is not None else "",
            source="local" if document is not None else "",
            place_latitude=mention.place.latitude,
            place_longitude=mention.place.longitude,
            recent_history=[],
        )
        if not evaluate(state, settings, budget)["should_trigger"]:
            continue
        interests.append(mention.place)
        if not speak or spoken is not None or already_heard(mention.place.name, history):
            continue
        script = str(narrate(state, settings, budget).get("final_audio_script", "")).strip()
        if not script:
            continue
        store.remember(user_id, mention.place.name)
        spoken = mention.place
        reply = script
    return GuideResult(reply, spoken, tuple(interests))


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
    )
    timeout = deadline.timeout_for(JEV_TIMEOUT_SECONDS, reserve=GUIDE_RESERVE_SECONDS)
    confidence = probability_of_true(jev_state, settings, timeout)
    return {
        "jev_confidence": confidence,
        "should_trigger": confidence > settings.jev_threshold,
    }


def narrate(state: GuideState, settings: Settings, deadline: Deadline) -> dict[str, object]:
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
        "jev_confidence": 0.0,
        "should_trigger": False,
        "final_audio_script": "",
    }
