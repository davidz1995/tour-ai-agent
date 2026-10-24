"""Últimos lugares dichos en esta ejecución. No se escribe a disco ni se guarda un perfil."""

from __future__ import annotations

import threading

from agent.situation import fold

MAX_USERS = 200
MAX_PLACES = 5


class MentionMemory:
    def __init__(self) -> None:
        self._names: dict[str, list[str]] = {}
        self._scripts: dict[str, dict[str, str]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._names.clear()
            self._scripts.clear()
            self._order.clear()

    def recent(self, user_id: str) -> list[str]:
        with self._lock:
            return list(self._names.get(_key(user_id), []))

    def remember(self, user_id: str, name: str, script: str = "") -> None:
        cleaned = " ".join(name.split())
        story = " ".join(script.split())
        if not cleaned:
            return
        key = _key(user_id)
        with self._lock:
            current = [item for item in self._names.get(key, []) if fold(item) != fold(cleaned)]
            current.append(cleaned)
            self._names[key] = current[-MAX_PLACES:]
            scripts = self._scripts.setdefault(key, {})
            if story:
                scripts[fold(cleaned)] = story
            kept = {fold(item) for item in self._names[key]}
            self._scripts[key] = {folded: text for folded, text in scripts.items() if folded in kept}
            self._touch(key)

    def script_for(self, user_id: str, names: list[str]) -> str:
        key = _key(user_id)
        with self._lock:
            scripts = self._scripts.get(key, {})
            for name in names:
                story = scripts.get(fold(name))
                if story:
                    return story
        return ""

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        overflow = len(self._order) - MAX_USERS
        if overflow <= 0:
            return
        for old in self._order[:overflow]:
            self._names.pop(old, None)
            self._scripts.pop(old, None)
        self._order = self._order[overflow:]


def _key(user_id: str) -> str:
    cleaned = user_id.strip()
    return cleaned or "anon"


SHARED_MEMORY = MentionMemory()
