"""Presupuesto de tiempo por pedido. El orquestador corta a los 10 segundos."""

from __future__ import annotations

import time

REQUEST_BUDGET_SECONDS = 8.5
MIN_CALL_SECONDS = 0.4


class Deadline:
    def __init__(self, seconds: float = REQUEST_BUDGET_SECONDS) -> None:
        self._ends_at = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._ends_at - time.monotonic())

    def timeout_for(self, wanted: float, reserve: float = 0.0) -> float | None:
        available = min(wanted, self.remaining() - reserve)
        if available < MIN_CALL_SECONDS:
            return None
        return available
