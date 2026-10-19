"""Voz de la guía. Arranque: python3 main.py"""

import sys

if sys.version_info < (3, 11):
    sys.stderr.write("Se necesita Python 3.11 o superior.\n")
    raise SystemExit(1)

from pathlib import Path

try:
    from agent.server import serve
except ModuleNotFoundError as error:
    if (error.name or "").startswith("langgraph"):
        sys.stderr.write("Falta LangGraph. Instalá requirements.txt en un entorno virtual.\n")
        raise SystemExit(1) from None
    raise


def main() -> None:
    root = Path(__file__).resolve().parent
    serve(root / ".env")


if __name__ == "__main__":
    main()
