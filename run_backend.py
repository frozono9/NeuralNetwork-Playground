"""Run the backend using the workspace .venv interpreter.

Why: many systems have multiple Python installs. The backend dependencies
(FastAPI, Socket.IO, torch, etc.) are installed in this repo's `.venv`.

Usage:
  python3 run_backend.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    backend_main = os.path.join(repo_root, "backend", "main.py")

    if not os.path.exists(backend_main):
        raise SystemExit(f"backend entrypoint not found: {backend_main}")

    if not os.path.exists(venv_python):
        raise SystemExit(
            "Workspace venv not found at .venv/bin/python. "
            "Create it and install dependencies, e.g.\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r backend/requirements.txt"
        )

    os.execv(venv_python, [venv_python, backend_main, *sys.argv[1:]])


if __name__ == "__main__":
    main()
