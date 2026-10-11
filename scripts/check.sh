#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format
uv run ruff check --fix
uv run mypy --strict
uv run lint-imports
uv run pytest
