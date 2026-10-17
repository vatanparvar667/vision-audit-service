#!/usr/bin/env bash
# Start the service on http://0.0.0.0:8000 (settings are read from .env)
cd "$(dirname "$0")"
exec ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
