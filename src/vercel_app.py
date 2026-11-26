"""Serve the production React build alongside the analysis API."""

from src.api import app

app.frontend("/", directory="frontend/dist", fallback=None)
