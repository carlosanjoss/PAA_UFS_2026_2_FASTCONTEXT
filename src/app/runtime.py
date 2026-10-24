"""Shared, process-local application runtime."""

from __future__ import annotations

from functools import lru_cache

from src.app.bootstrap import create_application
from src.app.models import ApplicationContainer


@lru_cache(maxsize=1)
def get_application() -> ApplicationContainer:
    """Build shared application resources once per API process."""
    return create_application()
