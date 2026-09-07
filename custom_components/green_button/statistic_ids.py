"""Stable IDs for statistics owned by Green Button."""

from hashlib import sha256

from .const import DOMAIN


def statistic_id_from_unique_id(unique_id: str) -> str:
    """Keep statistic IDs stable when a display entity is renamed."""
    return f"{DOMAIN}:{sha256(unique_id.encode()).hexdigest()}"
