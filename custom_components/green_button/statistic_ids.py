"""Stable IDs for statistics owned by Green Button."""

from hashlib import sha256

from .const import DOMAIN


def statistic_id_from_unique_id(unique_id: str) -> str:
    """Keep statistic IDs stable when a display entity is renamed."""
    return f"{DOMAIN}:{sha256(unique_id.encode()).hexdigest()}"


def stream_unique_id(
    entry_id: str,
    usage_point_id: str,
    meter_reading_id: str,
    suffix: str,
) -> str:
    """Return the stable display-entity ID for one provider stream."""
    identity = f"{usage_point_id}\x00{meter_reading_id}\x00{suffix}"
    return f"{entry_id}_{sha256(identity.encode()).hexdigest()}{suffix}"
