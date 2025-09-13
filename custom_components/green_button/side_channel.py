"""A module containing utilities to set and get side channels."""
from __future__ import annotations

from typing import Any
from typing import Generic
from typing import TypeVar

from homeassistant.core import HomeAssistant

from . import model
from . import state

_SIDE_CHANNEL_KEY = "side_channels"


T = TypeVar("T")


class SideChannelKey(Generic[T]):
    """A side channel key.

    It describes a specific side channel and the type that can be stored in it.
    """

    def __init__(self, name: str) -> None:
        """Initialize the instance."""
        super().__init__()
        self.name = name


class SideChannels:
    """A container for all side channels."""

    _side_channels: dict[str, Any]

    def __init__(self) -> None:
        """Initialize the instance."""
        self._side_channels = {}

    def _get_entry_side_channels(self, entry_unique_id: str) -> dict[str, Any]:
        return self._side_channels.setdefault(entry_unique_id, {})

    def get(self, entry_unique_id: str, key: SideChannelKey[T]) -> T | None:
        """Return the value of the side channel for a ConfigEntry."""
        return self._get_entry_side_channels(entry_unique_id).get(key.name)

    def set(self, entry_unique_id: str, key: SideChannelKey[T], value: T) -> None:
        """Set the value of the side channel for a ConfigEntry."""
        self._get_entry_side_channels(entry_unique_id)[key.name] = value


# INITIAL_IMPORT_USAGE_POINT: SideChannelKey[model.UsagePoint] = SideChannelKey(
#     "initial_import_usage_point"
INITIAL_IMPORT_INTERVAL_BLOCK: SideChannelKey[model.IntervalBlock] = SideChannelKey(
    "initial_import_interval_block"
)


def get(hass: HomeAssistant) -> SideChannels:
    """Get the side channels."""
    return state.get(hass).side_channels
