"""Shared helpers and base class for Green Button sensor entities."""

from __future__ import annotations

import asyncio
from functools import cached_property
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import model, scaling
from .const import (
    CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
)
from .coordinator import GreenButtonCoordinator
from .statistic_ids import statistic_id_from_unique_id


def _legacy_unique_id(entry_id: str, meter_reading_id: str, suffix: str) -> str:
    """Return the pre-2.0 identifier based on the final provider path segment."""
    return f"{entry_id}_{meter_reading_id.rsplit('/', 1)[-1]}{suffix}"


def _cost_multiplier(coordinator: GreenButtonCoordinator, gas: bool = False) -> int:
    """Return the configured fallback used only when XML omits a multiplier."""
    return scaling.configured_multiplier(
        coordinator.config_entry,
        (
            CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER
            if gas
            else CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER
        ),
        (
            DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER
            if gas
            else DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER
        ),
    )


def _has_interval_readings(meter_reading: model.MeterReading) -> bool:
    """Return whether a meter stream has usable interval data."""
    return bool(meter_reading.interval_blocks) and any(
        reading.value is not None
        for block in meter_reading.interval_blocks
        for reading in block.interval_readings
    )


def _schedule_hass_task_from_any_thread(hass: HomeAssistant, coro) -> None:
    """Schedule a coroutine on HA's event loop from any thread safely.

    If called on the event loop, schedule directly; otherwise, use call_soon_threadsafe.
    """
    loop = hass.loop
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        hass.async_create_task(coro)
    else:
        loop.call_soon_threadsafe(lambda: hass.async_create_task(coro))


class GreenButtonStatisticsSensor(
    CoordinatorEntity[GreenButtonCoordinator], SensorEntity
):
    """Display imported totals without automatic sensor statistics."""

    _legacy_unique_id: str
    _attr_state_class = None

    @property
    def legacy_unique_id(self) -> str:
        """Return the former path-suffix ID for a possible registry migration."""
        return self._legacy_unique_id

    @cached_property
    def unique_id(self) -> str:
        """Return the required unique ID for this statistics sensor."""
        unique_id = self._attr_unique_id
        if unique_id is None:
            raise RuntimeError("Green Button statistics sensor has no unique ID")
        return unique_id

    @property
    def long_term_statistics_id(self) -> str:
        """Return the external series associated with this display entity."""
        return statistic_id_from_unique_id(self.unique_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the series to select in the Energy dashboard."""
        return {"statistic_id": self.long_term_statistics_id}
