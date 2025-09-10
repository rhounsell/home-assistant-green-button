"""The Green Button integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from . import state

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]

async def async_setup_entry(hass, entry):
    from .coordinator import GreenButtonCoordinator

    # You may want to get the XML path from entry.data or options
    xml_path = entry.data.get("xml_path")
    coordinator = GreenButtonCoordinator(hass, xml_path)
    await coordinator.async_config_entry_first_refresh()
    hass.data["green_button_coordinator"] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_unload_entry(hass, entry):
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.pop("green_button_coordinator")
    return unload_ok
