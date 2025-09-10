"""The Green Button integration."""
# from __future__ import annotations

from typing import Final
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import GreenButtonCoordinator

_LOGGER: Final = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:

    # You may want to get the XML path from entry.data or options
    xml_path = config_entry.data.get("xml_path")
    hass.data.setdefault(DOMAIN, {})
    coordinator = GreenButtonCoordinator(hass, xml_path)
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][config_entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)
    return True

async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_forward_entry_unload(config_entry, Platform.SENSOR)
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id)
        _LOGGER.debug("Unloading of %s successful", config_entry.title)
    return unload_ok
