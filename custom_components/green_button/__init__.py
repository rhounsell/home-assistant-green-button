"""The Green Button integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import GreenButtonCoordinator
from .services_coordinator import async_setup_services, async_unload_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Green Button component."""
    _LOGGER.info("Setting up Green Button component services")
    try:
        # Set up services
        await async_setup_services(hass)
    except ImportError as err:
        _LOGGER.error("Failed to import services module: %s", err)
        return False
    except AttributeError as err:
        _LOGGER.error("Service registration error: %s", err)
        return False
    else:
        _LOGGER.info("Green Button component setup completed successfully")
        return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Green Button from a config entry."""
    _LOGGER.debug("Setting up Green Button integration")

    # Set up services if not already done
    if not hass.services.has_service(DOMAIN, "import_espi_xml"):
        _LOGGER.info("Setting up Green Button services from config entry")
        await async_setup_services(hass)

    # Create the coordinator
    coordinator = GreenButtonCoordinator(hass, entry)

    # Store the coordinator in hass.data
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
    }

    # If there's XML data in the config, load it
    xml_data = entry.data.get("xml")
    if xml_data:
        try:
            await coordinator.async_add_data(xml_data)
            _LOGGER.debug("Loaded initial XML data into coordinator")
        except (ValueError, OSError) as err:
            _LOGGER.warning("Failed to load initial XML data: %s", err)

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _LOGGER.info("Green Button integration setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a Green Button config entry."""
    _LOGGER.debug("Unloading Green Button integration")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Clean up coordinator
        hass.data[DOMAIN].pop(entry.entry_id)

        # If no more entries, unload services
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN, None)
            await async_unload_services(hass)

        _LOGGER.debug("Unloading of %s successful", entry.title)

    return unload_ok
