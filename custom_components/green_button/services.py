"""Service implementations for the Green Button integration."""

from __future__ import annotations

import logging


import aiofiles
import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

from . import statistics
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator

_LOGGER = logging.getLogger(__name__)

SERVICE_IMPORT_ESPI_XML = "import_espi_xml"
SERVICE_DELETE_STATISTICS = "delete_statistics"

IMPORT_ESPI_XML_SCHEMA = vol.Schema(
    {
        vol.Required("xml_file_path"): cv.string,
    }
)

DELETE_STATISTICS_SCHEMA = vol.Schema(
    {
        vol.Required("statistic_id"): cv.string,
    }
)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for the Green Button integration."""

    async def import_espi_xml_service(call: ServiceCall) -> None:
        """Handle the import_espi_xml service call."""

        import os
        xml_path = call.data["xml_file_path"]
        if not os.path.isfile(xml_path):
            _LOGGER.error("Specified XML file does not exist: %s", xml_path)
            return
        try:
            async with aiofiles.open(xml_path, "r", encoding="utf-8") as f:
                xml_data = await f.read()
        except Exception as e:
            _LOGGER.error("Failed to read XML file: %s", e)
            return

        _LOGGER.info("Importing ESPI XML data via service from file: %s", xml_path)

        try:
            # Get all Green Button config entries
            entries = list(hass.config_entries.async_entries(DOMAIN))

            if not entries:
                _LOGGER.warning("No Green Button integrations found")
                return

            # Process the XML data for each entry
            for entry in entries:
                coordinator_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
                coordinator: GreenButtonCoordinator | None = coordinator_data.get(
                    "coordinator"
                )

                if not coordinator:
                    _LOGGER.warning("No coordinator found for entry %s", entry.entry_id)
                    continue

                # Let the coordinator handle all data parsing and updates
                await coordinator.async_add_xml_data(xml_data)

                _LOGGER.info(
                    "Updated coordinator and refreshed entities for entry %s",
                    entry.entry_id,
                )

            _LOGGER.info("ESPI XML import completed successfully")

        except Exception as err:
            _LOGGER.error("Failed to import ESPI XML: %s", err)
            raise

    async def delete_statistics_service(call: ServiceCall) -> None:
        """Handle the delete_statistics service call."""
        statistic_id = call.data["statistic_id"]

        _LOGGER.info("Deleting statistics for ID: %s", statistic_id)

        try:
            await statistics.clear_statistic(hass, statistic_id)
            _LOGGER.info("Successfully deleted statistics for %s", statistic_id)
        except Exception as err:
            _LOGGER.error("Failed to delete statistics for %s: %s", statistic_id, err)
            raise

    # Register services
    try:
        hass.services.async_register(
            DOMAIN,
            SERVICE_IMPORT_ESPI_XML,
            import_espi_xml_service,
            schema=IMPORT_ESPI_XML_SCHEMA,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_DELETE_STATISTICS,
            delete_statistics_service,
            schema=DELETE_STATISTICS_SCHEMA,
        )

        _LOGGER.info("Green Button services registered successfully")
    except Exception as err:
        _LOGGER.error("Failed to register Green Button services: %s", err)
        raise


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload services for the Green Button integration."""
    hass.services.async_remove(DOMAIN, SERVICE_IMPORT_ESPI_XML)
    hass.services.async_remove(DOMAIN, SERVICE_DELETE_STATISTICS)
    _LOGGER.info("Green Button services unloaded")
