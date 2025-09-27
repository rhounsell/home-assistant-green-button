"""Service implementations for the Green Button integration."""

from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.service import async_register_admin_service

from . import statistics
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator
from .parsers import espi

_LOGGER = logging.getLogger(__name__)

SERVICE_IMPORT_ESPI_XML = "import_espi_xml"
SERVICE_DELETE_STATISTICS = "delete_statistics"

IMPORT_ESPI_XML_SCHEMA = vol.Schema(
    {
        vol.Required("xml"): cv.string,
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
        xml_data = call.data["xml"]

        _LOGGER.info("Importing ESPI XML data via service")

        try:
            # Parse the XML to get usage points
            usage_points = espi.parse_xml(xml_data)
            _LOGGER.info("Found %d usage points in XML", len(usage_points))

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

                # Update the coordinator with new XML data
                await coordinator.async_add_data(xml_data)
                _LOGGER.info("Updated coordinator for entry %s", entry.entry_id)

                # Get all sensor entities for this coordinator and update their statistics
                for usage_point in usage_points:
                    for meter_reading in usage_point.meter_readings:
                        # Find matching sensor entities by meter reading ID
                        entity_registry = hass.helpers.entity_registry.async_get(hass)
                        entities = (
                            hass.helpers.entity_registry.async_entries_for_config_entry(
                                entity_registry, entry.entry_id
                            )
                        )

                        for entity_entry in entities:
                            if entity_entry.platform == "sensor":
                                entity = hass.states.get(entity_entry.entity_id)
                                if entity and hasattr(
                                    entity, "async_update_sensor_and_statistics"
                                ):
                                    await entity.async_update_sensor_and_statistics(
                                        meter_reading
                                    )
                                    _LOGGER.debug(
                                        "Updated statistics for entity %s",
                                        entity_entry.entity_id,
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
    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_IMPORT_ESPI_XML,
        import_espi_xml_service,
        schema=IMPORT_ESPI_XML_SCHEMA,
    )

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_DELETE_STATISTICS,
        delete_statistics_service,
        schema=DELETE_STATISTICS_SCHEMA,
    )

    _LOGGER.info("Green Button services registered")


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload services for the Green Button integration."""
    hass.services.async_remove(DOMAIN, SERVICE_IMPORT_ESPI_XML)
    hass.services.async_remove(DOMAIN, SERVICE_DELETE_STATISTICS)
    _LOGGER.info("Green Button services unloaded")
