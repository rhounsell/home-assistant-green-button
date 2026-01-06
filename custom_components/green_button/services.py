"""Service implementations for the Green Button integration."""

from __future__ import annotations

import logging
from pathlib import Path

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.exceptions import HomeAssistantError

from . import statistics
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator

_LOGGER = logging.getLogger(__name__)

SERVICE_IMPORT_ESPI_XML = "import_espi_xml"
SERVICE_DELETE_STATISTICS = "delete_statistics"

IMPORT_ESPI_XML_SCHEMA = vol.Schema(
    {
        vol.Optional("xml_file_path"): cv.string,
        vol.Optional("xml"): cv.string,
    }
)

DELETE_STATISTICS_SCHEMA = vol.Schema(
    {
        vol.Required("statistic_id"): cv.entity_id,
    }
)


def _read_file_sync(file_path: Path) -> str:
    """Read file content synchronously."""
    return file_path.read_text(encoding="utf-8")


async def async_setup_services(hass: HomeAssistant) -> None:
    async def log_meter_reading_intervals_service(call: ServiceCall) -> None:
        """Log all meter readings, their interval block date ranges, and mapped sensor entities."""
        entity_registry = async_get_entity_registry(hass)
        entries = list(hass.config_entries.async_entries(DOMAIN))
        for entry in entries:
            coordinator: GreenButtonCoordinator | None = hass.data.get(DOMAIN, {}).get(entry.entry_id, {}).get("coordinator")
            if not coordinator or not coordinator.data:
                _LOGGER.info(f"Entry {entry.entry_id}: No coordinator or data available.")
                continue
            usage_points = coordinator.data.get("usage_points", [])
            for up_idx, usage_point in enumerate(usage_points):
                _LOGGER.info(f"UsagePoint {up_idx} (id={usage_point.id}): {len(usage_point.meter_readings)} meter readings.")
                for mr_idx, meter_reading in enumerate(usage_point.meter_readings):
                    clean_id = meter_reading.id.split("/")[-1] if "/" in meter_reading.id else meter_reading.id
                    unique_id = f"{entry.entry_id}_{clean_id}"
                    entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)
                    _LOGGER.info(f"  MeterReading {mr_idx} (id={meter_reading.id}): mapped entity_id={entity_id}")
                    for ib_idx, interval_block in enumerate(meter_reading.interval_blocks):
                        start = interval_block.start.isoformat()
                        end = (interval_block.start + interval_block.duration).isoformat()
                        _LOGGER.info(f"    IntervalBlock {ib_idx}: start={start}, end={end}, readings={len(interval_block.interval_readings)}")

    # Register the diagnostic service
    hass.services.async_register(
        DOMAIN,
        "log_meter_reading_intervals",
        log_meter_reading_intervals_service,
    )

    async def import_espi_xml_service(call: ServiceCall) -> None:
        """Handle the import_espi_xml service call."""
        xml_path = call.data.get("xml_file_path", "").strip()
        xml_content = call.data.get("xml", "").strip()
        
        # Validate that at least one is provided
        if not xml_path and not xml_content:
            msg = "No XML data provided. Please provide either xml_file_path or xml content."
            _LOGGER.error(msg)
            raise HomeAssistantError(msg)
        
        # Validate that both are not provided
        if xml_path and xml_content:
            msg = "Both xml_file_path and xml content provided. Please provide only one."
            _LOGGER.error(msg)
            raise HomeAssistantError(msg)
        
        # If file path is provided, read the file
        if xml_path:
            # Debug logging
            _LOGGER.debug("User provided xml_path: %s", xml_path)
            _LOGGER.debug("Current working directory: %s", Path.cwd())
            _LOGGER.debug("Home Assistant config directory: %s", hass.config.config_dir)

            # Try to resolve the path relative to HA config directory if it's not absolute
            xml_path_obj = Path(xml_path)
            if not xml_path_obj.is_absolute():
                resolved_path = Path(hass.config.config_dir) / xml_path
                _LOGGER.debug("Resolved relative path to: %s", resolved_path)
            else:
                resolved_path = xml_path_obj
                _LOGGER.debug("Using absolute path: %s", resolved_path)

            # Check if file exists using async executor to avoid blocking I/O
            file_exists = await hass.async_add_executor_job(resolved_path.is_file)
            if not file_exists:
                _LOGGER.error("Specified XML file does not exist: %s", resolved_path)
                _LOGGER.error(
                    "Checked paths - Original: %s, Resolved: %s", xml_path, resolved_path
                )
                raise HomeAssistantError(f"Specified XML file does not exist: {resolved_path}")

            try:
                xml_data = await hass.async_add_executor_job(_read_file_sync, resolved_path)
            except OSError as e:
                _LOGGER.error("Failed to read XML file: %s", e)
                raise HomeAssistantError(f"Failed to read XML file: {e}") from e

            _LOGGER.info("Importing ESPI XML data via service from file: %s", resolved_path)
        else:
            # Use the XML content provided directly
            xml_data = xml_content
            _LOGGER.info("Importing ESPI XML data via service from provided XML content")

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
                # Don't store in config to avoid triggering reload (merge in memory only)
                await coordinator.async_add_xml_data(xml_data, store_in_config=False)

                _LOGGER.info(
                    "Updated coordinator and refreshed entities for entry %s",
                    entry.entry_id,
                )

                # No direct entity lookup or warning needed; coordinator update will notify all entities

            _LOGGER.info("ESPI XML import completed successfully")

        except Exception as err:
            _LOGGER.error("Failed to import ESPI XML: %s", err)
            # Re-raise as HomeAssistantError if not already
            if isinstance(err, HomeAssistantError):
                raise
            raise HomeAssistantError(f"Failed to import ESPI XML: {err}") from err

    async def delete_statistics_service(call: ServiceCall) -> None:
        """Handle the delete_statistics service call."""
        statistic_id = call.data["statistic_id"]

        _LOGGER.info("Deleting statistics for ID: %s", statistic_id)

        # Validate that the entity exists and is a Green Button entity
        entity_registry = async_get_entity_registry(hass)
        entity_entry = entity_registry.async_get(statistic_id)
        
        if entity_entry is None:
            msg = f"Entity {statistic_id} not found"
            _LOGGER.error(msg)
            raise HomeAssistantError(msg)
        
        if entity_entry.platform != DOMAIN:
            msg = f"Entity {statistic_id} is not a Green Button entity (platform: {entity_entry.platform})"
            _LOGGER.warning(msg)
            # Allow it anyway, but warn the user

        try:
            await statistics.clear_statistic(hass, statistic_id)
            _LOGGER.info("✅ Successfully deleted statistics for %s", statistic_id)
        except Exception as err:
            _LOGGER.error("❌ Failed to delete statistics for %s: %s", statistic_id, err)
            raise HomeAssistantError(f"Failed to delete statistics: {err}") from err

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
