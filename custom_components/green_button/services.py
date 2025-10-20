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
    """Set up services for the Green Button integration."""

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

            if not resolved_path.is_file():
                _LOGGER.error("Specified XML file does not exist: %s", resolved_path)
                _LOGGER.error(
                    "Checked paths - Original: %s, Resolved: %s", xml_path, resolved_path
                )

                # Additional debugging - list files in the expected directory
                config_dir = Path(hass.config.config_dir)
                green_button_dir = config_dir / "custom_components" / "green_button"
                if green_button_dir.exists():
                    _LOGGER.debug("Files in green_button directory:")
                    for file_path in green_button_dir.iterdir():
                        _LOGGER.debug("  %s", file_path.name)
                else:
                    _LOGGER.debug(
                        "Green button directory does not exist: %s", green_button_dir
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
