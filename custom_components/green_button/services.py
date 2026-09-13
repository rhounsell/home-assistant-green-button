"""Service implementations for the Green Button integration."""

from __future__ import annotations

import logging
from pathlib import Path

import voluptuous as vol
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.service import async_register_admin_service

from . import scaling, statistics
from .const import (
    CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DOMAIN,
)
from .coordinator import GreenButtonCoordinator
from .parsers import espi
from .statistic_ids import statistic_id_from_unique_id, stream_unique_id
from .xml_storage import async_get_xml_storage

_LOGGER = logging.getLogger(__name__)

SERVICE_IMPORT_ESPI_XML = "import_espi_xml"
SERVICE_DELETE_STATISTICS = "delete_statistics"
SERVICE_LOG_METER_READING_INTERVALS = "log_meter_reading_intervals"
SERVICE_LOG_STORED_XMLS = "log_stored_xmls"
SERVICE_CLEAR_STORED_XML = "clear_stored_xml"
SERVICE_RECALCULATE_COST_STATISTICS = "recalculate_cost_statistics"

CONF_CONFIG_ENTRY_ID = "config_entry_id"

IMPORT_ESPI_XML_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("xml_file_path"): cv.string,
        vol.Optional("xml"): cv.string,
    }
)

DELETE_STATISTICS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CONFIG_ENTRY_ID): cv.string,
        vol.Required("statistic_id"): cv.entity_id,
    }
)

CLEAR_STORED_XML_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("commodity"): vol.In(["electricity", "gas"]),
    }
)

RECALCULATE_COST_STATISTICS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CONFIG_ENTRY_ID): cv.string,
        vol.Optional("commodity"): vol.In(["electricity", "gas", "both"]),
    }
)


def _read_file_sync(file_path: Path) -> str:
    """Return UTF-8 text read synchronously from a file path."""
    return file_path.read_text(encoding="utf-8")


def _is_allowed_import_path(hass: HomeAssistant, path: Path) -> bool:
    """Return whether a resolved path is in config or an allowed external directory."""
    try:
        resolved_path = path.resolve()
        config_dir = Path(hass.config.config_dir).resolve()
    except OSError:
        return False
    return resolved_path.is_relative_to(config_dir) or hass.config.is_allowed_path(
        str(resolved_path)
    )


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register administrator-only services for the Green Button integration."""

    def _config_entry(call: ServiceCall) -> ConfigEntry:
        """Return the Green Button entry explicitly selected by a service call."""
        entry_id = call.data[CONF_CONFIG_ENTRY_ID]
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            raise HomeAssistantError(
                f"Config entry {entry_id} is not a Green Button entry"
            )
        return entry

    async def log_meter_reading_intervals_service(call: ServiceCall) -> None:
        """Log all meter readings, their interval block date ranges, and mapped sensor entities."""
        entity_registry = async_get_entity_registry(hass)
        for entry in [_config_entry(call)]:
            coordinator: GreenButtonCoordinator | None = (
                hass.data.get(DOMAIN, {}).get(entry.entry_id, {}).get("coordinator")
            )
            if not coordinator or not coordinator.data:
                _LOGGER.info(
                    "Entry %s: No coordinator or data available.", entry.entry_id
                )
                continue
            usage_points = coordinator.data.get("usage_points", [])
            for up_idx, usage_point in enumerate(usage_points):
                _LOGGER.info(
                    "UsagePoint %s (id=%s): %s meter readings.",
                    up_idx,
                    usage_point.id,
                    len(usage_point.meter_readings),
                )
                for mr_idx, meter_reading in enumerate(usage_point.meter_readings):
                    clean_id = (
                        meter_reading.id.split("/")[-1]
                        if "/" in meter_reading.id
                        else meter_reading.id
                    )
                    unique_id = f"{entry.entry_id}_{clean_id}"
                    entity_id = entity_registry.async_get_entity_id(
                        "sensor", DOMAIN, unique_id
                    )
                    _LOGGER.info(
                        "  MeterReading %s (id=%s): mapped entity_id=%s",
                        mr_idx,
                        meter_reading.id,
                        entity_id,
                    )
                    for ib_idx, interval_block in enumerate(
                        meter_reading.interval_blocks
                    ):
                        start = interval_block.start.isoformat()
                        end = (
                            interval_block.start + interval_block.duration
                        ).isoformat()
                        _LOGGER.info(
                            "    IntervalBlock %s: start=%s, end=%s, readings=%s",
                            ib_idx,
                            start,
                            end,
                            len(interval_block.interval_readings),
                        )

    async def log_stored_xmls_service(call: ServiceCall) -> None:
        """Log archived XML labels, sizes, and parsed coverage for the selected entry."""

        for entry in [_config_entry(call)]:
            _LOGGER.info("=" * 60)
            _LOGGER.info("Config Entry: %s (entry_id: %s)", entry.title, entry.entry_id)

            # Load from new separate storage file
            xml_storage = await async_get_xml_storage(hass, entry.entry_id)
            stored_xmls = xml_storage.get_stored_xmls()

            # Fall back to config entry for backwards compatibility
            if not stored_xmls:
                stored_xmls = entry.data.get("stored_xmls", [])
                legacy_xml = entry.data.get("xml")

                if legacy_xml and not stored_xmls:
                    _LOGGER.info("  Found legacy single XML storage (not yet migrated)")
                    stored_xmls = [{"label": "legacy", "xmls": [legacy_xml]}]

            if not stored_xmls:
                _LOGGER.info("  No stored XMLs found")
                continue

            _LOGGER.info("  Found %d label(s)", len(stored_xmls))

            for idx, xml_entry in enumerate(stored_xmls):
                label = xml_entry.get("label", f"xml_{idx}")

                # Handle both old format (single "xml") and new format ("xmls" list)
                xml_list = xml_entry.get("xmls", [])
                if not xml_list and "xml" in xml_entry:
                    xml_list = [xml_entry["xml"]]

                total_size = sum(len(x) for x in xml_list if x)
                _LOGGER.info(
                    "  [%d] Label: '%s', %d XML(s), Total size: %d bytes",
                    idx,
                    label,
                    len(xml_list),
                    total_size,
                )

                for xml_idx, xml_data in enumerate(xml_list):
                    if not xml_data:
                        continue

                    _LOGGER.info("      XML[%d]: %d bytes", xml_idx, len(xml_data))

                    try:
                        # Parse XML to get date ranges
                        usage_points = await hass.async_add_executor_job(
                            espi.parse_xml, xml_data
                        )
                        for up in usage_points:
                            _LOGGER.info("        UsagePoint: %s", up.id)
                            for mr in up.meter_readings:
                                all_readings = [
                                    ir
                                    for ib in mr.interval_blocks
                                    for ir in ib.interval_readings
                                ]
                                if all_readings:
                                    min_start = min(ir.start for ir in all_readings)
                                    max_end = max(ir.end for ir in all_readings)
                                    _LOGGER.info(
                                        "          MeterReading %s: %s to %s (%d readings)",
                                        mr.id.split("/")[-1] if "/" in mr.id else mr.id,
                                        min_start.strftime("%Y-%m-%d %H:%M"),
                                        max_end.strftime("%Y-%m-%d %H:%M"),
                                        len(all_readings),
                                    )
                                else:
                                    _LOGGER.info(
                                        "          MeterReading %s: NO INTERVAL READINGS",
                                        mr.id.split("/")[-1] if "/" in mr.id else mr.id,
                                    )
                            if up.usage_summaries:
                                _LOGGER.info(
                                    "        UsageSummaries: %d",
                                    len(up.usage_summaries),
                                )
                    except espi.EspiXmlParseError as err:
                        _LOGGER.error("        Failed to parse XML: %s", err)

            _LOGGER.info("=" * 60)

    async def import_espi_xml_service(call: ServiceCall) -> None:
        """Validate and import ESPI XML for the selected Green Button entry."""
        xml_path = call.data.get("xml_file_path", "").strip()
        xml_content = call.data.get("xml", "").strip()

        # Validate that at least one is provided
        if not xml_path and not xml_content:
            msg = "No XML data provided. Please provide either xml_file_path or xml content."
            _LOGGER.error(msg)
            raise HomeAssistantError(msg)

        # Validate that both are not provided
        if xml_path and xml_content:
            msg = (
                "Both xml_file_path and xml content provided. Please provide only one."
            )
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

            path_is_allowed = await hass.async_add_executor_job(
                _is_allowed_import_path, hass, resolved_path
            )
            if not path_is_allowed:
                raise HomeAssistantError(
                    "XML file path is not allowed; use the Home Assistant config "
                    "directory or add its directory to allowlist_external_dirs"
                )

            # Check if file exists using async executor to avoid blocking I/O
            file_exists = await hass.async_add_executor_job(resolved_path.is_file)
            if not file_exists:
                _LOGGER.error("Specified XML file does not exist: %s", resolved_path)
                _LOGGER.error(
                    "Checked paths - Original: %s, Resolved: %s",
                    xml_path,
                    resolved_path,
                )
                raise HomeAssistantError(
                    f"Specified XML file does not exist: {resolved_path}"
                )

            try:
                xml_data = await hass.async_add_executor_job(
                    _read_file_sync, resolved_path
                )
            except OSError as e:
                _LOGGER.error("Failed to read XML file: %s", e)
                raise HomeAssistantError(f"Failed to read XML file: {e}") from e

            _LOGGER.info(
                "Importing ESPI XML data via service from file: %s", resolved_path
            )
        else:
            # Use the XML content provided directly
            xml_data = xml_content
            _LOGGER.info(
                "Importing ESPI XML data via service from provided XML content"
            )

        try:
            # Process the XML data for the explicitly selected entry only.
            for entry in [_config_entry(call)]:
                coordinator_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
                coordinator: GreenButtonCoordinator | None = coordinator_data.get(
                    "coordinator"
                )

                if not coordinator:
                    _LOGGER.warning("No coordinator found for entry %s", entry.entry_id)
                    continue

                # Let the coordinator handle all data parsing and updates
                # Label is auto-detected from XML content (electricity or gas)
                # Always store in config entry for persistence across restarts
                _LOGGER.info(
                    "[SERVICE IMPORT] Importing XML data for entry %s (size: %d bytes)",
                    entry.entry_id,
                    len(xml_data),
                )
                report = await coordinator.async_add_xml_data(
                    xml_data, store_in_config=True
                )

                _LOGGER.info(
                    "[SERVICE IMPORT] Entry %s accepted %d interval readings and skipped %d",
                    entry.entry_id,
                    report.accepted_readings,
                    report.skipped_readings,
                )

                # No direct entity lookup or warning needed; coordinator update will notify all entities

            _LOGGER.info(
                "ESPI XML import completed successfully (label auto-detected from commodity type)"
            )

        except Exception as err:
            _LOGGER.error("Failed to import ESPI XML: %s", err)
            # Re-raise as HomeAssistantError if not already
            if isinstance(err, HomeAssistantError):
                raise
            raise HomeAssistantError(f"Failed to import ESPI XML: {err}") from err

    async def delete_statistics_service(call: ServiceCall) -> None:
        """Delete an external statistic owned by the selected Green Button entry."""
        statistic_id = call.data["statistic_id"]
        config_entry = _config_entry(call)

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
            raise HomeAssistantError(msg)

        if entity_entry.config_entry_id != config_entry.entry_id:
            raise HomeAssistantError(
                f"Entity {statistic_id} does not belong to config entry "
                f"{config_entry.entry_id}"
            )

        statistic_id = statistic_id_from_unique_id(entity_entry.unique_id)

        try:
            await statistics.clear_statistic(hass, statistic_id)
            _LOGGER.info("✅ Successfully deleted statistics for %s", statistic_id)
        except Exception as err:
            _LOGGER.error(
                "❌ Failed to delete statistics for %s: %s", statistic_id, err
            )
            raise HomeAssistantError(f"Failed to delete statistics: {err}") from err

    async def clear_stored_xml_service(call: ServiceCall) -> None:
        """Clear archived XML for the selected entry and resync its stored usage data."""

        # commodity maps directly to label (electricity or gas)
        label_to_clear = call.data.get("commodity")

        for entry in [_config_entry(call)]:
            # Use new separate storage file
            xml_storage = await async_get_xml_storage(hass, entry.entry_id)
            removed_count, _remaining_count = await xml_storage.async_clear_label(
                label_to_clear
            )
            coordinator: GreenButtonCoordinator | None = (
                hass.data.get(DOMAIN, {}).get(entry.entry_id, {}).get("coordinator")
            )
            if coordinator is not None:
                await coordinator.async_sync_stored_usage_points()

            if label_to_clear:
                if removed_count > 0:
                    _LOGGER.info(
                        "✅ Cleared stored XML with label '%s' from entry %s",
                        label_to_clear,
                        entry.title,
                    )
                else:
                    _LOGGER.warning(
                        "No stored XML found with label '%s' in entry %s",
                        label_to_clear,
                        entry.title,
                    )
            else:
                if removed_count > 0:
                    _LOGGER.info(
                        "✅ Cleared ALL %d stored XML label(s) from entry %s",
                        removed_count,
                        entry.title,
                    )
                else:
                    _LOGGER.info("No stored XMLs found for entry %s", entry.entry_id)

    async def recalculate_cost_statistics_service(call: ServiceCall) -> None:
        """Rebuild selected electricity or gas cost statistics from archived source data."""
        commodity = call.data.get("commodity", "both")

        _LOGGER.info("Recalculating cost statistics for commodity: %s", commodity)

        recalculated_count = 0

        for entry in [_config_entry(call)]:
            coordinator_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
            coordinator: GreenButtonCoordinator | None = coordinator_data.get(
                "coordinator"
            )

            if not coordinator:
                _LOGGER.warning("No coordinator for entry %s", entry.title)
                continue

            usage_points = await coordinator.async_reconstruct_stored_usage_points()

            if not usage_points:
                _LOGGER.warning(
                    "No canonical source data found for entry %s", entry.title
                )
                continue

            # Get the entity registry to find sensor entities
            entity_registry = async_get_entity_registry(hass)

            # Log usage point types for debugging
            gas_count = sum(
                1
                for up in usage_points
                if up.sensor_device_class == SensorDeviceClass.GAS
            )
            elec_count = sum(
                1
                for up in usage_points
                if up.sensor_device_class != SensorDeviceClass.GAS
            )
            _LOGGER.info(
                "Found %d usage point(s): %d electricity, %d gas",
                len(usage_points),
                elec_count,
                gas_count,
            )

            for usage_point in usage_points:
                is_gas = usage_point.sensor_device_class == SensorDeviceClass.GAS

                # Skip if commodity filter doesn't match
                if commodity == "electricity" and is_gas:
                    _LOGGER.debug(
                        "Skipping gas usage point %s (filtering for electricity only)",
                        usage_point.id,
                    )
                    continue
                if commodity == "gas" and not is_gas:
                    _LOGGER.debug(
                        "Skipping electricity usage point %s (filtering for gas only)",
                        usage_point.id,
                    )
                    continue

                # Find cost sensor entities for this usage point
                if is_gas:
                    # Gas cost sensor
                    usage_allocation_mode = (
                        entry.options.get("gas_usage_allocation")
                        or entry.data.get("gas_usage_allocation")
                        or "daily_readings"
                    )
                    cost_allocation_mode = (
                        entry.options.get("gas_cost_allocation")
                        or entry.data.get("gas_cost_allocation")
                        or "pro_rate_daily"
                    )

                    eligible_mrs = [
                        mr
                        for mr in usage_point.meter_readings
                        if mr.interval_blocks
                        and any(
                            ir.value is not None
                            for blk in mr.interval_blocks
                            for ir in blk.interval_readings
                        )
                    ]

                    # Determine the meter_reading_id using the same selection
                    # policy as sensor setup.
                    if (
                        usage_allocation_mode == "monthly_increment"
                        and usage_point.usage_summaries
                    ):
                        meter_reading_id = (
                            eligible_mrs[0].id
                            if len(eligible_mrs) == 1
                            else usage_point.id
                        )
                    elif eligible_mrs:
                        primary_mr = min(eligible_mrs, key=lambda mr: mr.id)
                        meter_reading_id = primary_mr.id
                    else:
                        _LOGGER.debug("No gas data available for %s", usage_point.id)
                        continue

                    # Find the gas cost sensor entity
                    unique_id = stream_unique_id(
                        entry.entry_id,
                        usage_point.id,
                        meter_reading_id,
                        "_gas_cost",
                    )
                    entity_id = entity_registry.async_get_entity_id(
                        "sensor", DOMAIN, unique_id
                    )

                    if not entity_id:
                        _LOGGER.warning("Gas cost sensor not found for %s", unique_id)
                        continue

                    # Get the entity state object
                    entity_state = hass.states.get(entity_id)
                    if not entity_state:
                        _LOGGER.warning(
                            "Gas cost sensor state not found for %s", entity_id
                        )
                        continue

                    # Trigger statistics recalculation
                    _LOGGER.info("Recalculating gas cost statistics for %s", entity_id)

                    # Get gas cost multiplier
                    _LOGGER.debug("entry.options: %s", entry.options)
                    _LOGGER.debug("entry.data: %s", entry.data)

                    gas_multiplier = scaling.configured_multiplier(
                        entry,
                        CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER,
                        DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
                    )

                    # Get summaries
                    summaries = list(usage_point.usage_summaries)

                    # A single detailed stream remains the source for daily cost
                    # proration even when gas usage is published monthly.
                    meter_reading = next(
                        (
                            mr
                            for mr in usage_point.meter_readings
                            if mr.id == meter_reading_id
                        ),
                        None,
                    )
                    if (
                        cost_allocation_mode == "pro_rate_daily"
                        and meter_reading is None
                    ):
                        _LOGGER.info(
                            "Using monthly gas cost increments for %s because no "
                            "detailed meter stream is available for proration",
                            entity_id,
                        )
                        cost_allocation_mode = "monthly_increment"

                    # Create a mock entity object for statistics
                    class MockGasCostEntity:
                        """Mock entity for gas cost statistics recalculation."""

                        def __init__(
                            self, entity_id: str, name: str, unit: str, unique_id: str
                        ):
                            """Initialize metadata needed to update a gas cost statistic."""
                            self.entity_id = entity_id
                            self._statistic_id = statistic_id_from_unique_id(unique_id)
                            self.name = name
                            self._attr_native_unit_of_measurement = unit

                        @property
                        def long_term_statistics_id(self) -> str:
                            """Return the external statistic ID selected for recalculation."""
                            return self._statistic_id

                        @property
                        def native_unit_of_measurement(self) -> str:
                            """Return the currency unit used by the cost statistic."""
                            return self._attr_native_unit_of_measurement

                    currency = (
                        summaries[0].currency
                        if summaries
                        else meter_reading.reading_type.currency
                        if meter_reading
                        else "CAD"
                    )
                    mock_entity = MockGasCostEntity(
                        entity_id, entity_state.name or "Gas Cost", currency, unique_id
                    )

                    try:
                        await statistics.update_gas_cost_statistics(
                            hass,
                            mock_entity,
                            meter_reading,
                            summaries,
                            allocation_mode=cost_allocation_mode,
                            gas_cost_multiplier=gas_multiplier,
                            merge_with_existing=False,  # Recalculate ALL from scratch
                        )
                        _LOGGER.info(
                            "✅ Recalculated gas cost statistics for %s", entity_id
                        )
                        recalculated_count += 1
                    except ValueError as err:
                        _LOGGER.error(
                            "❌ Failed to recalculate gas cost statistics for %s: %s",
                            entity_id,
                            err,
                        )

                else:
                    # Electricity cost sensor
                    _LOGGER.info(
                        "Examining electricity UsagePoint %s: found %d total meter readings",
                        usage_point.id,
                        len(usage_point.meter_readings),
                    )

                    # Log details about each meter reading for debugging
                    for idx, mr in enumerate(usage_point.meter_readings, 1):
                        num_blocks = (
                            len(mr.interval_blocks) if mr.interval_blocks else 0
                        )
                        total_intervals = (
                            sum(
                                len(blk.interval_readings) for blk in mr.interval_blocks
                            )
                            if mr.interval_blocks
                            else 0
                        )
                        has_cost_data = (
                            any(
                                hasattr(ir, "cost") and ir.cost is not None
                                for blk in mr.interval_blocks
                                for ir in blk.interval_readings
                            )
                            if mr.interval_blocks
                            else False
                        )
                        _LOGGER.info(
                            "  Meter reading %d/%d: ID=%s, blocks=%d, intervals=%d, has_cost=%s",
                            idx,
                            len(usage_point.meter_readings),
                            mr.id.split("/")[-1] if "/" in mr.id else mr.id,
                            num_blocks,
                            total_intervals,
                            has_cost_data,
                        )

                    # Find eligible meter readings (cost sensors check for 'cost' attribute, not 'value')
                    eligible_electric_mrs = [
                        mr
                        for mr in usage_point.meter_readings
                        if mr.interval_blocks
                        and any(
                            hasattr(ir, "cost") and ir.cost is not None
                            for blk in mr.interval_blocks
                            for ir in blk.interval_readings
                        )
                    ]

                    if not eligible_electric_mrs:
                        _LOGGER.info(
                            "Skipping electricity UsagePoint %s: no eligible meter readings with cost data",
                            usage_point.id,
                        )
                        continue

                    multiplier = scaling.configured_multiplier(
                        entry,
                        CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
                        DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
                    )

                    class MockElectricityCostEntity:
                        """Mock entity for electricity cost statistics recalculation."""

                        def __init__(
                            self, entity_id: str, name: str, unit: str, unique_id: str
                        ):
                            """Initialize metadata needed to update an electricity cost statistic."""
                            self.entity_id = entity_id
                            self._statistic_id = statistic_id_from_unique_id(unique_id)
                            self.name = name
                            self._attr_native_unit_of_measurement = unit

                        @property
                        def long_term_statistics_id(self) -> str:
                            """Return the external statistic ID selected for recalculation."""
                            return self._statistic_id

                        @property
                        def native_unit_of_measurement(self) -> str:
                            """Return the currency unit used by the cost statistic."""
                            return self._attr_native_unit_of_measurement

                    for meter_reading in sorted(
                        eligible_electric_mrs, key=lambda mr: mr.id
                    ):
                        unique_id = stream_unique_id(
                            entry.entry_id,
                            usage_point.id,
                            meter_reading.id,
                            "_cost",
                        )
                        entity_id = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, unique_id
                        )
                        if not entity_id:
                            _LOGGER.warning(
                                "Electricity cost sensor not found for %s", unique_id
                            )
                            continue

                        entity_state = hass.states.get(entity_id)
                        if not entity_state:
                            _LOGGER.warning(
                                "Electricity cost sensor state not found for %s",
                                entity_id,
                            )
                            continue

                        _LOGGER.info(
                            "Recalculating electricity cost statistics for %s",
                            entity_id,
                        )
                        _LOGGER.info(
                            "Processing canonical meter reading %s for entity %s",
                            meter_reading.id,
                            entity_id,
                        )
                        mock_entity = MockElectricityCostEntity(
                            entity_id,
                            entity_state.name or "Electricity Cost",
                            meter_reading.reading_type.currency,
                            unique_id,
                        )
                        try:
                            await statistics.update_cost_statistics(
                                hass,
                                mock_entity,
                                statistics.CostDataExtractor(multiplier),
                                meter_reading,
                                merge_with_existing=False,
                            )
                            _LOGGER.info(
                                "✅ Recalculated electricity cost statistics for %s",
                                entity_id,
                            )
                            recalculated_count += 1
                        except ValueError as err:
                            _LOGGER.error(
                                "❌ Failed to recalculate electricity cost statistics for %s: %s",
                                entity_id,
                                err,
                            )

        if recalculated_count > 0:
            _LOGGER.info(
                "✅ Successfully recalculated %d cost statistic(s)", recalculated_count
            )
        else:
            _LOGGER.warning("No cost statistics were recalculated")

    # Register services
    try:
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

        hass.services.async_register(
            DOMAIN,
            SERVICE_LOG_METER_READING_INTERVALS,
            log_meter_reading_intervals_service,
            schema=vol.Schema({vol.Required(CONF_CONFIG_ENTRY_ID): cv.string}),
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_LOG_STORED_XMLS,
            log_stored_xmls_service,
            schema=vol.Schema({vol.Required(CONF_CONFIG_ENTRY_ID): cv.string}),
        )

        async_register_admin_service(
            hass,
            DOMAIN,
            SERVICE_CLEAR_STORED_XML,
            clear_stored_xml_service,
            schema=CLEAR_STORED_XML_SCHEMA,
        )

        async_register_admin_service(
            hass,
            DOMAIN,
            SERVICE_RECALCULATE_COST_STATISTICS,
            recalculate_cost_statistics_service,
            schema=RECALCULATE_COST_STATISTICS_SCHEMA,
        )

        _LOGGER.info("Green Button services registered successfully")
    except Exception as err:
        _LOGGER.error("Failed to register Green Button services: %s", err)
        raise


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unregister services for the Green Button integration."""
    hass.services.async_remove(DOMAIN, SERVICE_IMPORT_ESPI_XML)
    hass.services.async_remove(DOMAIN, SERVICE_DELETE_STATISTICS)
    hass.services.async_remove(DOMAIN, SERVICE_LOG_METER_READING_INTERVALS)
    hass.services.async_remove(DOMAIN, SERVICE_LOG_STORED_XMLS)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_STORED_XML)
    hass.services.async_remove(DOMAIN, SERVICE_RECALCULATE_COST_STATISTICS)
    _LOGGER.info("Green Button services unloaded")
