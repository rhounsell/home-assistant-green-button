"""Service implementations for the Green Button integration."""

from __future__ import annotations

import logging
from pathlib import Path

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components.sensor import SensorDeviceClass
from .parsers import espi
from .xml_storage import async_get_xml_storage
from . import statistics
from .const import (
    DOMAIN,
    CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER,
)
from .coordinator import GreenButtonCoordinator

_LOGGER = logging.getLogger(__name__)

SERVICE_IMPORT_ESPI_XML = "import_espi_xml"
SERVICE_DELETE_STATISTICS = "delete_statistics"
SERVICE_LOG_METER_READING_INTERVALS = "log_meter_reading_intervals"
SERVICE_LOG_STORED_XMLS = "log_stored_xmls"
SERVICE_CLEAR_STORED_XML = "clear_stored_xml"
SERVICE_RECALCULATE_COST_STATISTICS = "recalculate_cost_statistics"

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

CLEAR_STORED_XML_SCHEMA = vol.Schema(
    {
        vol.Optional("commodity"): vol.In(["electricity", "gas"]),
    }
)

RECALCULATE_COST_STATISTICS_SCHEMA = vol.Schema(
    {
        vol.Optional("commodity"): vol.In(["electricity", "gas", "both"]),
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
                _LOGGER.info("Entry %s: No coordinator or data available.", entry.entry_id)
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
                    clean_id = meter_reading.id.split("/")[-1] if "/" in meter_reading.id else meter_reading.id
                    unique_id = f"{entry.entry_id}_{clean_id}"
                    entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)
                    _LOGGER.info("  MeterReading %s (id=%s): mapped entity_id=%s", mr_idx, meter_reading.id, entity_id)
                    for ib_idx, interval_block in enumerate(meter_reading.interval_blocks):
                        start = interval_block.start.isoformat()
                        end = (interval_block.start + interval_block.duration).isoformat()
                        _LOGGER.info(
                            "    IntervalBlock %s: start=%s, end=%s, readings=%s",
                            ib_idx,
                            start,
                            end,
                            len(interval_block.interval_readings),
                        )

    async def log_stored_xmls_service(call: ServiceCall) -> None:
        """Log information about stored XMLs in storage files."""

        entries = list(hass.config_entries.async_entries(DOMAIN))
        for entry in entries:
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
                _LOGGER.info("  [%d] Label: '%s', %d XML(s), Total size: %d bytes", idx, label, len(xml_list), total_size)

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
                                    ir for ib in mr.interval_blocks for ir in ib.interval_readings
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
                                _LOGGER.info("        UsageSummaries: %d", len(up.usage_summaries))
                    except Exception as e:
                        _LOGGER.error("        Failed to parse XML: %s", e)

            _LOGGER.info("=" * 60)


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
                # Label is auto-detected from XML content (electricity or gas)
                # Always store in config entry for persistence across restarts
                _LOGGER.info("[SERVICE IMPORT] Importing XML data for entry %s (size: %d bytes)", 
                            entry.entry_id, len(xml_data))
                await coordinator.async_add_xml_data(xml_data, store_in_config=True)

                _LOGGER.info(
                    "[SERVICE IMPORT] Updated coordinator and refreshed entities for entry %s",
                    entry.entry_id,
                )

                # No direct entity lookup or warning needed; coordinator update will notify all entities

            _LOGGER.info("ESPI XML import completed successfully (label auto-detected from commodity type)")

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

    async def clear_stored_xml_service(call: ServiceCall) -> None:
        """Handle the clear_stored_xml service call."""

        # commodity maps directly to label (electricity or gas)
        label_to_clear = call.data.get("commodity")

        entries = list(hass.config_entries.async_entries(DOMAIN))

        if not entries:
            _LOGGER.warning("No Green Button integrations found")
            return

        for entry in entries:
            # Use new separate storage file
            xml_storage = await async_get_xml_storage(hass, entry.entry_id)
            removed_count, remaining_count = await xml_storage.async_clear_label(label_to_clear)

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
        """Handle the recalculate_cost_statistics service call."""
        commodity = call.data.get("commodity", "both")

        _LOGGER.info("Recalculating cost statistics for commodity: %s", commodity)

        entries = list(hass.config_entries.async_entries(DOMAIN))

        if not entries:
            _LOGGER.warning("No Green Button integrations found")
            return

        recalculated_count = 0

        for entry in entries:
            coordinator_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
            coordinator: GreenButtonCoordinator | None = coordinator_data.get("coordinator")

            if not coordinator:
                _LOGGER.warning("No coordinator for entry %s", entry.title)
                continue

            # Reload all data from XML storage to get complete historical data
            # (coordinator.data only has cached/recent meter readings)
            _LOGGER.info("Reloading all XML data from storage for entry %s", entry.title)
            
            from . import xml_storage as xml_storage_module
            
            xml_storage = await xml_storage_module.async_get_xml_storage(hass, entry.entry_id)
            stored_xmls = xml_storage.get_stored_xmls()
            
            if not stored_xmls:
                _LOGGER.warning("No stored XML data found for entry %s", entry.title)
                continue
            
            # Parse all stored XMLs to get complete usage points
            usage_points = []
            _LOGGER.info("Loading %d stored label(s) from XML storage for recalculation", len(stored_xmls))
            
            xml_count = 0
            for idx, xml_entry in enumerate(stored_xmls):
                label = xml_entry.get("label", f"xml_{idx}")
                
                # Handle both old format (single "xml") and new format ("xmls" list)
                xml_list = xml_entry.get("xmls", [])
                if not xml_list and "xml" in xml_entry:
                    xml_list = [xml_entry["xml"]]
                
                if not xml_list:
                    _LOGGER.warning("Skipping empty XML entry with label '%s'", label)
                    continue
                
                for xml_idx, xml_data in enumerate(xml_list):
                    if not xml_data:
                        continue
                    
                    xml_count += 1
                    _LOGGER.debug("Parsing stored XML '%s' [%d/%d]", label, xml_idx + 1, len(xml_list))
                    
                    # Parse XML in executor to avoid blocking
                    parsed_usage_points = await hass.async_add_executor_job(
                        espi.parse_xml, xml_data
                    )
                    
                    if parsed_usage_points:
                        # Merge with existing usage points
                        if not usage_points:
                            usage_points = parsed_usage_points
                        else:
                            # Merge meter readings by usage point ID
                            from dataclasses import replace
                            
                            for new_up in parsed_usage_points:
                                existing_idx = next(
                                    (i for i, up in enumerate(usage_points) if up.id == new_up.id),
                                    None
                                )
                                if existing_idx is not None:
                                    existing_up = usage_points[existing_idx]
                                    # Combine meter readings and usage summaries (UsagePoint is frozen, must use replace)
                                    combined_mrs = list(existing_up.meter_readings) + list(new_up.meter_readings)
                                    summary_map = {
                                        us.id: us for us in existing_up.usage_summaries if getattr(us, "id", None)
                                    }
                                    no_id_summaries = [
                                        us for us in existing_up.usage_summaries if not getattr(us, "id", None)
                                    ]
                                    skipped_duplicates = 0
                                    for us in new_up.usage_summaries:
                                        us_id = getattr(us, "id", None)
                                        if us_id:
                                            if us_id in summary_map:
                                                skipped_duplicates += 1
                                                continue
                                            summary_map[us_id] = us
                                        else:
                                            no_id_summaries.append(us)
                                    combined_summaries = list(summary_map.values()) + no_id_summaries
                                    combined_summaries.sort(
                                        key=lambda us: (
                                            getattr(us, "start", None) or 0,
                                            str(getattr(us, "duration", None) or ""),
                                            getattr(us, "id", "") or "",
                                        )
                                    )
                                    _LOGGER.warning(
                                        "[GB DEBUG] recalculation merge usage_point=%s combined_summaries=%d existing=%d new=%d skipped_duplicates=%d",
                                        new_up.id,
                                        len(combined_summaries),
                                        len(existing_up.usage_summaries),
                                        len(new_up.usage_summaries),
                                        skipped_duplicates,
                                    )
                                    
                                    # Create new UsagePoint with combined data
                                    merged_up = replace(
                                        existing_up,
                                        meter_readings=combined_mrs,
                                        usage_summaries=combined_summaries
                                    )
                                    usage_points[existing_idx] = merged_up
                                    
                                    _LOGGER.debug(
                                        "Merged UsagePoint %s: %d meter readings (%d + %d), %d summaries (%d + %d)",
                                        new_up.id.split("/")[-1] if "/" in new_up.id else new_up.id,
                                        len(combined_mrs), len(existing_up.meter_readings), len(new_up.meter_readings),
                                        len(combined_summaries), len(existing_up.usage_summaries), len(new_up.usage_summaries)
                                    )
                                else:
                                    # Add new usage point
                                    usage_points.append(new_up)
            
            _LOGGER.info("Loaded %d usage point(s) from %d XML(s) for recalculation", 
                        len(usage_points), xml_count)
            
            if not usage_points:
                _LOGGER.warning("No usage points found after parsing XMLs for entry %s", entry.title)
                continue

            # Get the entity registry to find sensor entities
            entity_registry = async_get_entity_registry(hass)
            
            # Log usage point types for debugging
            gas_count = sum(1 for up in usage_points if up.sensor_device_class == SensorDeviceClass.GAS)
            elec_count = sum(1 for up in usage_points if up.sensor_device_class != SensorDeviceClass.GAS)
            _LOGGER.info("Found %d usage point(s): %d electricity, %d gas", 
                        len(usage_points), elec_count, gas_count)

            for usage_point in usage_points:
                is_gas = usage_point.sensor_device_class == SensorDeviceClass.GAS

                # Skip if commodity filter doesn't match
                if commodity == "electricity" and is_gas:
                    _LOGGER.debug("Skipping gas usage point %s (filtering for electricity only)", 
                                usage_point.id)
                    continue
                if commodity == "gas" and not is_gas:
                    _LOGGER.debug("Skipping electricity usage point %s (filtering for gas only)", 
                                usage_point.id)
                    continue

                # Find cost sensor entities for this usage point
                if is_gas:
                    # Gas cost sensor
                    allocation_mode = (
                        entry.options.get("gas_usage_allocation")
                        or entry.data.get("gas_usage_allocation")
                        or "daily_readings"
                    )

                    # Determine the meter_reading_id based on allocation mode
                    if allocation_mode == "monthly_increment" and usage_point.usage_summaries:
                        meter_reading_id = usage_point.id
                    elif usage_point.meter_readings:
                        # Find the primary meter reading (same logic as sensor creation)
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
                        if not eligible_mrs:
                            _LOGGER.debug("No eligible gas meter readings for %s", usage_point.id)
                            continue
                        primary_mr = sorted(eligible_mrs, key=lambda mr: mr.id)[0]
                        meter_reading_id = primary_mr.id
                    else:
                        _LOGGER.debug("No gas data available for %s", usage_point.id)
                        continue

                    # Find the gas cost sensor entity
                    clean_id = meter_reading_id.split("/")[-1] if "/" in meter_reading_id else meter_reading_id
                    unique_id = f"{entry.entry_id}_{clean_id}_gas_cost"
                    entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)

                    if not entity_id:
                        _LOGGER.warning("Gas cost sensor not found for %s", unique_id)
                        continue

                    # Get the entity state object
                    entity_state = hass.states.get(entity_id)
                    if not entity_state:
                        _LOGGER.warning("Gas cost sensor state not found for %s", entity_id)
                        continue

                    # Trigger statistics recalculation
                    _LOGGER.info("Recalculating gas cost statistics for %s", entity_id)

                    # Get gas cost multiplier
                    _LOGGER.debug("entry.options: %s", entry.options)
                    _LOGGER.debug("entry.data: %s", entry.data)
                    
                    raw_gas_multiplier = (
                        entry.options.get(CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER)
                        if entry.options.get(CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER) is not None
                        else entry.data.get(CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER)
                    )
                    gas_multiplier = int(raw_gas_multiplier) if raw_gas_multiplier is not None else -5
                    
                    _LOGGER.info("Gas cost multiplier for %s: raw=%s, final=%s", 
                                entity_id, raw_gas_multiplier, gas_multiplier)

                    # Get summaries
                    summaries = list(usage_point.usage_summaries)

                    # Get meter reading if available
                    meter_reading = None
                    if allocation_mode != "monthly_increment" or not usage_point.usage_summaries:
                        for mr in usage_point.meter_readings:
                            if mr.id == meter_reading_id:
                                meter_reading = mr
                                break

                    # Create a mock entity object for statistics
                    class MockGasCostEntity:
                        """Mock entity for gas cost statistics recalculation."""
                        def __init__(self, entity_id: str, name: str, unit: str):
                            self.entity_id = entity_id
                            self.name = name
                            self._attr_native_unit_of_measurement = unit

                        @property
                        def long_term_statistics_id(self) -> str:
                            return self.entity_id

                        @property
                        def native_unit_of_measurement(self) -> str:
                            return self._attr_native_unit_of_measurement

                    mock_entity = MockGasCostEntity(entity_id, entity_state.name or "Gas Cost", "CAD")

                    try:
                        cost_allocation_mode = (
                            entry.options.get("gas_cost_allocation")
                            or entry.data.get("gas_cost_allocation")
                            or "pro_rate_daily"
                        )
                        await statistics.update_gas_cost_statistics(
                            hass,
                            mock_entity,
                            meter_reading,
                            summaries,
                            allocation_mode=cost_allocation_mode,
                            gas_cost_multiplier=gas_multiplier,
                            merge_with_existing=False,  # Recalculate ALL from scratch
                        )
                        _LOGGER.info("✅ Recalculated gas cost statistics for %s", entity_id)
                        recalculated_count += 1
                    except Exception as e:
                        _LOGGER.error("❌ Failed to recalculate gas cost statistics for %s: %s", entity_id, e)

                else:
                    # Electricity cost sensor
                    _LOGGER.info("Examining electricity UsagePoint %s: found %d total meter readings", 
                                usage_point.id, len(usage_point.meter_readings))
                    
                    # Log details about each meter reading for debugging
                    for idx, mr in enumerate(usage_point.meter_readings, 1):
                        num_blocks = len(mr.interval_blocks) if mr.interval_blocks else 0
                        total_intervals = sum(len(blk.interval_readings) for blk in mr.interval_blocks) if mr.interval_blocks else 0
                        has_cost_data = any(
                            hasattr(ir, "cost") and ir.cost is not None
                            for blk in mr.interval_blocks
                            for ir in blk.interval_readings
                        ) if mr.interval_blocks else False
                        _LOGGER.info("  Meter reading %d/%d: ID=%s, blocks=%d, intervals=%d, has_cost=%s", 
                                    idx, len(usage_point.meter_readings), 
                                    mr.id.split("/")[-1] if "/" in mr.id else mr.id,
                                    num_blocks, total_intervals, has_cost_data)
                    
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
                        _LOGGER.info("Skipping electricity UsagePoint %s: no eligible meter readings with cost data", usage_point.id)
                        continue

                    # Pick the primary meter reading (matches sensor creation: sorted by ID, first one)
                    primary_electric_mr = sorted(eligible_electric_mrs, key=lambda mr: mr.id)[0]
                    
                    # Get the cost sensor entity ID based on the primary meter reading
                    clean_id = primary_electric_mr.id.split("/")[-1] if "/" in primary_electric_mr.id else primary_electric_mr.id
                    unique_id = f"{entry.entry_id}_{clean_id}_cost"
                    entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)

                    if not entity_id:
                        _LOGGER.warning("Electricity cost sensor not found for %s", unique_id)
                        continue

                    # Get the entity state object
                    entity_state = hass.states.get(entity_id)
                    if not entity_state:
                        _LOGGER.warning("Electricity cost sensor state not found for %s", entity_id)
                        continue

                    # Get electricity cost multiplier
                    raw_multiplier = (
                        entry.options.get(CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER)
                        if entry.options.get(CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER) is not None
                        else entry.data.get(CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER)
                    )
                    multiplier = int(raw_multiplier) if raw_multiplier is not None else -5
                    
                    _LOGGER.info("Recalculating electricity cost statistics for %s", entity_id)
                    _LOGGER.info("Electricity cost multiplier: raw=%s, final=%s", raw_multiplier, multiplier)
                    _LOGGER.info("Processing %d meter reading(s) for entity %s", len(eligible_electric_mrs), entity_id)

                    # Create a mock entity object for statistics
                    class MockElectricityCostEntity:
                        """Mock entity for electricity cost statistics recalculation."""
                        def __init__(self, entity_id: str, name: str, unit: str):
                            self.entity_id = entity_id
                            self.name = name
                            self._attr_native_unit_of_measurement = unit

                        @property
                        def long_term_statistics_id(self) -> str:
                            return self.entity_id

                        @property
                        def native_unit_of_measurement(self) -> str:
                            return self._attr_native_unit_of_measurement

                    mock_entity = MockElectricityCostEntity(entity_id, entity_state.name or "Electricity Cost", "CAD")

                    try:
                        # Process all meter readings for this entity
                        for idx, meter_reading in enumerate(eligible_electric_mrs, 1):
                            is_first = (idx == 1)
                            
                            _LOGGER.info("Processing meter reading %d of %d for %s (merge_with_existing=%s)", 
                                        idx, len(eligible_electric_mrs), entity_id, not is_first)
                            
                            # First meter reading: clear all and regenerate from scratch
                            # Subsequent meter readings: merge with what we just calculated
                            await statistics.update_cost_statistics(
                                hass,
                                mock_entity,
                                statistics.CostDataExtractor(multiplier),
                                meter_reading,
                                merge_with_existing=not is_first,
                            )
                        
                        _LOGGER.info("✅ Recalculated electricity cost statistics for %s (%d meter readings processed)", 
                                    entity_id, len(eligible_electric_mrs))
                        recalculated_count += 1
                    except Exception as e:
                        _LOGGER.error("❌ Failed to recalculate electricity cost statistics for %s: %s", entity_id, e)

        if recalculated_count > 0:
            _LOGGER.info("✅ Successfully recalculated %d cost statistic(s)", recalculated_count)
        else:
            _LOGGER.warning("No cost statistics were recalculated")

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

        hass.services.async_register(
            DOMAIN,
            SERVICE_LOG_METER_READING_INTERVALS,
            log_meter_reading_intervals_service,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_LOG_STORED_XMLS,
            log_stored_xmls_service,
        )

        hass.services.async_register(
            DOMAIN,
            SERVICE_CLEAR_STORED_XML,
            clear_stored_xml_service,
            schema=CLEAR_STORED_XML_SCHEMA,
        )

        hass.services.async_register(
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
    """Unload services for the Green Button integration."""
    hass.services.async_remove(DOMAIN, SERVICE_IMPORT_ESPI_XML)
    hass.services.async_remove(DOMAIN, SERVICE_DELETE_STATISTICS)
    hass.services.async_remove(DOMAIN, SERVICE_LOG_METER_READING_INTERVALS)
    hass.services.async_remove(DOMAIN, SERVICE_LOG_STORED_XMLS)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_STORED_XML)
    hass.services.async_remove(DOMAIN, SERVICE_RECALCULATE_COST_STATISTICS)
    _LOGGER.info("Green Button services unloaded")
