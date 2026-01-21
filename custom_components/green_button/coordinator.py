"""Green Button data coordinator."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from . import model
from .const import DOMAIN
from .parsers import espi

_LOGGER = logging.getLogger(__name__)


class GreenButtonCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage Green Button data updates (manual updates only, no polling)."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the Green Button coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            config_entry=config_entry,
            # No update_interval - manual updates only
        )
        self.config_entry = config_entry
        self.usage_points: list[model.UsagePoint] = []

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch and parse the latest Green Button data."""
        try:
            usage_points = None
            # Get XML data from config entry instead of file path
            xml_data = self.config_entry.data.get("xml")
            if xml_data:
                usage_points = await self.hass.async_add_executor_job(
                    espi.parse_xml, xml_data
                )
        except Exception as err:
            raise UpdateFailed(f"Error updating Green Button data: {err}") from err
        else:
            self.usage_points = usage_points or []
            return {"usage_points": usage_points or []}

    async def async_add_xml_data(self, xml_data: str, store_in_config: bool = True) -> None:
        """Add new Green Button XML data and update entities.
        
        Args:
            xml_data: The XML data to parse and add
            store_in_config: If True, store the XML in config entry (for initial setup).
                           If False, just merge the data without persisting (for service imports).
        """
        try:
            # Only store XML in config entry if requested (e.g., during initial setup)
            # For service imports, we don't store to avoid triggering a reload
            if store_in_config:
                data_updates = dict(self.config_entry.data)
                data_updates["xml"] = xml_data
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data=data_updates
                )

            # Parse and update immediately
            usage_points = await self.hass.async_add_executor_job(
                espi.parse_xml, xml_data
            )
            new_usage_points = usage_points or []

            # Log what we're processing
            total_readings = sum(len(up.meter_readings) for up in new_usage_points)
            _LOGGER.info(
                "Processing %d usage points with %d total meter readings",
                len(new_usage_points),
                total_readings,
            )
            # Log interval block date ranges for each new usage point and meter reading
            if _LOGGER.isEnabledFor(logging.DEBUG):
                for up in new_usage_points:
                    for mr in up.meter_readings:
                        for ib in mr.interval_blocks:
                            if ib.interval_readings:
                                start = ib.interval_readings[0].start
                                end = ib.interval_readings[-1].end
                                _LOGGER.debug(
                                    "[IMPORT] UsagePoint %s MeterReading %s IntervalBlock: %s - %s (%d readings)",
                                    up.id,
                                    mr.id,
                                    start,
                                    end,
                                    len(ib.interval_readings),
                                )
                            else:
                                _LOGGER.debug(
                                    "[IMPORT] UsagePoint %s MeterReading %s IntervalBlock: No readings",
                                    up.id,
                                    mr.id,
                                )

            # Merge new data with existing data (combine multiple imports)
            self._merge_usage_points(new_usage_points)

            # Debug: Log detailed data structure
            for i, up in enumerate(self.usage_points):
                _LOGGER.info(
                    "UsagePoint %d: %d meter readings", i, len(up.meter_readings)
                )
                for j, mr in enumerate(up.meter_readings):
                    _LOGGER.info(
                        "  MeterReading %d: %d intervals", j, len(mr.interval_blocks)
                    )
                    for ib in mr.interval_blocks:
                        if ib.interval_readings:
                            start = ib.interval_readings[0].start
                            end = ib.interval_readings[-1].end
                            _LOGGER.debug(
                                "  IntervalBlock: %s - %s (%d readings)",
                                start,
                                end,
                                len(ib.interval_readings),
                            )
                        else:
                            _LOGGER.debug(
                                "  IntervalBlock: No readings",
                            )

            # Update the data and notify all entities
            self.async_set_updated_data({"usage_points": self.usage_points})

            _LOGGER.info("Successfully updated coordinator with new data")
            
            # Trigger statistics generation for all meter readings after import
            await self._trigger_statistics_update_for_all_readings()

        except Exception as err:
            _LOGGER.error("Error adding Green Button XML data: %s", err)
            raise UpdateFailed(f"Error adding Green Button XML data: {err}") from err

    async def _trigger_statistics_update_for_all_readings(self) -> None:
        """Trigger statistics update for all meter readings in coordinator data.
        
        This ensures that after import, statistics are generated for every meter reading,
        including newly merged ones from imports. The coordinator update listeners
        (entity sensors) will be notified and will generate statistics automatically.
        """
        _LOGGER.info("Starting statistics update for all meter readings")
        
        if not self.data or not self.data.get("usage_points"):
            _LOGGER.info("No coordinator data available for statistics update")
            return
        
        if _LOGGER.isEnabledFor(logging.DEBUG):
            # Log all meter readings that need statistics generated
            total_meter_readings = 0
            for usage_point in self.usage_points:
                for meter_reading in usage_point.meter_readings:
                    total_meter_readings += 1
                    interval_count = sum(len(blk.interval_readings) for blk in meter_reading.interval_blocks)
                    if interval_count > 0:
                        _LOGGER.debug(
                            "Will generate statistics for meter reading %s: %d total readings across %d interval blocks",
                            meter_reading.id.split("/")[-1] if "/" in meter_reading.id else meter_reading.id,
                            interval_count,
                            len(meter_reading.interval_blocks),
                        )
                        for ib in meter_reading.interval_blocks:
                            if ib.interval_readings:
                                first = ib.interval_readings[0].start
                                last = ib.interval_readings[-1].end
                                _LOGGER.debug(
                                    "  IntervalBlock: %s - %s (%d readings)",
                                    first.isoformat(),
                                    last.isoformat(),
                                    len(ib.interval_readings),
                                )
                            else:
                                _LOGGER.debug("  IntervalBlock: No readings")
        
            _LOGGER.info("Statistics update scheduled for %d meter readings", total_meter_readings)

    def has_existing_entities(self) -> bool:
        """Check if entities already exist for the current data."""
        return bool(self.usage_points)

    async def async_load_stored_data(self) -> None:
        """Load XML data from config entry (used during startup)."""
        xml_data = self.config_entry.data.get("xml")
        if not xml_data:
            _LOGGER.debug("No stored XML data found in config entry")
            return

        if self.has_existing_entities():
            _LOGGER.debug("Entities already exist, skipping XML re-parsing on restart")
            return

        try:
            _LOGGER.debug("[RESTART] Loading stored XML data from config entry")
            usage_points = await self.hass.async_add_executor_job(
                espi.parse_xml, xml_data
            )
            self.usage_points = usage_points or []
            self.async_set_updated_data({"usage_points": self.usage_points})
            self.last_update_success = True
            _LOGGER.debug(
                "[RESTART] Successfully loaded %d usage points from stored data. last_update_success set to True.",
                len(self.usage_points),
            )
        except (ValueError, OSError) as err:
            self.last_update_success = False
            _LOGGER.warning("[RESTART] Failed to load stored XML data: %s. last_update_success set to False.", err)

    def _merge_usage_points(self, new_usage_points: list[model.UsagePoint]) -> None:
        """Merge new usage points with existing ones, combining interval blocks."""
        if not self.usage_points:
            # No existing data, just use new data
            self.usage_points = new_usage_points
            _LOGGER.info("[MERGE] No existing usage points, using new data only.")
            for up in new_usage_points:
                for mr in up.meter_readings:
                    for ib in mr.interval_blocks:
                        if ib.interval_readings:
                            start = ib.interval_readings[0].start
                            end = ib.interval_readings[-1].end
                            _LOGGER.debug(
                                "[MERGE] UsagePoint %s MeterReading %s IntervalBlock: %s - %s (%d readings)",
                                up.id,
                                mr.id,
                                start,
                                end,
                                len(ib.interval_readings),
                            )
                        else:
                            _LOGGER.debug(
                                "[MERGE] UsagePoint %s MeterReading %s IntervalBlock: No readings",
                                up.id,
                                mr.id,
                            )
            return

        # Create a mapping of existing usage points by ID
        existing_up_map = {up.id: up for up in self.usage_points}

        for new_up in new_usage_points:
            if new_up.id in existing_up_map:
                # Merge meter readings for existing usage point
                existing_up = existing_up_map[new_up.id]
                self._merge_meter_readings(existing_up, list(new_up.meter_readings))
                # Merge usage summaries (unique by id)
                existing_summaries = {us.id: us for us in existing_up.usage_summaries}
                merged_summaries = list(existing_up.usage_summaries)
                added = 0
                for us in new_up.usage_summaries:
                    if us.id not in existing_summaries:
                        merged_summaries.append(us)
                        added += 1
                if added:
                    merged_up = dataclasses.replace(existing_up, usage_summaries=merged_summaries)
                    self.usage_points = [merged_up if up.id == existing_up.id else up for up in self.usage_points]
                _LOGGER.info(
                    "[MERGE] Merged usage point %s: %d meter readings, %d usage summaries",
                    new_up.id,
                    len(existing_up.meter_readings),
                    len(merged_summaries),
                )
            else:
                # Add new usage point
                self.usage_points.append(new_up)
                _LOGGER.info(
                    "[MERGE] Added new usage point %s: %d meter readings, %d usage summaries",
                    new_up.id,
                    len(new_up.meter_readings),
                    len(new_up.usage_summaries),
                )

    def _merge_meter_readings(
        self,
        existing_up: model.UsagePoint,
        new_meter_readings: list[model.MeterReading],
    ) -> None:
        """Merge new meter readings with existing ones in a usage point."""
        # Since objects are immutable, we need to rebuild everything
        existing_mr_map = {mr.id: mr for mr in existing_up.meter_readings}
        merged_meter_readings = []

        # Process existing meter readings
        for existing_mr in existing_up.meter_readings:
            # Check if this meter reading has new data to merge
            matching_new_mr = None
            for new_mr in new_meter_readings:
                if new_mr.id == existing_mr.id:
                    matching_new_mr = new_mr
                    break

            if matching_new_mr:
                # Merge interval blocks, avoiding duplicates by time period
                existing_blocks = {
                    (ib.start, ib.duration): ib for ib in existing_mr.interval_blocks
                }
                merged_blocks = list(existing_mr.interval_blocks)
                new_blocks_added = 0

                for new_block in matching_new_mr.interval_blocks:
                    block_key = (new_block.start, new_block.duration)
                    if block_key not in existing_blocks:
                        merged_blocks.append(new_block)
                        new_blocks_added += 1
                        # Log interval block date range for merged block
                        if new_block.interval_readings:
                            start = new_block.interval_readings[0].start
                            end = new_block.interval_readings[-1].end
                            _LOGGER.debug(
                                "[MERGE] MeterReading %s: Merged IntervalBlock %s - %s (%d readings)",
                                existing_mr.id,
                                start,
                                end,
                                len(new_block.interval_readings),
                            )
                        else:
                            _LOGGER.debug(
                                "[MERGE] MeterReading %s: Merged IntervalBlock with no readings",
                                existing_mr.id,
                            )

                if new_blocks_added > 0:
                    # Sort blocks by start time to maintain chronological order
                    merged_blocks.sort(key=lambda block: block.start)
                    # Create new meter reading with merged blocks
                    merged_mr = dataclasses.replace(
                        existing_mr, interval_blocks=merged_blocks
                    )
                    merged_meter_readings.append(merged_mr)
                    _LOGGER.info(
                        "Merged %d new interval blocks into meter reading %s",
                        new_blocks_added,
                        existing_mr.id,
                    )
                else:
                    # No new blocks, keep existing
                    merged_meter_readings.append(existing_mr)
            else:
                # No matching new data, keep existing
                merged_meter_readings.append(existing_mr)

        # Add completely new meter readings (not in existing)
        for new_mr in new_meter_readings:
            if new_mr.id not in existing_mr_map:
                merged_meter_readings.append(new_mr)
                _LOGGER.info(
                    "[MERGE] Added new meter reading: %s to usage point %s",
                    new_mr.id,
                    existing_up.id,
                )
                for ib in new_mr.interval_blocks:
                    if ib.interval_readings:
                        start = ib.interval_readings[0].start
                        end = ib.interval_readings[-1].end
                        _LOGGER.debug(
                            "[MERGE] MeterReading %s: Added IntervalBlock %s - %s (%d readings)",
                            new_mr.id,
                            start,
                            end,
                            len(ib.interval_readings),
                        )
                    else:
                        _LOGGER.debug(
                            "[MERGE] MeterReading %s: Added IntervalBlock with no readings",
                            new_mr.id,
                        )

        # Replace usage point with merged meter readings
        merged_up = dataclasses.replace(
            existing_up, meter_readings=merged_meter_readings
        )
        self.usage_points = [
            merged_up if up.id == existing_up.id else up for up in self.usage_points
        ]

    def get_meter_readings(self) -> list[model.MeterReading]:
        """Get all meter readings from usage points."""
        meter_readings = []
        for usage_point in self.usage_points:
            meter_readings.extend(usage_point.meter_readings)
        return meter_readings

    def get_usage_summaries_for_meter_reading(self, meter_reading_id: str) -> list[model.UsageSummary]:
        """Get usage summaries for the usage point that owns the meter reading."""
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                if meter_reading.id == meter_reading_id:
                    return list(getattr(usage_point, "usage_summaries", []) or [])
        return []

    def get_meter_reading_by_id(
        self, meter_reading_id: str
    ) -> model.MeterReading | None:
        """Get a specific meter reading by ID."""
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                if meter_reading.id == meter_reading_id:
                    return meter_reading
        return None

    def get_latest_cumulative_energy_kwh(self) -> float | None:
        """Return the latest cumulative energy usage in kWh from usage_points."""
        if not self.usage_points:
            return None

        latest_value = None
        latest_time = None

        # Navigate through the hierarchy: UsagePoint -> MeterReading -> IntervalBlock -> IntervalReading
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                for interval_block in meter_reading.interval_blocks:
                    for interval_reading in interval_block.interval_readings:
                        # Check if this reading is the latest
                        if latest_time is None or interval_reading.end > latest_time:
                            latest_time = interval_reading.end
                            # Convert value based on power of ten multiplier and unit
                            power_multiplier = (
                                interval_reading.reading_type.power_of_ten_multiplier
                            )
                            value = interval_reading.value * (10**power_multiplier)
                            # Convert to kWh if needed (assuming base unit is Wh)
                            latest_value = float(value) / 1000.0

        return latest_value
