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

    async def async_add_xml_data(self, xml_data: str) -> None:
        """Add new Green Button XML data and update entities."""
        try:
            # Store the XML data in config entry for future refreshes
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

            # Update the data and notify all entities
            self.async_set_updated_data({"usage_points": self.usage_points})

            _LOGGER.info("Successfully updated coordinator with new data")

        except Exception as err:
            _LOGGER.error("Error adding Green Button XML data: %s", err)
            raise UpdateFailed(f"Error adding Green Button XML data: {err}") from err

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
            _LOGGER.info("Loading stored XML data from config entry (restart)")
            # Parse stored XML data
            usage_points = await self.hass.async_add_executor_job(
                espi.parse_xml, xml_data
            )
            self.usage_points = usage_points or []

            # Update the data and notify all entities
            self.async_set_updated_data({"usage_points": self.usage_points})

            _LOGGER.info(
                "Successfully loaded %d usage points from stored data",
                len(self.usage_points),
            )

        except (ValueError, OSError) as err:
            _LOGGER.warning("Failed to load stored XML data: %s", err)

    def _merge_usage_points(self, new_usage_points: list[model.UsagePoint]) -> None:
        """Merge new usage points with existing ones, combining interval blocks."""
        if not self.usage_points:
            # No existing data, just use new data
            self.usage_points = new_usage_points
            return

        # Create a mapping of existing usage points by ID
        existing_up_map = {up.id: up for up in self.usage_points}

        for new_up in new_usage_points:
            if new_up.id in existing_up_map:
                # Merge meter readings for existing usage point
                existing_up = existing_up_map[new_up.id]
                self._merge_meter_readings(existing_up, new_up.meter_readings)
            else:
                # Add new usage point
                self.usage_points.append(new_up)

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
                    "Added new meter reading: %s to usage point %s",
                    new_mr.id,
                    existing_up.id,
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
