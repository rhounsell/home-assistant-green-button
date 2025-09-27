"""Green Button data coordinator."""

from __future__ import annotations

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

    async def async_add_data(self, xml_data: str) -> None:
        """Manually add new Green Button XML data and update entities."""
        try:
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

            # Replace existing data (this handles duplicates by overwriting)
            self.usage_points = new_usage_points

            # Update the data and notify all entities
            self.async_set_updated_data({"usage_points": self.usage_points})

            _LOGGER.info("Successfully updated coordinator with new data")

        except Exception as err:
            _LOGGER.error("Error adding Green Button data: %s", err)
            raise UpdateFailed(f"Error adding Green Button data: {err}") from err

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
