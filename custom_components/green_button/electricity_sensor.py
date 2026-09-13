"""Electricity sensor entities for the Green Button integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.helpers.device_registry import DeviceInfo

from . import allocation, model, scaling, statistics
from ._sensor_common import (
    GreenButtonStatisticsSensor,
    _cost_multiplier,
    _legacy_unique_id,
    _schedule_hass_task_from_any_thread,
)
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator
from .statistic_ids import stream_unique_id

_LOGGER = logging.getLogger(__name__)


def _electricity_usage_total(meter_reading: model.MeterReading) -> float:
    """Return the same complete-hour energy total published to statistics."""
    values = allocation.hourly_values(
        allocation.interval_readings(meter_reading),
        lambda reading: allocation.energy_to_kwh(
            scaling.interval_value(reading), reading.reading_type.unit_of_measurement
        ),
    )
    return float(sum(values.values(), allocation.ZERO))


def _electricity_cost_total(
    meter_reading: model.MeterReading, fallback_multiplier: int
) -> float | None:
    """Return the same complete-hour cost total published to statistics."""
    readings = allocation.interval_readings(meter_reading)
    if any(reading.cost is None for reading in readings):
        return None
    values = allocation.hourly_values(
        readings,
        lambda reading: scaling.interval_cost(reading, fallback_multiplier),
    )
    return float(sum(values.values(), allocation.ZERO))


class GreenButtonSensor(GreenButtonStatisticsSensor):
    """A sensor for Green Button energy data."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = "kWh"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
        usage_point_id: str | None = None,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._usage_point_id = usage_point_id
        self._cached_native_value: float = 0.0  # Cache last imported statistics value

        self._legacy_unique_id = _legacy_unique_id(
            coordinator.config_entry.entry_id, meter_reading_id, ""
        )
        self._attr_unique_id = (
            stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Usage"

    @property
    def device_info(self) -> DeviceInfo:
        """Group electricity sensors under a dedicated device in the integration UI."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_electricity_device")
            },
            name=f"{self.coordinator.config_entry.title} Electricity",
            manufacturer="Green Button",
            model="Electricity",
        )

    @property
    def native_value(self) -> float:
        """Return the cached total for display."""
        return self._cached_native_value

    @property
    def available(self) -> bool:
        available = self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )
        _LOGGER.debug(
            "Sensor %s: available property evaluated to %s (last_update_success=%s, data is not None=%s)",
            getattr(self, "entity_id", self._attr_unique_id),
            available,
            self.coordinator.last_update_success,
            self.coordinator.data is not None,
        )
        return available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        meter_reading = self.coordinator.get_meter_reading_by_id(
            self._meter_reading_id, self._usage_point_id
        )
        if not meter_reading:
            return super().extra_state_attributes

        attributes = {
            **super().extra_state_attributes,
            "meter_reading_id": meter_reading.id,
            "interval_blocks_count": len(meter_reading.interval_blocks),
        }

        # Add latest interval information
        if meter_reading.interval_blocks:
            latest_block = meter_reading.interval_blocks[-1]
            attributes.update(
                {
                    "latest_block_start": latest_block.start.isoformat(),
                    "latest_block_duration": str(latest_block.duration),
                    "latest_block_readings_count": len(latest_block.interval_readings),
                }
            )

            if latest_block.interval_readings:
                latest_reading = latest_block.interval_readings[-1]
                attributes.update(
                    {
                        "latest_reading_start": latest_reading.start.isoformat(),
                        "latest_reading_duration": str(latest_reading.duration),
                        "latest_reading_value": latest_reading.value,
                    }
                )

        return attributes

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the native unit of measurement for statistics protocol."""
        return self._attr_native_unit_of_measurement or "kWh"

    async def async_added_to_hass(self) -> None:
        """Initialize the display sensor and schedule imported statistics."""
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        meter_reading = self.coordinator.get_meter_reading_by_id(
            self._meter_reading_id, self._usage_point_id
        )
        if meter_reading:
            self._attr_native_value = _electricity_usage_total(meter_reading)
            _LOGGER.info(
                "Sensor %s: Internal state set to %.2f kWh (cached for display)",
                self.entity_id,
                self._attr_native_value,
            )
        else:
            # If no data, set to 0.0 for internal reference
            if self._attr_native_value is None:
                self._attr_native_value = 0.0
            _LOGGER.info(
                "Sensor %s: Internal state set to 0.0 (no meter reading found, NOT written to HA)",
                self.entity_id,
            )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""

        # Update statistics for all meter readings in coordinator data
        if self.coordinator.data and "usage_points" in self.coordinator.data:
            usage_points = self.coordinator.data["usage_points"]
            _LOGGER.info(
                "Sensor %s: Found %d usage points for statistics update",
                self.entity_id,
                len(usage_points),
            )
            for usage_point in usage_points:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        # Schedule statistics update (statistics system is idempotent)
                        _LOGGER.info(
                            "Sensor %s: Scheduling statistics update for meter reading %s",
                            self.entity_id,
                            meter_reading.id,
                        )
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )
        else:
            _LOGGER.info(
                "Sensor %s: No coordinator data available for statistics update",
                self.entity_id,
            )

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update cached values and schedule historical statistics."""
        self._attr_native_value = _electricity_usage_total(meter_reading)

        _LOGGER.debug(
            "🔍 %s: Setting sensor state to %.2f kWh (cumulative).",
            self.entity_id,
            self._attr_native_value,
        )

        # Update statistics for Energy Dashboard (run in background to not block startup)
        if hasattr(self, "hass") and self.hass is not None:
            statistics.async_schedule_statistics_update(
                self.hass,
                self.coordinator.config_entry,
                lambda: self._update_statistics_async(meter_reading),
            )
            _LOGGER.debug(
                "%s: Statistics update scheduled in background.",
                self.entity_id,
            )

    async def _update_statistics_async(self, meter_reading: model.MeterReading) -> None:
        """Update statistics in background without blocking."""
        try:
            await statistics.update_statistics(
                self.hass,
                self,
                statistics.DefaultDataExtractor(),
                meter_reading,
            )

            # Cache the last statistics sum value for display as sensor state
            # This prevents Energy Dashboard "unavailable" warnings
            self._cached_native_value = _electricity_usage_total(meter_reading)

            # Write the state once after statistics import to update the sensor display
            self.async_write_ha_state()

            _LOGGER.info(
                "%s: Statistics update completed, state set to %.2f kWh.",
                self.entity_id,
                self._cached_native_value,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Statistics update failed.",
                self.entity_id,
            )


class GreenButtonCostSensor(GreenButtonStatisticsSensor):
    """A sensor for Green Button monetary cost data (total)."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
        usage_point_id: str | None = None,
    ) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._usage_point_id = usage_point_id
        self._cached_native_value: float = 0.0  # Initialize to 0 for Energy Dashboard

        self._legacy_unique_id = _legacy_unique_id(
            coordinator.config_entry.entry_id, meter_reading_id, "_cost"
        )
        self._attr_unique_id = (
            stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "_cost",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Cost"

        # Default currency; will be set on first update if available from reading type
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def device_info(self) -> DeviceInfo:
        """Group electricity cost sensors under the electricity device."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_electricity_device")
            },
            name=f"{self.coordinator.config_entry.title} Electricity",
            manufacturer="Green Button",
            model="Electricity",
        )

    @property
    def native_value(self) -> float | None:
        """Return the current total cost value."""
        if not self.coordinator.data or not self.coordinator.data.get("usage_points"):
            return self._cached_native_value  # Return cached value instead of None

        meter_reading = self.coordinator.get_meter_reading_by_id(
            self._meter_reading_id, self._usage_point_id
        )
        if not meter_reading:
            return self._cached_native_value  # Return cached value instead of None

        # Set currency if available
        currency = getattr(meter_reading.reading_type, "currency", None)
        if currency:
            self._attr_native_unit_of_measurement = currency

        total_cost = _electricity_cost_total(
            meter_reading, _cost_multiplier(self.coordinator)
        )
        if total_cost is None:
            return None
        self._cached_native_value = total_cost
        return self._cached_native_value

    @property
    def available(self) -> bool:
        available = self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )
        _LOGGER.debug(
            "Cost Sensor %s: available property evaluated to %s (last_update_success=%s, data is not None=%s)",
            getattr(self, "entity_id", self._attr_unique_id),
            available,
            self.coordinator.last_update_success,
            self.coordinator.data is not None,
        )
        return available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        meter_reading = self.coordinator.get_meter_reading_by_id(
            self._meter_reading_id, self._usage_point_id
        )
        if not meter_reading:
            return super().extra_state_attributes

        attributes = {
            **super().extra_state_attributes,
            "meter_reading_id": meter_reading.id,
            "interval_blocks_count": len(meter_reading.interval_blocks),
        }

        if meter_reading.interval_blocks:
            latest_block = meter_reading.interval_blocks[-1]
            attributes.update(
                {
                    "latest_block_start": latest_block.start.isoformat(),
                    "latest_block_duration": str(latest_block.duration),
                    "latest_block_readings_count": len(latest_block.interval_readings),
                }
            )

        return attributes

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        """Initialize the display sensor and schedule imported statistics."""
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Cost Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            usage_points = self.coordinator.data["usage_points"]
            for usage_point in usage_points:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update cached values and schedule historical statistics."""
        # Update state
        total_cost = _electricity_cost_total(
            meter_reading, _cost_multiplier(self.coordinator)
        )
        if total_cost is None:
            self._attr_native_value = None
            return

        # Set currency
        currency = getattr(meter_reading.reading_type, "currency", None)
        if currency:
            self._attr_native_unit_of_measurement = currency

        self._attr_native_value = total_cost

        # Update long-term statistics (run in background to not block startup)
        if hasattr(self, "hass") and self.hass is not None:
            statistics.async_schedule_statistics_update(
                self.hass,
                self.coordinator.config_entry,
                lambda: self._update_cost_statistics_async(meter_reading),
            )
            _LOGGER.debug(
                "%s: Cost statistics update scheduled in background.",
                self.entity_id,
            )

    async def _update_cost_statistics_async(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update cost statistics in background without blocking."""
        try:
            multiplier = _cost_multiplier(self.coordinator)
            await statistics.update_cost_statistics(
                self.hass,
                self,
                statistics.CostDataExtractor(multiplier),
                meter_reading,
            )
            _LOGGER.info(
                "%s: Cost statistics update completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Cost statistics update failed.",
                self.entity_id,
            )
