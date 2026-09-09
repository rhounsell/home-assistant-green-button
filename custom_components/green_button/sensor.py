"""Sensor platform for the Green Button integration."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import logging
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import allocation, model, scaling, statistics
from .const import (
    CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DOMAIN,
)
from .coordinator import GreenButtonCoordinator
from .statistic_ids import statistic_id_from_unique_id

_LOGGER = logging.getLogger(__name__)


def _legacy_unique_id(entry_id: str, meter_reading_id: str, suffix: str) -> str:
    """Return the pre-2.0 identifier based on the final provider path segment."""
    return f"{entry_id}_{meter_reading_id.rsplit('/', 1)[-1]}{suffix}"


def _stream_unique_id(
    entry_id: str,
    usage_point_id: str,
    meter_reading_id: str,
    suffix: str,
) -> str:
    """Return a stable identifier for one provider usage-point stream."""
    identity = "\x00".join((usage_point_id, meter_reading_id, suffix))
    return f"{entry_id}_{sha256(identity.encode()).hexdigest()}{suffix}"


def _cost_multiplier(coordinator: GreenButtonCoordinator, gas: bool = False) -> int:
    """Return the configured fallback used only when XML omits a multiplier."""
    return scaling.configured_multiplier(
        coordinator.config_entry,
        (
            CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER
            if gas
            else CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER
        ),
        (
            DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER
            if gas
            else DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER
        ),
    )


def _has_interval_readings(meter_reading: model.MeterReading) -> bool:
    """Return whether a meter stream has usable interval data."""
    return bool(meter_reading.interval_blocks) and any(
        reading.value is not None
        for block in meter_reading.interval_blocks
        for reading in block.interval_readings
    )


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


def _schedule_hass_task_from_any_thread(hass: HomeAssistant, coro) -> None:
    """Schedule a coroutine on HA's event loop from any thread safely.

    If called on the event loop, schedule directly; otherwise, use call_soon_threadsafe.
    """
    loop = hass.loop
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        hass.async_create_task(coro)
    else:
        loop.call_soon_threadsafe(lambda: hass.async_create_task(coro))


class GreenButtonStatisticsSensor(
    CoordinatorEntity[GreenButtonCoordinator], SensorEntity
):
    """Display imported totals without automatic sensor statistics."""

    _attr_state_class = None

    @property
    def legacy_unique_id(self) -> str:
        """Return the former path-suffix ID for a possible registry migration."""
        return self._legacy_unique_id

    @property
    def long_term_statistics_id(self) -> str:
        """Return the external series associated with this display entity."""
        return statistic_id_from_unique_id(self._attr_unique_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the series to select in the Energy dashboard."""
        return {"statistic_id": self.long_term_statistics_id}


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
            _stream_unique_id(
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
            _stream_unique_id(
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


class GreenButtonGasSensor(GreenButtonStatisticsSensor):
    """Display imported gas consumption in m³."""

    _attr_device_class = SensorDeviceClass.GAS
    _attr_native_unit_of_measurement = "m³"
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
            coordinator.config_entry.entry_id, meter_reading_id, "_gas"
        )
        self._attr_unique_id = (
            _stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "_gas",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Usage"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas entities under a dedicated device."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")
            },
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float:
        """Return the cached total for display."""
        return self._cached_native_value

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "m³"

    async def async_added_to_hass(self) -> None:
        """Initialize the display sensor and schedule imported statistics."""
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Gas Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        found_meter_reading = True
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )
                        break
                if found_meter_reading:
                    break

            # If not found as meter reading, check if it's a UsagePoint ID (UsageSummary-only case)
            if not found_meter_reading:
                for usage_point in self.coordinator.data["usage_points"]:
                    if (
                        (
                            self._usage_point_id is None
                            or usage_point.id == self._usage_point_id
                        )
                        and usage_point.id == self._meter_reading_id
                        and usage_point.usage_summaries
                    ):
                        meter_readings = [
                            meter_reading
                            for meter_reading in usage_point.meter_readings
                            if _has_interval_readings(meter_reading)
                        ]
                        if len(meter_readings) == 1:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics(meter_readings[0]),
                            )
                        else:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics_from_summaries(
                                    usage_point
                                ),
                            )
                        break

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update cached values and schedule historical statistics."""
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )
        self._attr_native_value = statistics.gas_usage_total(
            meter_reading, summaries, usage_allocation_mode
        )
        # Run statistics update in background to not block startup
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_statistics_async(
                meter_reading, summaries, usage_allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        usage_allocation_mode: str,
    ) -> None:
        """Update gas statistics in background without blocking."""
        try:
            await statistics.update_gas_statistics(
                self.hass,
                self,
                meter_reading,
                usage_summaries=summaries,
                allocation_mode=usage_allocation_mode,
            )

            self._cached_native_value = statistics.gas_usage_total(
                meter_reading, summaries, usage_allocation_mode
            )

            # Write the state once after statistics import to update the sensor display
            self.async_write_ha_state()

            _LOGGER.info(
                "%s: Gas statistics update completed, state set to %.2f m³.",
                self.entity_id,
                self._cached_native_value,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update failed.",
                self.entity_id,
            )

    async def update_sensor_and_statistics_from_summaries(
        self, usage_point: model.UsagePoint
    ) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no daily MeterReadings)."""
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )
        self._attr_native_value = statistics.gas_usage_total(
            None, list(usage_point.usage_summaries), usage_allocation_mode
        )

        if usage_allocation_mode == "monthly_increment" and usage_point.usage_summaries:
            _LOGGER.info(
                "Gas Sensor %s: Generating statistics from UsageSummaries (no daily readings)",
                self.entity_id,
            )
            # Call update_gas_statistics in background - no meter reading available
            statistics.async_schedule_statistics_update(
                self.hass,
                self.coordinator.config_entry,
                lambda: self._update_gas_statistics_from_summaries_async(
                    usage_point, usage_allocation_mode
                ),
            )
            _LOGGER.debug(
                "%s: Gas statistics update (from summaries) scheduled in background.",
                self.entity_id,
            )
        else:
            _LOGGER.warning(
                "Gas Sensor %s: Cannot generate statistics - monthly_increment mode required for UsageSummary-only data",
                self.entity_id,
            )

    async def _update_gas_statistics_from_summaries_async(
        self, usage_point: model.UsagePoint, usage_allocation_mode: str
    ) -> None:
        """Update gas statistics from summaries in background without blocking."""
        try:
            await statistics.update_gas_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                usage_summaries=list(usage_point.usage_summaries),
                allocation_mode=usage_allocation_mode,
            )

            self._cached_native_value = statistics.gas_usage_total(
                None, list(usage_point.usage_summaries), usage_allocation_mode
            )

            # Write the state once after statistics import to update the sensor display
            self.async_write_ha_state()

            _LOGGER.info(
                "%s: Gas statistics update (from summaries) completed, state set to %.2f m³.",
                self.entity_id,
                self._cached_native_value,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update (from summaries) failed.",
                self.entity_id,
            )


class GreenButtonGasCostSensor(GreenButtonStatisticsSensor):
    """Gas cost sensor (monetary total) using UsageSummary pro-rated per day."""

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
        self._legacy_unique_id = _legacy_unique_id(
            coordinator.config_entry.entry_id, meter_reading_id, "_gas_cost"
        )
        self._attr_unique_id = (
            _stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "_gas_cost",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Cost"
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas cost under the gas device."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")
            },
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float | None:
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        if not summaries:
            return 0.0
        if any(summary.total_cost is None for summary in summaries):
            return None
        self._attr_native_unit_of_measurement = summaries[0].currency
        return float(
            sum(
                scaling.usage_summary_cost(us, _cost_multiplier(self.coordinator, True))
                for us in summaries
            )
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Describe the billing-cost allocation used by this display series."""
        attributes = super().extra_state_attributes
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        if allocation_mode == "pro_rate_daily":
            attributes["cost_allocation"] = "estimated_daily_proration"
        else:
            attributes["cost_allocation"] = "billing_period_increment"
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
            "Gas Cost Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        found_meter_reading = True
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )
                        break
                if found_meter_reading:
                    break

            # If not found as meter reading, check if it's a UsagePoint ID (UsageSummary-only case)
            if not found_meter_reading:
                for usage_point in self.coordinator.data["usage_points"]:
                    if (
                        (
                            self._usage_point_id is None
                            or usage_point.id == self._usage_point_id
                        )
                        and usage_point.id == self._meter_reading_id
                        and usage_point.usage_summaries
                    ):
                        meter_readings = [
                            meter_reading
                            for meter_reading in usage_point.meter_readings
                            if _has_interval_readings(meter_reading)
                        ]
                        if len(meter_readings) == 1:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics(meter_readings[0]),
                            )
                        else:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics_from_summaries(
                                    usage_point
                                ),
                            )
                        break

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        # Update state
        self._attr_native_value = self.native_value

        # Update long-term statistics with pro-rated daily cost
        # Run in background to not block startup
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_cost_statistics_async(
                meter_reading, summaries, allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        allocation_mode: str,
    ) -> None:
        """Update gas cost statistics in background without blocking."""
        try:
            gas_multiplier = _cost_multiplier(self.coordinator, True)
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                meter_reading,
                summaries,
                allocation_mode=allocation_mode,
                gas_cost_multiplier=gas_multiplier,
            )
            _LOGGER.info(
                "%s: Gas cost statistics update completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas cost statistics update failed.",
                self.entity_id,
            )

    async def update_sensor_and_statistics_from_summaries(
        self, usage_point: model.UsagePoint
    ) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no MeterReadings)."""
        # Update entity state (sum of all UsageSummary total_cost values)
        if any(us.total_cost is None for us in usage_point.usage_summaries):
            self._attr_native_value = None
            return
        total = sum(
            float(
                scaling.usage_summary_cost(us, _cost_multiplier(self.coordinator, True))
            )
            for us in usage_point.usage_summaries
        )
        self._attr_native_value = total if total > 0 else 0.0

        # Import gas cost statistics
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )

        # Force monthly_increment mode for UsageSummary-only data since pro_rate_daily requires daily readings
        if allocation_mode == "pro_rate_daily":
            _LOGGER.info(
                "Gas Cost Sensor %s: Forcing monthly_increment mode (UsageSummary-only data, no daily readings for pro-rating)",
                self.entity_id,
            )
            allocation_mode = "monthly_increment"

        _LOGGER.info(
            "Gas Cost Sensor %s: Generating statistics from UsageSummaries, mode=%s",
            self.entity_id,
            allocation_mode,
        )

        # Call update_gas_cost_statistics in background - no meter reading available
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_cost_statistics_from_summaries_async(
                usage_point, allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update (from summaries) scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_from_summaries_async(
        self, usage_point: model.UsagePoint, allocation_mode: str
    ) -> None:
        """Update gas cost statistics from summaries in background without blocking."""
        try:
            gas_multiplier = _cost_multiplier(self.coordinator, True)
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                list(usage_point.usage_summaries),
                allocation_mode=allocation_mode,
                gas_cost_multiplier=gas_multiplier,
            )
            _LOGGER.info(
                "%s: Gas cost statistics update (from summaries) completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas cost statistics update (from summaries) failed.",
                self.entity_id,
            )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up one stable entity pair for every unambiguous provider stream."""
    coordinator: GreenButtonCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    active_unique_ids: set[str] = set()

    def _eligible_meter_readings(
        usage_point: model.UsagePoint,
    ) -> list[model.MeterReading]:
        return sorted(
            (
                meter_reading
                for meter_reading in usage_point.meter_readings
                if _has_interval_readings(meter_reading)
            ),
            key=lambda meter_reading: meter_reading.id,
        )

    def _migrate_legacy_entities(
        entity_registry: Any, candidates: list[GreenButtonStatisticsSensor]
    ) -> None:
        """Move an unambiguous suffix-based entity and its external series."""
        by_legacy_id: dict[str, list[GreenButtonStatisticsSensor]] = {}
        for candidate in candidates:
            if candidate.unique_id != candidate.legacy_unique_id:
                by_legacy_id.setdefault(candidate.legacy_unique_id, []).append(
                    candidate
                )

        for legacy_unique_id, matching_candidates in by_legacy_id.items():
            if len(matching_candidates) != 1:
                _LOGGER.warning(
                    "Not migrating ambiguous legacy Green Button stream ID %s",
                    legacy_unique_id,
                )
                continue

            candidate = matching_candidates[0]
            if entity_registry.async_get_entity_id(
                "sensor", DOMAIN, candidate.unique_id
            ):
                continue
            if not (
                legacy_entity_id := entity_registry.async_get_entity_id(
                    "sensor", DOMAIN, legacy_unique_id
                )
            ):
                continue

            entity_registry.async_update_entity(
                legacy_entity_id, new_unique_id=candidate.unique_id
            )
            statistics.rename_external_statistic(
                hass,
                candidate.legacy_unique_id,
                candidate.unique_id,
            )
            _LOGGER.info(
                "Migrated Green Button stream %s to full provider identity",
                legacy_entity_id,
            )

    def _async_create_entities() -> None:
        """Create entities for all streams currently available from the provider."""
        if not coordinator.data or not coordinator.data.get("usage_points"):
            return

        candidates: list[GreenButtonStatisticsSensor] = []
        for usage_point in sorted(coordinator.usage_points, key=lambda point: point.id):
            meter_readings = _eligible_meter_readings(usage_point)
            if usage_point.sensor_device_class == SensorDeviceClass.GAS:
                allocation_mode = (
                    entry.options.get("gas_usage_allocation")
                    or entry.data.get("gas_usage_allocation")
                    or "daily_readings"
                )
                if (
                    allocation_mode == "monthly_increment"
                    and usage_point.usage_summaries
                ):
                    summary_stream_id = (
                        meter_readings[0].id
                        if len(meter_readings) == 1
                        else usage_point.id
                    )
                    candidates.extend(
                        (
                            GreenButtonGasSensor(
                                coordinator, summary_stream_id, usage_point.id
                            ),
                            GreenButtonGasCostSensor(
                                coordinator, summary_stream_id, usage_point.id
                            ),
                        )
                    )
                    continue

                for meter_reading in meter_readings:
                    candidates.append(
                        GreenButtonGasSensor(
                            coordinator, meter_reading.id, usage_point.id
                        )
                    )

                if len(meter_readings) == 1:
                    candidates.append(
                        GreenButtonGasCostSensor(
                            coordinator, meter_readings[0].id, usage_point.id
                        )
                    )
                elif meter_readings and usage_point.usage_summaries:
                    _LOGGER.warning(
                        "Skipping gas cost entities for UsagePoint %s: its billing "
                        "summary cannot be attributed to one of %d meter streams",
                        usage_point.id,
                        len(meter_readings),
                    )
                continue

            for meter_reading in meter_readings:
                candidates.extend(
                    (
                        GreenButtonSensor(
                            coordinator, meter_reading.id, usage_point.id
                        ),
                        GreenButtonCostSensor(
                            coordinator, meter_reading.id, usage_point.id
                        ),
                    )
                )

        if not candidates:
            return

        entity_registry = async_get_entity_registry(hass)
        _migrate_legacy_entities(entity_registry, candidates)
        new_entities = [
            candidate
            for candidate in candidates
            if candidate.unique_id not in active_unique_ids
        ]
        if not new_entities:
            return

        async_add_entities(new_entities)
        active_unique_ids.update(entity.unique_id for entity in new_entities)

    _async_create_entities()
    entry.async_on_unload(coordinator.async_add_listener(_async_create_entities))
