"""Sensor platform for the Green Button integration."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import model
from . import statistics
from .coordinator import GreenButtonCoordinator
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


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


class GreenButtonSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """A sensor for Green Button energy data."""

    _attr_device_class = SensorDeviceClass.ENERGY
    # Set state_class to satisfy Energy Dashboard requirements
    # BUT we disable recorder statistics for this entity (see _attr_should_poll)
    # and manually manage statistics via async_import_statistics
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id

        # Create a cleaner unique ID from the meter reading ID
        # Extract the last part after the final slash for a shorter identifier
        clean_id = (
            meter_reading_id.split("/")[-1]
            if "/" in meter_reading_id
            else meter_reading_id
        )
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}"
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Usage"

    @property
    def device_info(self) -> DeviceInfo:
        """Group electricity sensors under a dedicated device in the integration UI."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self.coordinator.config_entry.entry_id}_electricity_device")},
            name=f"{self.coordinator.config_entry.title} Electricity",
            manufacturer="Green Button",
            model="Electricity",
        )
    @property
    def native_value(self):
        """Return the native value of the sensor."""
        value = self._attr_native_value
        # Only compare to a numeric threshold for numeric types to avoid type errors
        if isinstance(value, (int, float)) and value > 10000:  # Log if suspiciously high
            # Get the call stack to see WHO is accessing this value
            stack = traceback.extract_stack()
            caller_info = []
            for frame in stack[-5:-1]:  # Last 4 frames before this one
                if 'green_button' not in frame.filename:  # Only non-Green Button frames
                    caller_info.append(f"{frame.filename}:{frame.lineno} in {frame.name}")
        return value

    @property
    def available(self) -> bool:
        available = self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )
        _LOGGER.info(
            "Sensor %s: available property evaluated to %s (last_update_success=%s, data is not None=%s)",
            getattr(self, 'entity_id', self._attr_unique_id),
            available,
            self.coordinator.last_update_success,
            self.coordinator.data is not None,
        )
        return available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if not meter_reading:
            return {}

        attributes = {
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
    def long_term_statistics_id(self) -> str:
        """Return the statistic ID associated with the entity."""
        return self.entity_id

    @property
    def name(self) -> str:
        """Return the entity name for statistics protocol."""
        # Return the config entry title (name set when adding integration)
        return self._attr_name or self.coordinator.config_entry.title

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the native unit of measurement for statistics protocol."""
        return self._attr_native_unit_of_measurement or "kWh"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass, re-calculate state from merged data and trigger statistics generation."""
        await super().async_added_to_hass()

        _LOGGER.info(
            "Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Always re-calculate state from merged data after restart
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if meter_reading:
            # Calculate total energy from all interval blocks
            total_energy = 0
            for interval_block in meter_reading.interval_blocks:
                for interval_reading in interval_block.interval_readings:
                    if interval_reading.value is not None:
                        power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
                        value = interval_reading.value * (10**power_multiplier)
                        total_energy += value
            if total_energy > 0:
                self._attr_native_value = total_energy / 1000.0
            else:
                self._attr_native_value = 0.0
            self.async_write_ha_state()
            _LOGGER.info(
                "Sensor %s: State re-calculated from merged data: %.2f kWh",
                self.entity_id,
                self._attr_native_value,
            )
        else:
            # If no data, initialize to 0.0 to make entity available
            if self._attr_native_value is None:
                self._attr_native_value = 0.0
                self.async_write_ha_state()
                _LOGGER.info(
                    "Sensor %s: Initialized with value 0.0 to make entity available (no meter reading found)",
                    self.entity_id,
                )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        # DO NOT call super()._handle_coordinator_update()!
        # That would call async_write_ha_state(), which triggers HA's Recorder
        # to auto-generate statistics using the cumulative sensor value,
        # creating corrupted records (state/sum swap, negative sums).
        # We manually manage statistics in update_sensor_and_statistics(),
        # so we only need to mark the entity as available for updates.

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
                    if meter_reading.id == self._meter_reading_id:
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
        """Update the entity's state and statistics to match the reading."""
        # Calculate total energy from all interval blocks
        total_energy = 0
        for interval_block in meter_reading.interval_blocks:
            for interval_reading in interval_block.interval_readings:
                if interval_reading.value is not None:
                    # Apply power of ten multiplier
                    power_multiplier = (
                        interval_reading.reading_type.power_of_ten_multiplier
                    )
                    value = interval_reading.value * (10**power_multiplier)
                    total_energy += value

        # Convert from Wh to kWh if needed
        if total_energy > 0:
            self._attr_native_value = total_energy / 1000.0
        else:
            self._attr_native_value = 0

        _LOGGER.info(
            "🔍 [DIAGNOSTIC] %s: Setting sensor state to %.2f kWh (cumulative).",
            self.entity_id,
            self._attr_native_value,
        )

        # NOTE: Do NOT call async_write_ha_state() here!
        # Calling it before statistics update causes HA's Recorder to automatically
        # create a statistics record using the sensor's cumulative state value,
        # which creates corrupted records (state/sum swap, massive consumption values).
        # The coordinator will handle state updates, and we manually manage statistics below.

        # Update statistics for Energy Dashboard
        if hasattr(self, "hass") and self.hass is not None:
            await statistics.update_statistics(
                self.hass,
                self,
                statistics.DefaultDataExtractor(),
                meter_reading,
            )
            _LOGGER.info(
                "%s: Statistics update completed.",
                self.entity_id,
            )


class GreenButtonCostSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """A sensor for Green Button monetary cost data (total)."""

    _attr_device_class = SensorDeviceClass.MONETARY
    # Set state_class to TOTAL (not TOTAL_INCREASING) for monetary sensors
    # Monetary values can decrease (refunds, adjustments), so use TOTAL
    # We disable recorder statistics for this entity and manually manage via async_import_statistics
    _attr_state_class = SensorStateClass.TOTAL
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
    ) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._cached_native_value: float = 0.0  # Initialize to 0 for Energy Dashboard

        # Build unique id with cost suffix
        clean_id = (
            meter_reading_id.split("/")[-1]
            if "/" in meter_reading_id
            else meter_reading_id
        )
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_cost"
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Cost"

        # Default currency; will be set on first update if available from reading type
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def device_info(self) -> DeviceInfo:
        """Group electricity cost sensors under the electricity device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self.coordinator.config_entry.entry_id}_electricity_device")},
            name=f"{self.coordinator.config_entry.title} Electricity",
            manufacturer="Green Button",
            model="Electricity",
        )

    @property
    def native_value(self) -> float | None:
        """Return the current total cost value."""
        if not self.coordinator.data or not self.coordinator.data.get("usage_points"):
            return self._cached_native_value  # Return cached value instead of None

        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if not meter_reading:
            return self._cached_native_value  # Return cached value instead of None

        # Set currency if available
        currency = getattr(meter_reading.reading_type, "currency", None)
        if currency:
            self._attr_native_unit_of_measurement = currency

        total_cost = 0.0
        for interval_block in meter_reading.interval_blocks:
            for interval_reading in interval_block.interval_readings:
                cost_raw = getattr(interval_reading, "cost", 0) or 0
                power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
                total_cost += cost_raw * (10 ** power_multiplier)

        self._cached_native_value = float(total_cost)  # Cache the value
        return self._cached_native_value

    @property
    def available(self) -> bool:
        available = self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )
        _LOGGER.info(
            "Cost Sensor %s: available property evaluated to %s (last_update_success=%s, data is not None=%s)",
            getattr(self, 'entity_id', self._attr_unique_id),
            available,
            self.coordinator.last_update_success,
            self.coordinator.data is not None,
        )
        return available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if not meter_reading:
            return {}

        attributes = {
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
    def long_term_statistics_id(self) -> str:
        return self.entity_id

    @property
    def name(self) -> str:
        return self._attr_name or f"{self.coordinator.config_entry.title} Natural Gas Cost"

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        _LOGGER.info(
            "Cost Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Initialize entity state to make it "available" for Energy Dashboard
        # native_value property will return _cached_native_value (initialized to 0.0)
        self.async_write_ha_state()
        _LOGGER.info(
            "Cost Sensor %s: Initialized state to make entity available",
            self.entity_id,
        )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

    def _handle_coordinator_update(self) -> None:
        # DO NOT call super()._handle_coordinator_update()! (See GreenButtonSensor for explanation)

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            usage_points = self.coordinator.data["usage_points"]
            for usage_point in usage_points:
                for meter_reading in usage_point.meter_readings:
                    if meter_reading.id == self._meter_reading_id:
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )

    async def update_sensor_and_statistics(self, meter_reading: model.MeterReading) -> None:
        """
        Update the sensor's displayed value and long-term cost statistics from a MeterReading.

        This coroutine computes the total cost from the provided MeterReading and updates
        the sensor instance attributes accordingly, without calling async_write_ha_state().
        It also triggers updating of Home Assistant's long-term statistics (cost statistics)
        when a Home Assistant instance is available on the sensor.

        Behavior:
        - Iterates all interval_blocks and their interval_readings in meter_reading.
        - For each interval_reading, reads the "cost" attribute (treating missing or None as 0)
            and scales it by the reading_type.power_of_ten_multiplier (10 ** power_of_ten_multiplier).
        - Sums the scaled costs to produce total_cost.
        - If meter_reading.reading_type.currency is present, sets the sensor's
            _attr_native_unit_of_measurement to that currency.
        - Sets the sensor's _attr_native_value to float(total_cost).
        - Does NOT call async_write_ha_state() here — the caller is responsible for writing state
            to Home Assistant if needed.
        - If the sensor has a non-None hass attribute, calls statistics.update_cost_statistics(...)
            with statistics.CostDataExtractor() to refresh long-term statistics.

        Parameters:
        - meter_reading (model.MeterReading): Meter reading data containing interval_blocks,
            interval_readings, and reading_type metadata (including currency and power_of_ten_multiplier).

        Returns:
        - None

        Notes:
        - This is an async function and must be awaited.
        - Side effects: mutates sensor instance attributes:
                - _attr_native_unit_of_measurement (may be set to currency)
                - _attr_native_value (set to computed total cost as float)
            and may update Home Assistant long-term statistics.
        - Designed to be called from within Home Assistant's async context.
        """
        # Update state
        total_cost = 0.0
        for interval_block in meter_reading.interval_blocks:
            for interval_reading in interval_block.interval_readings:
                cost_raw = getattr(interval_reading, "cost", 0) or 0
                power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
                total_cost += cost_raw * (10 ** power_multiplier)

        # Set currency
        currency = getattr(meter_reading.reading_type, "currency", None)
        if currency:
            self._attr_native_unit_of_measurement = currency

        self._attr_native_value = float(total_cost)

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

        # Update long-term statistics
        if hasattr(self, "hass") and self.hass is not None:
            await statistics.update_cost_statistics(
                self.hass,
                self,
                statistics.CostDataExtractor(),
                meter_reading,
            )


class GreenButtonGasSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """Gas consumption sensor (m³, total increasing)."""

    _attr_device_class = SensorDeviceClass.GAS
    # Set state_class to satisfy Energy Dashboard requirements
    # BUT we disable recorder statistics for this entity
    # and manually manage statistics via async_import_statistics
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "m³"
    _attr_has_entity_name = True

    def __init__(self, coordinator: GreenButtonCoordinator, meter_reading_id: str) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._cached_native_value: float = 0.0  # Initialize to 0 for Energy Dashboard
        clean_id = meter_reading_id.split("/")[-1] if "/" in meter_reading_id else meter_reading_id
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_gas"
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Usage"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas entities under a dedicated device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")},
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float | None:
        # Try to get meter reading first (normal case)
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if meter_reading:
            total = 0.0
            for block in meter_reading.interval_blocks:
                for rd in block.interval_readings:
                    total += float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
            self._cached_native_value = total  # Cache the value
            return self._cached_native_value

        # If no meter reading, this might be a UsagePoint ID (UsageSummary-only case)
        # In this case, return the sum of all UsageSummary consumption values
        for usage_point in self.coordinator.usage_points:
            if usage_point.id == self._meter_reading_id and usage_point.usage_summaries:
                total = sum(us.consumption_m3 or 0.0 for us in usage_point.usage_summaries)
                if total > 0:
                    self._cached_native_value = total
                    return self._cached_native_value

        return self._cached_native_value  # Return cached value instead of None

    @property
    def long_term_statistics_id(self) -> str:
        return self.entity_id

    @property
    def name(self) -> str:
        return self._attr_name or f"{self.coordinator.config_entry.title} Natural Gas Usage"

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "m³"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        _LOGGER.info(
            "Gas Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Initialize entity state to make it "available" for Energy Dashboard
        # native_value property will return _cached_native_value (initialized to 0.0)
        self.async_write_ha_state()
        _LOGGER.info(
            "Gas Sensor %s: Initialized state to make entity available",
            self.entity_id,
        )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        super()._handle_coordinator_update()

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if meter_reading.id == self._meter_reading_id:
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
                    if usage_point.id == self._meter_reading_id and usage_point.usage_summaries:
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics_from_summaries(usage_point)
                        )
                        break
    async def update_sensor_and_statistics(self, meter_reading: model.MeterReading) -> None:
        """
        Asynchronously update the sensor's native value from a MeterReading and import
        gas usage statistics for the entity.

        This coroutine computes the total measured value by iterating over
        meter_reading.interval_blocks and summing each interval_reading.value scaled by
        its reading_type.power_of_ten_multiplier (value * 10**power_of_ten_multiplier).
        The computed total is stored on the entity as self._attr_native_value. This
        function intentionally does not call async_write_ha_state(); the caller is
        responsible for writing the entity state to Home Assistant.

        After updating the sensor value, the function retrieves usage summaries for the
        current meter reading from the coordinator and determines the gas usage
        allocation mode from the config entry (options first, then data, falling back
        to "daily_readings"). It then awaits statistics.update_gas_statistics to import
        or update Home Assistant statistics for this entity.

        Parameters
        ----------
        meter_reading : model.MeterReading
            The Green Button MeterReading object containing interval_blocks and
            interval_readings used to compute the total consumption for this sensor.

        Returns
        -------
        None

        Side effects
        ------------
        - Mutates self._attr_native_value with the computed total.
        - Calls coordinator.get_usage_summaries_for_meter_reading(...) and may access
          coordinator.config_entry to determine allocation mode.
        - Awaits statistics.update_gas_statistics(...), which updates Home Assistant's
          recorded statistics for this entity.

        Raises
        ------
        Any exception raised by coordinator methods or statistics.update_gas_statistics
        is propagated to the caller.

        Notes
        -----
        - This is an async coroutine and must be awaited or scheduled on the Home
          Assistant event loop.
        - Do NOT call async_write_ha_state() inside this method; that is handled
          elsewhere by the integration.
        """
        # Update entity state
        total = 0.0
        for block in meter_reading.interval_blocks:
            for rd in block.interval_readings:
                total += float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
        self._attr_native_value = total

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

        # Import gas m³ statistics per selected allocation mode
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(self._meter_reading_id)
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )
        await statistics.update_gas_statistics(
            self.hass,
            self,
            meter_reading,
            usage_summaries=summaries,
            allocation_mode=usage_allocation_mode,
        )

    async def update_sensor_and_statistics_from_summaries(self, usage_point: model.UsagePoint) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no daily MeterReadings)."""
        # Update entity state (sum of all UsageSummary consumption values)
        total = sum(us.consumption_m3 or 0.0 for us in usage_point.usage_summaries)
        self._attr_native_value = total if total > 0 else 0.0

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

        # Import gas statistics in monthly_increment mode (UsageSummaries only)
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )

        if usage_allocation_mode == "monthly_increment" and usage_point.usage_summaries:
            _LOGGER.info(
                "Gas Sensor %s: Generating statistics from UsageSummaries (no daily readings)",
                self.entity_id,
            )
            # Call update_gas_statistics with no meter_reading (will use only UsageSummaries)
            await statistics.update_gas_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                usage_summaries=list(usage_point.usage_summaries),
                allocation_mode=usage_allocation_mode,
            )
        else:
            _LOGGER.warning(
                "Gas Sensor %s: Cannot generate statistics - monthly_increment mode required for UsageSummary-only data",
                self.entity_id,
            )


class GreenButtonGasCostSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """Gas cost sensor (monetary total) using UsageSummary pro-rated per day."""

    _attr_device_class = SensorDeviceClass.MONETARY
    # Set state_class to TOTAL (not TOTAL_INCREASING) for monetary sensors
    # Monetary values can decrease (refunds, adjustments), so use TOTAL
    # We disable recorder statistics for this entity and manually manage via async_import_statistics
    _attr_state_class = SensorStateClass.TOTAL
    _attr_has_entity_name = True

    def __init__(self, coordinator: GreenButtonCoordinator, meter_reading_id: str) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        clean_id = meter_reading_id.split("/")[-1] if "/" in meter_reading_id else meter_reading_id
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_gas_cost"
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Cost"
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas cost under the gas device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")},
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float | None:
        # Sensor state is cumulative total of UsageSummary total_cost
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(self._meter_reading_id)
        if not summaries:
            return 0.0
        return float(sum(us.total_cost for us in summaries))

    @property
    def long_term_statistics_id(self) -> str:
        return self.entity_id

    @property
    def name(self) -> str:
        if self._attr_name:
            return self._attr_name
        # Fallback: replace "Electricity" with "Natural Gas" if present
        base_name = self.coordinator.config_entry.title
        if "Electricity" in base_name:
            return base_name.replace("Electricity", "Natural Gas") + " Cost"
        return f"{base_name} Gas Cost"

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        _LOGGER.info(
            "Gas Cost Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Initialize entity state to make it "available" for Energy Dashboard
        # native_value property will return a value (defaults to 0.0 when no summaries)
        self.async_write_ha_state()
        _LOGGER.info(
            "Gas Cost Sensor %s: Initialized state to make entity available",
            self.entity_id,
        )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        super()._handle_coordinator_update()

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if meter_reading.id == self._meter_reading_id:
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
                    if usage_point.id == self._meter_reading_id and usage_point.usage_summaries:
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics_from_summaries(usage_point)
                        )
                        break

    async def update_sensor_and_statistics(self, meter_reading: model.MeterReading) -> None:
        # Update state
        self._attr_native_value = self.native_value

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

        # Update long-term statistics with pro-rated daily cost
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(self._meter_reading_id)
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        await statistics.update_gas_cost_statistics(
            self.hass,
            self,
            meter_reading,
            summaries,
            allocation_mode=allocation_mode,
        )

    async def update_sensor_and_statistics_from_summaries(self, usage_point: model.UsagePoint) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no MeterReadings)."""
        # Update entity state (sum of all UsageSummary total_cost values)
        total = sum(us.total_cost for us in usage_point.usage_summaries)
        self._attr_native_value = total if total > 0 else 0.0

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

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

        # Call update_gas_cost_statistics with no meter_reading (will use only UsageSummaries)
        await statistics.update_gas_cost_statistics(
            self.hass,
            self,
            None,  # No meter reading available
            list(usage_point.usage_summaries),
            allocation_mode=allocation_mode,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Green Button sensor entities from a config entry."""
    _LOGGER.debug("Setting up Green Button sensor platform")

    # Get the coordinator from hass.data
    coordinator: GreenButtonCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    # Track only entities created by this setup so we can safely write state afterwards
    created_entities: list[SensorEntity] = []

    async def _async_update_created_entities() -> None:
        """Write state for newly created entities after they are added to HA."""
        if not created_entities:
            _LOGGER.debug("No newly created entities to update")
            return

        _LOGGER.info(
            "Updating %d newly created sensor entities (data_available=%s, last_update_success=%s)",
            len(created_entities),
            coordinator.data is not None,
            coordinator.last_update_success,
        )

        for entity in created_entities:
            _LOGGER.info(
                "Calling async_write_ha_state() on newly created sensor %s",
                getattr(entity, "entity_id", "UNKNOWN"),
            )
            entity.async_write_ha_state()

    def _async_create_entities() -> None:
        """Create new entities when data becomes available."""
        entities = []
        entity_registry = async_get_entity_registry(hass)

        # Debug: Check what data is available
        # _LOGGER.debug("Entity creation: coordinator.data = %s", coordinator.data)
        # _LOGGER.debug(
        #     "Entity creation: usage_points count = %d", len(coordinator.usage_points)
        # )

        # If we have data, create entities for each meter reading
        if coordinator.data and coordinator.data.get("usage_points"):
            for usage_point in coordinator.usage_points:
                is_gas = usage_point.sensor_device_class == SensorDeviceClass.GAS

                # Check if this is a gas usage point with only UsageSummaries (no daily meter readings)
                # This happens with some Enbridge gas data that only provides monthly billing summaries
                if is_gas and not usage_point.meter_readings and usage_point.usage_summaries:
                    # Check if monthly_increment mode is enabled
                    allocation_mode = (
                        entry.options.get("gas_usage_allocation")
                        or entry.data.get("gas_usage_allocation")
                        or "daily_readings"
                    )

                    if allocation_mode == "monthly_increment":
                        # Use UsagePoint ID as the "meter_reading_id" - the sensor will handle this
                        gas_sensor = GreenButtonGasSensor(coordinator, usage_point.id)
                        # Check if this entity already exists in the registry
                        if gas_sensor.unique_id:
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Gas sensor %s already exists (entity_id: %s), skipping creation",
                                    gas_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                entities.append(gas_sensor)
                                created_entities.append(gas_sensor)
                                _LOGGER.info(
                                    "Created gas sensor %s for UsagePoint %s (UsageSummary only, no daily readings)",
                                    gas_sensor.unique_id,
                                    usage_point.id,
                                )
                        else:
                            _LOGGER.warning("Gas sensor has no unique_id, skipping creation")

                        gas_cost_sensor = GreenButtonGasCostSensor(coordinator, usage_point.id)
                        # Check if this entity already exists in the registry
                        if gas_cost_sensor.unique_id:
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_cost_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Gas cost sensor %s already exists (entity_id: %s), skipping creation",
                                    gas_cost_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                entities.append(gas_cost_sensor)
                                created_entities.append(gas_cost_sensor)
                                _LOGGER.info(
                                    "Created gas cost sensor %s for UsagePoint %s (UsageSummary only, no daily readings)",
                                    gas_cost_sensor.unique_id,
                                    usage_point.id,
                                )
                        else:
                            _LOGGER.warning("Gas cost sensor has no unique_id, skipping creation")
                    else:
                        _LOGGER.warning(
                            "Gas UsagePoint %s has only UsageSummaries (no daily readings). "
                            "Enable 'monthly_increment' mode in integration settings to use this data.",
                            usage_point.id,
                        )

                # Gas meter readings: create exactly one pair per UsagePoint using a deterministic choice
                if is_gas and usage_point.meter_readings:
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
                        _LOGGER.info(
                            "Skipping gas UsagePoint %s because no meter reading has interval data",
                            usage_point.id,
                        )
                        continue

                    # Pick a deterministic meter reading so repeated calls don't create new pairs
                    primary_mr = sorted(eligible_mrs, key=lambda mr: mr.id)[0]

                    gas_sensor = GreenButtonGasSensor(coordinator, primary_mr.id)
                    # Check if this entity already exists in the registry
                    if gas_sensor.unique_id:
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, gas_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Gas sensor %s already exists (entity_id: %s), skipping creation",
                                gas_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            entities.append(gas_sensor)
                            created_entities.append(gas_sensor)
                            _LOGGER.info(
                                "Created gas sensor %s for meter reading %s (UsagePoint %s; %d eligible meter readings)",
                                gas_sensor.unique_id,
                                primary_mr.id,
                                usage_point.id,
                                len(eligible_mrs),
                            )
                    else:
                        _LOGGER.warning("Gas sensor has no unique_id, skipping creation")

                    gas_cost_sensor = GreenButtonGasCostSensor(coordinator, primary_mr.id)
                    # Check if this entity already exists in the registry
                    if gas_cost_sensor.unique_id:
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, gas_cost_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Gas cost sensor %s already exists (entity_id: %s), skipping creation",
                                gas_cost_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            entities.append(gas_cost_sensor)
                            created_entities.append(gas_cost_sensor)
                            _LOGGER.info(
                                "Created gas cost sensor %s for meter reading %s (UsagePoint %s)",
                                gas_cost_sensor.unique_id,
                                primary_mr.id,
                                usage_point.id,
                            )
                    else:
                        _LOGGER.warning("Gas cost sensor has no unique_id, skipping creation")

                    # Skip electricity creation for this usage point
                    continue

                # Electricity meter readings (or non-gas usage points)
                # Create exactly one pair per UsagePoint to avoid duplicates on reload
                if usage_point.meter_readings:
                    eligible_electric_mrs = [
                        mr
                        for mr in usage_point.meter_readings
                        if mr.interval_blocks
                        and any(
                            ir.value is not None
                            for blk in mr.interval_blocks
                            for ir in blk.interval_readings
                        )
                    ]

                    if not eligible_electric_mrs:
                        _LOGGER.info(
                            "Skipping electricity UsagePoint %s because no meter reading has interval data",
                            usage_point.id,
                        )
                        continue

                    # Pick a deterministic meter reading (sorted by ID for consistency)
                    primary_electric_mr = sorted(eligible_electric_mrs, key=lambda mr: mr.id)[0]

                    energy_sensor = GreenButtonSensor(coordinator, primary_electric_mr.id)
                    # Check if this entity already exists in the registry
                    if energy_sensor.unique_id:
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, energy_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Energy sensor %s already exists (entity_id: %s), skipping creation",
                                energy_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            entities.append(energy_sensor)
                            created_entities.append(energy_sensor)
                            _LOGGER.info(
                                "Created energy sensor %s for meter reading %s (UsagePoint %s; %d eligible meter readings)",
                                energy_sensor.unique_id,
                                primary_electric_mr.id,
                                usage_point.id,
                                len(eligible_electric_mrs),
                            )
                    else:
                        _LOGGER.warning("Energy sensor has no unique_id, skipping creation")

                    cost_sensor = GreenButtonCostSensor(coordinator, primary_electric_mr.id)
                    # Check if this entity already exists in the registry
                    if cost_sensor.unique_id:
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, cost_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Cost sensor %s already exists (entity_id: %s), skipping creation",
                                cost_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            entities.append(cost_sensor)
                            created_entities.append(cost_sensor)
                            _LOGGER.info(
                                "Created cost sensor %s for meter reading %s (UsagePoint %s)",
                                cost_sensor.unique_id,
                                primary_electric_mr.id,
                                usage_point.id,
                            )
                    else:
                        _LOGGER.warning("Cost sensor has no unique_id, skipping creation")

        # Add new entities to Home Assistant
        if entities:
            async_add_entities(entities)
            _LOGGER.info("Added %d new Green Button sensor entities", len(entities))

            # After adding, schedule a state write for just-created entities
            _schedule_hass_task_from_any_thread(hass, _async_update_created_entities())

    # Create initial entities (if any data is already available)
    _async_create_entities()

    # Subscribe to coordinator updates to create entities when new data arrives
    entry.async_on_unload(coordinator.async_add_listener(_async_create_entities))
