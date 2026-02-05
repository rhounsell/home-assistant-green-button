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
    # We set state_class to satisfy HA's validation (prevents "fix issue" warnings)
    # but we return None from native_value so there's no state history for HA
    # to auto-compile. We manually import statistics via async_import_statistics().
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
        """Return the native value of the sensor (cumulative total).
        
        NOTE: To prevent HA's automatic statistics compilation from creating duplicate
        records, you MUST exclude this entity from the recorder in configuration.yaml:
        
        recorder:
          exclude:
            entities:
              - sensor.home_electricity_usage
              - sensor.home_electricity_cost
              - sensor.home_natural_gas_usage
              - sensor.home_natural_gas_cost
        
        We manually import statistics via async_import_statistics().
        The Energy Dashboard uses those manually imported statistics.
        """
        return self._attr_native_value

    @property
    def available(self) -> bool:
        available = self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )
        _LOGGER.debug(
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
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the native unit of measurement for statistics protocol."""
        return self._attr_native_unit_of_measurement or "kWh"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass, trigger statistics generation without updating sensor state.
        
        IMPORTANT: We do NOT call async_write_ha_state() here!
        This integration manually imports historical statistics via async_import_statistics().
        If we write a cumulative total as the sensor state, HA's automatic statistics
        compilation (in homeassistant/components/sensor/recorder.py) would see a massive
        state jump and generate a corrupted statistics record for "today's date" showing
        fake consumption equal to the difference from the last recorded state.
        
        The Energy Dashboard uses the statistics we import, not the sensor state.
        The sensor state is only used by HA's automatic statistics compilation, which
        we want to avoid since we manage statistics manually.
        """
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Sensor %s: Entity added to Home Assistant (NOT writing state to avoid recorder auto-statistics)",
            self.entity_id,
        )

        # Set the internal value for reference but DON'T call async_write_ha_state()
        # This prevents HA's recorder from auto-generating statistics from state changes
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if meter_reading:
            # Calculate total energy from all interval blocks (for internal tracking only)
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
            # DO NOT call async_write_ha_state() - see docstring above
            _LOGGER.info(
                "Sensor %s: Internal state set to %.2f kWh (NOT written to HA to avoid recorder spike)",
                self.entity_id,
                self._attr_native_value,
            )
        else:
            # If no data, set to 0.0 for internal reference
            if self._attr_native_value is None:
                self._attr_native_value = 0.0
            # DO NOT call async_write_ha_state() - see docstring above
            _LOGGER.info(
                "Sensor %s: Internal state set to 0.0 (no meter reading found, NOT written to HA)",
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

        _LOGGER.debug(
            "🔍 %s: Setting sensor state to %.2f kWh (cumulative).",
            self.entity_id,
            self._attr_native_value,
        )

        # NOTE: Do NOT call async_write_ha_state() here!
        # Calling it before statistics update causes HA's Recorder to automatically
        # create a statistics record using the sensor's cumulative state value,
        # which creates corrupted records (state/sum swap, massive consumption values).
        # The coordinator will handle state updates, and we manually manage statistics below.

        # Update statistics for Energy Dashboard (run in background to not block startup)
        if hasattr(self, "hass") and self.hass is not None:
            asyncio.create_task(
                self._update_statistics_async(meter_reading),
                name=f"green_button_stats_{self.entity_id}"
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
            _LOGGER.info(
                "%s: Statistics update completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Statistics update failed.",
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
        _LOGGER.debug(
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
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass, trigger statistics generation without updating sensor state.
        
        See GreenButtonSensor.async_added_to_hass() for detailed explanation.
        We avoid calling async_write_ha_state() to prevent HA's automatic statistics
        compilation from generating corrupted records.
        """
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Cost Sensor %s: Entity added to Home Assistant (NOT writing state to avoid recorder auto-statistics)",
            self.entity_id,
        )

        # DO NOT call async_write_ha_state() - see GreenButtonSensor docstring

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

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

        # Update long-term statistics (run in background to not block startup)
        if hasattr(self, "hass") and self.hass is not None:
            asyncio.create_task(
                self._update_cost_statistics_async(meter_reading),
                name=f"green_button_cost_stats_{self.entity_id}"
            )
            _LOGGER.debug(
                "%s: Cost statistics update scheduled in background.",
                self.entity_id,
            )

    async def _update_cost_statistics_async(self, meter_reading: model.MeterReading) -> None:
        """Update cost statistics in background without blocking."""
        try:
            await statistics.update_cost_statistics(
                self.hass,
                self,
                statistics.CostDataExtractor(),
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


class GreenButtonGasSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """Gas consumption sensor (m³, total increasing)."""

    _attr_device_class = SensorDeviceClass.GAS
    # We set state_class to satisfy HA's validation (prevents "fix issue" warnings)
    # but we return None from native_value so there's no state history for HA
    # to auto-compile. We manually import statistics via async_import_statistics().
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
        """Return the native value of the sensor (cumulative total).
        
        NOTE: To prevent HA's automatic statistics compilation from creating duplicate
        records, you MUST exclude this entity from the recorder in configuration.yaml.
        See GreenButtonSensor.native_value docstring for configuration example.
        
        We manually import statistics via async_import_statistics().
        The Energy Dashboard uses those manually imported statistics.
        """
        # Try to get meter reading first (normal case)
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if meter_reading:
            total = 0.0
            for block in meter_reading.interval_blocks:
                for rd in block.interval_readings:
                    total += float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
            self._cached_native_value = total
            return self._cached_native_value

        # If no meter reading, this might be a UsagePoint ID (UsageSummary-only case)
        # In this case, return the sum of all UsageSummary consumption values
        for usage_point in self.coordinator.usage_points:
            if usage_point.id == self._meter_reading_id and usage_point.usage_summaries:
                total = sum(us.consumption_m3 or 0.0 for us in usage_point.usage_summaries)
                if total > 0:
                    self._cached_native_value = total
                    return self._cached_native_value

        return self._cached_native_value

    @property
    def long_term_statistics_id(self) -> str:
        return self.entity_id

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "m³"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass, trigger statistics generation without updating sensor state.
        
        See GreenButtonSensor.async_added_to_hass() for detailed explanation.
        We avoid calling async_write_ha_state() to prevent HA's automatic statistics
        compilation from generating corrupted records.
        """
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Gas Sensor %s: Entity added to Home Assistant (NOT writing state to avoid recorder auto-statistics)",
            self.entity_id,
        )

        # DO NOT call async_write_ha_state() - see GreenButtonSensor docstring

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        # DO NOT call super()._handle_coordinator_update()!
        # That would call async_write_ha_state(), which triggers HA's Recorder
        # to auto-generate statistics using the cumulative sensor value.
        # We manually manage statistics via async_import_statistics().

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
        # Run statistics update in background to not block startup
        asyncio.create_task(
            self._update_gas_statistics_async(meter_reading, summaries, usage_allocation_mode),
            name=f"green_button_gas_stats_{self.entity_id}"
        )
        _LOGGER.debug(
            "%s: Gas statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        usage_allocation_mode: str
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
            _LOGGER.info(
                "%s: Gas statistics update completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update failed.",
                self.entity_id,
            )

    async def update_sensor_and_statistics_from_summaries(self, usage_point: model.UsagePoint) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no daily MeterReadings)."""
        # Update entity state (sum of all UsageSummary consumption values)
        total = sum(us.consumption_m3 or 0.0 for us in usage_point.usage_summaries)
        self._attr_native_value = total if total > 0 else 0.0

        # NOTE: Do NOT call async_write_ha_state() here!
        # See explanation in GreenButtonSensor class above.

        # Import gas statistics in monthly_increment mode (UsageSummaries only)
        # Run in background to not block startup
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
            # Call update_gas_statistics in background - no meter reading available
            asyncio.create_task(
                self._update_gas_statistics_from_summaries_async(usage_point, usage_allocation_mode),
                name=f"green_button_gas_stats_summaries_{self.entity_id}"
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
        self,
        usage_point: model.UsagePoint,
        usage_allocation_mode: str
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
            _LOGGER.info(
                "%s: Gas statistics update (from summaries) completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update (from summaries) failed.",
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
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass, trigger statistics generation without updating sensor state.
        
        See GreenButtonSensor.async_added_to_hass() for detailed explanation.
        We avoid calling async_write_ha_state() to prevent HA's automatic statistics
        compilation from generating corrupted records.
        """
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Gas Cost Sensor %s: Entity added to Home Assistant (NOT writing state to avoid recorder auto-statistics)",
            self.entity_id,
        )

        # DO NOT call async_write_ha_state() - see GreenButtonSensor docstring

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:
        # DO NOT call super()._handle_coordinator_update()!
        # That would call async_write_ha_state(), which triggers HA's Recorder
        # to auto-generate statistics using the cumulative sensor value.
        # We manually manage statistics via async_import_statistics().

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
        # Run in background to not block startup
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(self._meter_reading_id)
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        asyncio.create_task(
            self._update_gas_cost_statistics_async(meter_reading, summaries, allocation_mode),
            name=f"green_button_gas_cost_stats_{self.entity_id}"
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        allocation_mode: str
    ) -> None:
        """Update gas cost statistics in background without blocking."""
        try:
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                meter_reading,
                summaries,
                allocation_mode=allocation_mode,
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

        # Call update_gas_cost_statistics in background - no meter reading available
        asyncio.create_task(
            self._update_gas_cost_statistics_from_summaries_async(usage_point, allocation_mode),
            name=f"green_button_gas_cost_stats_summaries_{self.entity_id}"
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update (from summaries) scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_from_summaries_async(
        self,
        usage_point: model.UsagePoint,
        allocation_mode: str
    ) -> None:
        """Update gas cost statistics from summaries in background without blocking."""
        try:
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                list(usage_point.usage_summaries),
                allocation_mode=allocation_mode,
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
    """Set up Green Button sensor entities from a config entry."""
    _LOGGER.debug("Setting up Green Button sensor platform")

    # Get the coordinator from hass.data
    coordinator: GreenButtonCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]

    # Track only entities created by this setup so we can safely write state afterwards
    created_entities: list[SensorEntity] = []
    # Track entities that have been added to HA to avoid re-adding reused entities on subsequent coordinator updates
    entities_added_to_hass: bool = False

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

        # If we have data, create entities for each meter reading
        if coordinator.data and coordinator.data.get("usage_points"):
            for usage_point in coordinator.usage_points:
                is_gas = usage_point.sensor_device_class == SensorDeviceClass.GAS

                # For gas usage points, check the allocation mode setting first
                if is_gas:
                    allocation_mode = (
                        entry.options.get("gas_usage_allocation")
                        or entry.data.get("gas_usage_allocation")
                        or "daily_readings"
                    )

                    # If monthly_increment mode is enabled, use UsageSummaries regardless of meter readings availability
                    if allocation_mode == "monthly_increment" and usage_point.usage_summaries:
                        # Use UsagePoint ID as the "meter_reading_id" - the sensor will handle this
                        gas_sensor = GreenButtonGasSensor(coordinator, usage_point.id)
                        if gas_sensor.unique_id:
                            # Check if this entity already exists in the registry (to log if reused)
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Reusing gas sensor %s (entity_id: %s)",
                                    gas_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                _LOGGER.info(
                                    "Created gas sensor %s for UsagePoint %s (using monthly increment mode with UsageSummaries)",
                                    gas_sensor.unique_id,
                                    usage_point.id,
                                )
                                created_entities.append(gas_sensor)
                            # Always add to entities list so it gets instantiated and added to HA
                            entities.append(gas_sensor)
                        else:
                            _LOGGER.warning("Gas sensor has no unique_id, skipping creation")

                        gas_cost_sensor = GreenButtonGasCostSensor(coordinator, usage_point.id)
                        if gas_cost_sensor.unique_id:
                            # Check if this entity already exists in the registry (to log if reused)
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_cost_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Reusing gas cost sensor %s (entity_id: %s)",
                                    gas_cost_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                _LOGGER.info(
                                    "Created gas cost sensor %s for UsagePoint %s (using monthly increment mode with UsageSummaries)",
                                    gas_cost_sensor.unique_id,
                                    usage_point.id,
                                )
                                created_entities.append(gas_cost_sensor)
                            # Always add to entities list so it gets instantiated and added to HA
                            entities.append(gas_cost_sensor)
                        else:
                            _LOGGER.warning("Gas cost sensor has no unique_id, skipping creation")
                        
                        # Skip to next usage point since we handled this gas data with monthly mode
                        continue

                    # Gas meter readings: create exactly one pair per UsagePoint using a deterministic choice
                    # (only if NOT using monthly_increment mode)
                    if usage_point.meter_readings:
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
                        if gas_sensor.unique_id:
                            # Check if this entity already exists in the registry (to log if reused)
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Reusing gas sensor %s (entity_id: %s)",
                                    gas_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                _LOGGER.info(
                                    "Created gas sensor %s for meter reading %s (UsagePoint %s; %d eligible meter readings)",
                                    gas_sensor.unique_id,
                                    primary_mr.id,
                                    usage_point.id,
                                    len(eligible_mrs),
                                )
                                created_entities.append(gas_sensor)
                            # Always add to entities list so it gets instantiated and added to HA
                            entities.append(gas_sensor)
                        else:
                            _LOGGER.warning("Gas sensor has no unique_id, skipping creation")

                        gas_cost_sensor = GreenButtonGasCostSensor(coordinator, primary_mr.id)
                        if gas_cost_sensor.unique_id:
                            # Check if this entity already exists in the registry (to log if reused)
                            existing_entity = entity_registry.async_get_entity_id(
                                "sensor", DOMAIN, gas_cost_sensor.unique_id
                            )
                            if existing_entity:
                                _LOGGER.debug(
                                    "Reusing gas cost sensor %s (entity_id: %s)",
                                    gas_cost_sensor.unique_id,
                                    existing_entity,
                                )
                            else:
                                _LOGGER.info(
                                    "Created gas cost sensor %s for meter reading %s (UsagePoint %s)",
                                    gas_cost_sensor.unique_id,
                                    primary_mr.id,
                                    usage_point.id,
                                )
                                created_entities.append(gas_cost_sensor)
                            # Always add to entities list so it gets instantiated and added to HA
                            entities.append(gas_cost_sensor)
                        else:
                            _LOGGER.warning("Gas cost sensor has no unique_id, skipping creation")

                        # Skip electricity creation for this usage point
                        continue
                    else:
                        # Gas usage point has no meter readings and either no summaries or not using monthly mode
                        if not usage_point.usage_summaries:
                            _LOGGER.warning(
                                "Skipping gas UsagePoint %s: no meter readings and no usage summaries",
                                usage_point.id,
                            )
                        else:
                            _LOGGER.warning(
                                "Gas UsagePoint %s has UsageSummaries but monthly_increment mode is not enabled. "
                                "Enable 'Single billing-period usage increment' in integration settings to use this data.",
                                usage_point.id,
                            )
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
                    if energy_sensor.unique_id:
                        # Check if this entity already exists in the registry (to log if reused)
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, energy_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Reusing energy sensor %s (entity_id: %s)",
                                energy_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            _LOGGER.info(
                                "Created energy sensor %s for meter reading %s (UsagePoint %s; %d eligible meter readings)",
                                energy_sensor.unique_id,
                                primary_electric_mr.id,
                                usage_point.id,
                                len(eligible_electric_mrs),
                            )
                            created_entities.append(energy_sensor)
                        # Always add to entities list so it gets instantiated and added to HA
                        entities.append(energy_sensor)
                    else:
                        _LOGGER.warning("Energy sensor has no unique_id, skipping creation")

                    cost_sensor = GreenButtonCostSensor(coordinator, primary_electric_mr.id)
                    if cost_sensor.unique_id:
                        # Check if this entity already exists in the registry (to log if reused)
                        existing_entity = entity_registry.async_get_entity_id(
                            "sensor", DOMAIN, cost_sensor.unique_id
                        )
                        if existing_entity:
                            _LOGGER.debug(
                                "Reusing cost sensor %s (entity_id: %s)",
                                cost_sensor.unique_id,
                                existing_entity,
                            )
                        else:
                            _LOGGER.info(
                                "Created cost sensor %s for meter reading %s (UsagePoint %s)",
                                cost_sensor.unique_id,
                                primary_electric_mr.id,
                                usage_point.id,
                            )
                            created_entities.append(cost_sensor)
                        # Always add to entities list so it gets instantiated and added to HA
                        entities.append(cost_sensor)
                    else:
                        _LOGGER.warning("Cost sensor has no unique_id, skipping creation")

        # Add entities to Home Assistant only if this is the first time (initial setup)
        # or if there are newly created entities. On subsequent coordinator updates,
        # skip async_add_entities if all entities are reused (to avoid duplicate ID errors).
        nonlocal entities_added_to_hass
        if not entities_added_to_hass and entities:
            # First time: add all entities (new and reused)
            async_add_entities(entities)
            entities_added_to_hass = True
            _LOGGER.info("Added %d Green Button sensor entities (%d newly created, %d reused)",
                        len(entities), len(created_entities), len(entities) - len(created_entities))

            # After adding, schedule a state write for created entities only
            # (reused entities will have state written via async_added_to_hass or coordinator updates)
            _schedule_hass_task_from_any_thread(hass, _async_update_created_entities())
        elif entities and created_entities:
            # Subsequent updates: only add if there are newly created entities
            async_add_entities(created_entities)
            _LOGGER.info("Added %d new Green Button sensor entities on data import",
                        len(created_entities))

            # Schedule state write for these new entities
            _schedule_hass_task_from_any_thread(hass, _async_update_created_entities())
        elif entities:
            # All entities are reused, no need to call async_add_entities
            _LOGGER.debug("All %d entities already exist in Home Assistant, skipping async_add_entities",
                         len(entities))

    # Create initial entities (if any data is already available)
    _async_create_entities()

    # Subscribe to coordinator updates to create entities when new data arrives
    entry.async_on_unload(coordinator.async_add_listener(_async_create_entities))
