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
from homeassistant.core import HomeAssistant, CoreState
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.helpers.entity_platform import AddEntitiesCallback
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
    # DO NOT set state_class! We manually manage statistics via async_import_statistics.
    # Setting state_class would cause HA Recorder to auto-generate duplicate/corrupted statistics.
    _attr_state_class = None
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
        # Use the config entry title (name set when adding integration) as the sensor name
        self._attr_name = coordinator.config_entry.title

    @property
    def native_value(self) -> float | None:
        """Return the current energy value."""
        if not self.coordinator.data or not self.coordinator.data.get("usage_points"):
            return None

        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if not meter_reading:
            return None

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
            return total_energy / 1000.0
        return 0

    @property
    def native_value(self):
        """Return the native value of the sensor."""
        value = self._attr_native_value
        if value is not None and value > 10000:  # Log if suspiciously high
            # Get the call stack to see WHO is accessing this value
            stack = traceback.extract_stack()
            caller_info = []
            for frame in stack[-5:-1]:  # Last 4 frames before this one
                if 'green_button' not in frame.filename:  # Only non-Green Button frames
                    caller_info.append(f"{frame.filename}:{frame.lineno} in {frame.name}")
            
            _LOGGER.warning(
                "🔍 [DIAGNOSTIC] %s: native_value property accessed, returning %.2f kWh (cumulative). "
                "Called from: %s",
                self.entity_id if hasattr(self, 'entity_id') else 'unknown',
                value,
                ' -> '.join(caller_info) if caller_info else 'internal',
            )
        return value

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )

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
        """When entity is added to hass, trigger statistics generation."""
        await super().async_added_to_hass()

        _LOGGER.info(
            "Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        # DO NOT call super()._handle_coordinator_update()!
        # That would call async_write_ha_state(), which triggers HA's Recorder
        # to auto-generate statistics using the cumulative sensor value,
        # creating corrupted records (state/sum swap, negative sums).
        # We manually manage statistics in update_sensor_and_statistics(),
        # so we only need to mark the entity as available for updates.
        
        _LOGGER.warning(
            "🔍 [DIAGNOSTIC] %s: Coordinator update received. "
            "NOT calling async_write_ha_state() to prevent corruption.",
            self.entity_id if hasattr(self, 'entity_id') else 'unknown',
        )

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

        _LOGGER.warning(
            "🔍 [DIAGNOSTIC] %s: Setting sensor state to %.2f kWh (cumulative). "
            "Stack trace will show if this is called from unexpected location.",
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
            _LOGGER.warning(
                "🔍 [DIAGNOSTIC] %s: Starting statistics update with total energy %.2f kWh. "
                "This should write hourly intervals, NOT cumulative totals.",
                self.entity_id,
                self._attr_native_value,
            )
            await statistics.update_statistics(
                self.hass,
                self,
                statistics.DefaultDataExtractor(),
                meter_reading,
            )
            _LOGGER.warning(
                "🔍 [DIAGNOSTIC] %s: Statistics update completed. "
                "Check database to verify no corruption was introduced.",
                self.entity_id,
            )


class GreenButtonCostSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """A sensor for Green Button monetary cost data (total)."""

    _attr_device_class = SensorDeviceClass.MONETARY
    # DO NOT set state_class! We manually manage statistics via async_import_statistics.
    # Setting state_class would cause HA Recorder to auto-generate duplicate/corrupted statistics.
    _attr_state_class = None
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
    ) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id

        # Build unique id with cost suffix
        clean_id = (
            meter_reading_id.split("/")[-1]
            if "/" in meter_reading_id
            else meter_reading_id
        )
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_cost"
        # Name is entry title + " Cost"
        base_name = coordinator.config_entry.title
        self._attr_name = f"{base_name} Cost"

        # Default currency; will be set on first update if available from reading type
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def native_value(self) -> float | None:
        """Return the current total cost value."""
        if not self.coordinator.data or not self.coordinator.data.get("usage_points"):
            return None

        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if not meter_reading:
            return None

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

        return float(total_cost)

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success and (
            self.coordinator.data is not None
        )

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
        return self._attr_name or f"{self.coordinator.config_entry.title} Cost"

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        _LOGGER.info(
            "Cost Sensor %s: Entity added to Home Assistant",
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
    # DO NOT set state_class! We manually manage statistics via async_import_statistics.
    # Setting state_class would cause HA Recorder to auto-generate duplicate/corrupted statistics.
    _attr_state_class = None
    _attr_native_unit_of_measurement = "m³"
    _attr_has_entity_name = True

    def __init__(self, coordinator: GreenButtonCoordinator, meter_reading_id: str) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        clean_id = meter_reading_id.split("/")[-1] if "/" in meter_reading_id else meter_reading_id
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_gas"
        base_name = coordinator.config_entry.title
        # Replace "Electricity" with "Natural Gas" if present, otherwise just append "Gas"
        if "Electricity" in base_name:
            self._attr_name = base_name.replace("Electricity", "Natural Gas")
        else:
            self._attr_name = f"{base_name} Gas"

    @property
    def native_value(self) -> float | None:
        # Try to get meter reading first (normal case)
        meter_reading = self.coordinator.get_meter_reading_by_id(self._meter_reading_id)
        if meter_reading:
            total = 0.0
            for block in meter_reading.interval_blocks:
                for rd in block.interval_readings:
                    total += float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
            return total
        
        # If no meter reading, this might be a UsagePoint ID (UsageSummary-only case)
        # In this case, return the sum of all UsageSummary consumption values
        for usage_point in self.coordinator.usage_points:
            if usage_point.id == self._meter_reading_id and usage_point.usage_summaries:
                total = sum(us.consumption_m3 or 0.0 for us in usage_point.usage_summaries)
                return total if total > 0 else None
        
        return None

    @property
    def long_term_statistics_id(self) -> str:
        return self.entity_id

    @property
    def name(self) -> str:
        return self._attr_name or f"{self.coordinator.config_entry.title} Gas"

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "m³"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        _LOGGER.info(
            "Gas Sensor %s: Entity added to Home Assistant",
            self.entity_id,
        )

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

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
                usage_summaries=usage_point.usage_summaries,
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
    # DO NOT set state_class! We manually manage statistics via async_import_statistics.
    # Setting state_class would cause HA Recorder to auto-generate duplicate/corrupted statistics.
    _attr_state_class = None
    _attr_has_entity_name = True

    def __init__(self, coordinator: GreenButtonCoordinator, meter_reading_id: str) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        clean_id = meter_reading_id.split("/")[-1] if "/" in meter_reading_id else meter_reading_id
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{clean_id}_gas_cost"
        base_name = coordinator.config_entry.title
        # Replace "Electricity" with "Natural Gas" if present, otherwise just append "Gas Cost"
        if "Electricity" in base_name:
            self._attr_name = base_name.replace("Electricity", "Natural Gas") + " Cost"
        else:
            self._attr_name = f"{base_name} Gas Cost"
        self._attr_native_unit_of_measurement = "CAD"

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

        # Don't generate statistics during bootstrap - rely on _handle_coordinator_update
        # which will be called after bootstrap completes
        # This prevents "Setup timed out for bootstrap" warnings

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
            usage_point.usage_summaries,
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

    # Track created entities to avoid duplicates
    created_entities: set[str] = set()

    def _async_create_entities() -> None:
        """Create new entities when data becomes available."""
        entities = []

        # Debug: Check what data is available
        _LOGGER.debug("Entity creation: coordinator.data = %s", coordinator.data)
        _LOGGER.debug(
            "Entity creation: usage_points count = %d", len(coordinator.usage_points)
        )

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
                        # Create a virtual sensor using the UsagePoint ID since there's no MeterReading
                        virtual_key = f"{usage_point.id}__gas_summary"
                        if virtual_key not in created_entities:
                            # Use UsagePoint ID as the "meter_reading_id" - the sensor will handle this
                            gas_sensor = GreenButtonGasSensor(coordinator, usage_point.id)
                            entities.append(gas_sensor)
                            created_entities.add(virtual_key)
                            _LOGGER.info(
                                "Created gas sensor %s for UsagePoint %s (UsageSummary only, no daily readings)",
                                gas_sensor.unique_id,
                                usage_point.id,
                            )
                        
                        virtual_cost_key = f"{usage_point.id}__gas_cost_summary"
                        if virtual_cost_key not in created_entities:
                            gas_cost_sensor = GreenButtonGasCostSensor(coordinator, usage_point.id)
                            entities.append(gas_cost_sensor)
                            created_entities.add(virtual_cost_key)
                            _LOGGER.info(
                                "Created gas cost sensor %s for UsagePoint %s (UsageSummary only, no daily readings)",
                                gas_cost_sensor.unique_id,
                                usage_point.id,
                            )
                    else:
                        _LOGGER.warning(
                            "Gas UsagePoint %s has only UsageSummaries (no daily readings). "
                            "Enable 'monthly_increment' mode in integration settings to use this data.",
                            usage_point.id,
                        )
                
                # Create sensors for meter readings (normal case)
                for meter_reading in usage_point.meter_readings:
                    if is_gas:
                        # Gas consumption
                        gas_key = f"{meter_reading.id}__gas"
                        if gas_key not in created_entities:
                            gas_sensor = GreenButtonGasSensor(coordinator, meter_reading.id)
                            entities.append(gas_sensor)
                            created_entities.add(gas_key)
                            _LOGGER.info(
                                "Created gas sensor %s for meter reading %s",
                                gas_sensor.unique_id,
                                meter_reading.id,
                            )
                        # Gas cost
                        gas_cost_key = f"{meter_reading.id}__gas_cost"
                        if gas_cost_key not in created_entities:
                            gas_cost_sensor = GreenButtonGasCostSensor(coordinator, meter_reading.id)
                            entities.append(gas_cost_sensor)
                            created_entities.add(gas_cost_key)
                            _LOGGER.info(
                                "Created gas cost sensor %s for meter reading %s",
                                gas_cost_sensor.unique_id,
                                meter_reading.id,
                            )
                    else:
                        # Electricity energy
                        if meter_reading.id not in created_entities:
                            energy_sensor = GreenButtonSensor(coordinator, meter_reading.id)
                            entities.append(energy_sensor)
                            created_entities.add(meter_reading.id)
                            _LOGGER.info(
                                "Created energy sensor %s for meter reading %s",
                                energy_sensor.unique_id,
                                meter_reading.id,
                            )
                        # Electricity cost
                        cost_key = f"{meter_reading.id}__cost"
                        if cost_key not in created_entities:
                            cost_sensor = GreenButtonCostSensor(coordinator, meter_reading.id)
                            entities.append(cost_sensor)
                            created_entities.add(cost_key)
                            _LOGGER.info(
                                "Created cost sensor %s for meter reading %s",
                                cost_sensor.unique_id,
                                meter_reading.id,
                            )

        # Add new entities to Home Assistant
        if entities:
            async_add_entities(entities)
            _LOGGER.info("Added %d new Green Button sensor entities", len(entities))

    # Create initial entities (if any data is already available)
    _async_create_entities()

    # Subscribe to coordinator updates to create entities when new data arrives
    entry.async_on_unload(coordinator.async_add_listener(_async_create_entities))
