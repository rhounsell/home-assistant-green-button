"""Sensor platform for the Green Button integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import model
from . import statistics
from .coordinator import GreenButtonCoordinator
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class GreenButtonSensor(CoordinatorEntity[GreenButtonCoordinator], SensorEntity):
    """A sensor for Green Button energy data."""

    _attr_device_class = SensorDeviceClass.ENERGY
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
            "Sensor %s: Entity added to Home Assistant, triggering initial statistics generation",
            self.entity_id,
        )

        # Trigger statistics generation for initial data
        # This ensures statistics are created even when coordinator already has data at startup
        if self.coordinator.data and "usage_points" in self.coordinator.data:
            usage_points = self.coordinator.data["usage_points"]
            for usage_point in usage_points:
                for meter_reading in usage_point.meter_readings:
                    if meter_reading.id == self._meter_reading_id:
                        _LOGGER.info(
                            "Sensor %s: Scheduling initial statistics generation for meter reading %s",
                            self.entity_id,
                            meter_reading.id,
                        )
                        self.hass.async_create_task(
                            self.update_sensor_and_statistics(meter_reading)
                        )
                        break
        else:
            _LOGGER.warning(
                "Sensor %s: No coordinator data available during entity initialization",
                self.entity_id,
            )

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        # Update entity state (calls native_value property)
        super()._handle_coordinator_update()

        _LOGGER.info(
            "Sensor %s: Coordinator update - data available: %s",
            self.entity_id,
            bool(self.coordinator.data),
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
                        self.hass.async_create_task(
                            self.update_sensor_and_statistics(meter_reading)
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

        # Update the entity state
        self.async_write_ha_state()

        # Update statistics for Energy Dashboard
        if hasattr(self, "hass") and self.hass is not None:
            _LOGGER.info(
                "Sensor %s: Updating statistics with total energy %s kWh",
                self.entity_id,
                self._attr_native_value,
            )
            await statistics.update_statistics(
                self.hass,
                self,
                statistics.DefaultDataExtractor(),
                meter_reading,
            )
            _LOGGER.info(
                "Sensor %s: Statistics update completed",
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

    # Track created entities to avoid duplicates
    created_entities: set[str] = set()

    def _async_create_entities() -> None:
        """Create new entities when data becomes available."""
        entities = []

        # Debug: Check what data is available
        _LOGGER.info("Entity creation: coordinator.data = %s", coordinator.data)
        _LOGGER.info(
            "Entity creation: usage_points count = %d", len(coordinator.usage_points)
        )

        # If we have data, create entities for each meter reading
        if coordinator.data and coordinator.data.get("usage_points"):
            meter_readings = coordinator.get_meter_readings()
            _LOGGER.info(
                "Found %d meter readings to create sensors for", len(meter_readings)
            )

            for meter_reading in meter_readings:
                if meter_reading.id not in created_entities:
                    sensor = GreenButtonSensor(coordinator, meter_reading.id)
                    entities.append(sensor)
                    created_entities.add(meter_reading.id)
                    _LOGGER.info(
                        "Created sensor entity %s for meter reading %s",
                        sensor.unique_id,
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
