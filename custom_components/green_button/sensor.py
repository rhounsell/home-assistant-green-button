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
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{meter_reading_id}"
        self._attr_name = f"Energy Usage {meter_reading_id}"

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
        return f"sensor.{self.unique_id}"

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
            await statistics.update_statistics(
                self.hass,
                self,
                statistics.DefaultDataExtractor(),
                meter_reading,
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

    # Create sensor entities for each meter reading found in the data
    entities = []

    # If we have data, create entities for each meter reading
    if coordinator.data and coordinator.data.get("usage_points"):
        meter_readings = coordinator.get_meter_readings()
        for meter_reading in meter_readings:
            sensor = GreenButtonSensor(coordinator, meter_reading.id)
            entities.append(sensor)
            _LOGGER.debug(
                "Created sensor entity %s for meter reading %s",
                sensor.unique_id,
                meter_reading.id,
            )
    else:
        # Create a default sensor for when no data is available yet
        sensor = GreenButtonSensor(coordinator, "default_energy_meter")
        entities.append(sensor)
        _LOGGER.debug("Created default sensor entity %s", sensor.unique_id)

    # Add entities to Home Assistant
    if entities:
        async_add_entities(entities)
        _LOGGER.info("Added %d Green Button sensor entities", len(entities))
    else:
        _LOGGER.warning("No sensor entities created")
