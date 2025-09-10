from functools import cached_property
from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import GreenButtonCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: GreenButtonCoordinator = hass.data["green_button_coordinator"]
    entities = [GreenButtonEnergySensor(coordinator)]
    async_add_entities(entities)

class GreenButtonEnergySensor(SensorEntity):
    """Sensor for Green Button energy usage."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"

    def __init__(self, coordinator: GreenButtonCoordinator):
        self.coordinator = coordinator
        self._attr_name = "Green Button Energy Usage"
        self._attr_unique_id = "green_button_energy_usage"

    @cached_property
    def native_value(self) -> float | None:
        """Return the latest energy value."""
        if self.coordinator.data:
            # Replace with actual logic to extract the latest value from self.coordinator.data
            return self.coordinator.data.get_latest_energy_kwh()
        return None

    async def async_update(self):
        """Request coordinator update."""
        await self.coordinator.async_request_refresh()
