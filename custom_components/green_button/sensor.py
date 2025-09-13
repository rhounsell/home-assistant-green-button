from functools import cached_property
from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import GreenButtonCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: GreenButtonCoordinator = hass.data[DOMAIN][config_entry.entry_id]
    entities = [GreenButtonEnergySensor(coordinator)]
    async_add_entities(entities)

class GreenButtonEnergySensor(CoordinatorEntity, SensorEntity):
    """Sensor for Green Button energy usage."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = "kWh"
    _attr_icon="mdi:lightning-bolt"

    def __init__(self, coordinator: GreenButtonCoordinator, entity_unique_id: str = DOMAIN) -> None:
        super().__init__(coordinator)
        self.coordinator: GreenButtonCoordinator = coordinator # Type hint for coordinator
        self._attr_name = "Green Button Energy Usage"
        self._attr_unique_id = entity_unique_id

    @cached_property
    def native_value(self) -> float | None:
        """Return the latest energy value."""
        if self.coordinator.interval_blocks:
            # Replace with actual logic to extract the latest value from self.coordinator.data
            return self.coordinator.get_latest_cumulative_energy_kwh()
        return None

    async def async_update(self):
        """Request coordinator update."""
        await self.coordinator.async_request_refresh()
