# filepath: custom_components/green_button/sensor.py
from homeassistant.components.sensor import SensorEntity
from homeassistant.components.sensor import SensorDeviceClass
from propcache.api import cached_property

class GreenButtonEnergySensor(SensorEntity):
    def __init__(self, energy_total: float) -> None:
        super().__init__()
        self._energy_total: float = energy_total

    @cached_property
    def name(self) -> str:
        return "Green Button Energy Usage"

    @cached_property
    def unique_id(self):
        return "green_button_energy_usage"

    @cached_property
    def native_value(self) -> float:
        # Return the total energy usage in kWh
        return self._energy_total

    @cached_property
    def device_class(self):
        return SensorDeviceClass.ENERGY

    @cached_property
    def state_class(self) -> str:
        return "total_increasing"

    @cached_property
    def native_unit_of_measurement (self):
        return "kWh"
