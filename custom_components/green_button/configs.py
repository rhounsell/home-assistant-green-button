"""Module containing config classes for the component."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any
from typing import Final
from typing import final

import voluptuous as vol
#from homeassistant.backports import enum as backports_enum
from enum import StrEnum
from homeassistant.components import sensor
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import selector

from . import model
from . import side_channel
from .parsers import espi


class InvalidUserInputError(ValueError):
    """Exception to indicate a user input is invalid."""

    def __init__(self, errors: dict[str, str]) -> None:
        """Create a new error."""
        super().__init__(f"Invalid user input: {errors}")
        self.errors = errors


#class _MeterReadingConfigField(backports_enum.StrEnum):
class _MeterReadingConfigField(StrEnum):
    ID = "id"
    SENSOR_DEVICE_CLASS = "sensor_device_class"
    UNIT_OF_MEASUREMENT = "unit_of_measurement"
    CURRENCY = "currency"


@final
@dataclasses.dataclass(frozen=True)
class MeterReadingConfig:
    """A dataclass holding the config for a MeterReading."""

    id: str
    sensor_device_class: sensor.SensorDeviceClass
    unit_of_measurement: str
    currency: str
    initial_meter_reading: model.MeterReading | None

    def to_mapping(self) -> Mapping[str, str]:
        """Convert the config to a Mapping that can be stored in a ConfigEntry."""
        return {
            _MeterReadingConfigField.ID: self.id,
            _MeterReadingConfigField.SENSOR_DEVICE_CLASS: self.sensor_device_class.value,
            _MeterReadingConfigField.UNIT_OF_MEASUREMENT: self.unit_of_measurement,
            _MeterReadingConfigField.CURRENCY: self.currency,
        }

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, str], initial_usage_point: model.UsagePoint | None
    ) -> MeterReadingConfig:
        """Create a config instance from a ConfigEntry Mapping."""
        meter_reading_id = data[_MeterReadingConfigField.ID]
        initial_meter_reading = None
        if initial_usage_point is not None:
            initial_meter_reading = initial_usage_point.get_meter_reading_by_id(
                meter_reading_id
            )
        return MeterReadingConfig(
            id=meter_reading_id,
            sensor_device_class=sensor.SensorDeviceClass(
                data[_MeterReadingConfigField.SENSOR_DEVICE_CLASS]
            ),
            unit_of_measurement=data[_MeterReadingConfigField.UNIT_OF_MEASUREMENT],
            currency=data[_MeterReadingConfigField.CURRENCY],
            initial_meter_reading=initial_meter_reading,
        )

    @classmethod
    def from_model(
        cls, usage_point: model.UsagePoint, meter_reading: model.MeterReading
    ) -> MeterReadingConfig:
        """Create a config instance from a MeterReading."""
        return MeterReadingConfig(
            id=meter_reading.id,
            sensor_device_class=usage_point.sensor_device_class,
            unit_of_measurement=meter_reading.reading_type.unit_of_measurement,
            currency=meter_reading.reading_type.currency,
            initial_meter_reading=None,
        )


class _ComponentConfigField:
    NAME: Final = "name"
    XML: Final = "xml"
    UNIQUE_ID: Final = "usage_point_id"
    METER_READING_CONFIGS: Final = "meter_reading_configs"
    ENERGY_UNIT: Final = "energy_unit"
    COST_CURRENCY: Final = "cost_currency"


@final
@dataclasses.dataclass(frozen=True)
class ComponentConfig:
    """A dataclass containing the configs for the one instance of the component."""

    name: str
    unique_id: str
    meter_reading_configs: list[MeterReadingConfig]
    initial_usage_point: model.UsagePoint | None

    def set_side_channels(self, hass: HomeAssistant):
        """Set the config's side channel values.

        This is used to pass data from the config to the component setup code
        without serializing it through ConfigEntry data.
        """
        if self.initial_usage_point is not None:
            side_channel.get(hass).set(
                self.unique_id,
                side_channel.INITIAL_IMPORT_USAGE_POINT,
                self.initial_usage_point,
            )

    def to_mapping(self) -> Mapping[str, Any]:
        """Convert the config to a Mapping that can be stored in a ConfigEntry."""
        return {
            _ComponentConfigField.NAME: self.name,
            _ComponentConfigField.UNIQUE_ID: self.unique_id,
            _ComponentConfigField.METER_READING_CONFIGS: [
                meter_reading_config.to_mapping()
                for meter_reading_config in self.meter_reading_configs
            ],
        }

    @classmethod
    def make_config_entry_step_schema(
        cls, user_input: dict[str, Any] | None
    ) -> vol.Schema:
        """Create a step schema for the config."""
        if user_input is None:
            user_input = {
                _ComponentConfigField.NAME: "Home",
            }
        return vol.Schema(
            {
                vol.Required(
                    _ComponentConfigField.NAME,
                    default=user_input.get(_ComponentConfigField.NAME),
                ): str,
                vol.Required(
                    _ComponentConfigField.XML,
                    default=user_input.get(_ComponentConfigField.XML),
                ): selector.TextSelector( # type: ignore
                    selector.TextSelectorConfig(
                        multiline=True,
                    )
                ),
            }
        )

    @classmethod
    def from_mapping(cls, user_input: Mapping[str, Any]) -> ComponentConfig:
        """Create a config instance from a ConfigEntry Mapping.

        The mapping should conform to the expected schema from
        `make_config_entry_step_schema`.
        """
        try:
            usage_points = espi.parse_xml(user_input[_ComponentConfigField.XML])
        except espi.EspiXmlParseError as ex:
            raise InvalidUserInputError(
                {_ComponentConfigField.XML: "invalid_espi_xml"}
            ) from ex

        if not usage_points:
            raise InvalidUserInputError(
                {_ComponentConfigField.XML: "no_usage_points_found"}
            )
        if len(usage_points) > 1:
            raise InvalidUserInputError(
                {_ComponentConfigField.XML: "multiple_usage_points_found"}
            )

        usage_point = usage_points[0]
        meter_reading_configs = [
            MeterReadingConfig.from_model(usage_point, meter_reading)
            for meter_reading in usage_point.meter_readings
        ]

        return ComponentConfig(
            name=user_input[_ComponentConfigField.NAME],
            unique_id=usage_point.id,
            meter_reading_configs=meter_reading_configs,
            initial_usage_point=usage_point,
        )

    @classmethod
    def from_entry(cls, hass: HomeAssistant, entry: ConfigEntry) -> ComponentConfig:
        """Create a config from a ConfigEntry.

        This method will re-constitute side-channel configs.
        """
        unique_id = entry.unique_id
        assert unique_id is not None

        initial_usage_point = side_channel.get(hass).get(
            unique_id, side_channel.INITIAL_IMPORT_USAGE_POINT
        )
        return ComponentConfig(
            name=entry.data[_ComponentConfigField.NAME],
            unique_id=entry.data[_ComponentConfigField.UNIQUE_ID],
            meter_reading_configs=[
                MeterReadingConfig.from_mapping(
                    meter_reading_config, initial_usage_point
                )
                for meter_reading_config in entry.data[
                    _ComponentConfigField.METER_READING_CONFIGS
                ]
            ],
            initial_usage_point=initial_usage_point,
        )
