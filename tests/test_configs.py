"""Unit tests for configs.py in green_button integration."""
import pytest
from homeassistant.components import sensor
from custom_components.green_button.configs import (
    InvalidUserInputError,
    MeterReadingConfig,
    _MeterReadingConfigField,
)


def test_invalid_user_input_error():
    errors = {"field": "invalid value"}
    err = InvalidUserInputError(errors)
    assert isinstance(err, ValueError)
    assert err.errors == errors
    assert "Invalid user input" in str(err)


def test_meter_reading_config_fields():
    # Test StrEnum values
    assert _MeterReadingConfigField.ID == "id"
    assert _MeterReadingConfigField.SENSOR_DEVICE_CLASS == "sensor_device_class"
    assert _MeterReadingConfigField.UNIT_OF_MEASUREMENT == "unit_of_measurement"
    assert _MeterReadingConfigField.CURRENCY == "currency"


def test_meter_reading_config_dataclass():
    # Create a config and check fields
    # Create a dummy ReadingType and MeterReading for initial_meter_reading
    from custom_components.green_button.model import ReadingType, MeterReading
    reading_type = ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )
    meter_reading = MeterReading(
        id="meter1",
        reading_type=reading_type,
        interval_blocks=[]
    )
    config = MeterReadingConfig(
        id="meter1",
        sensor_device_class=sensor.SensorDeviceClass.ENERGY,
        unit_of_measurement="kWh",
        currency="CAD",
        initial_meter_reading=meter_reading
    )
    assert config.id == "meter1"
    assert config.sensor_device_class == sensor.SensorDeviceClass.ENERGY
    assert config.unit_of_measurement == "kWh"
    assert config.currency == "CAD"
    assert config.initial_meter_reading == meter_reading


def test_meter_reading_config_to_mapping():
    from homeassistant.components import sensor
    from custom_components.green_button.model import ReadingType, MeterReading
    config = MeterReadingConfig(
        id="meter1",
        sensor_device_class=sensor.SensorDeviceClass.ENERGY,
        unit_of_measurement="kWh",
        currency="CAD",
        initial_meter_reading=None
    )
    mapping = config.to_mapping()
    assert mapping["id"] == "meter1"
    assert mapping["sensor_device_class"] == sensor.SensorDeviceClass.ENERGY.value
    assert mapping["unit_of_measurement"] == "kWh"
    assert mapping["currency"] == "CAD"


def test_meter_reading_config_from_mapping():
    from homeassistant.components import sensor
    from custom_components.green_button.model import UsagePoint
    mapping = {
        "id": "meter1",
        "sensor_device_class": sensor.SensorDeviceClass.ENERGY.value,
        "unit_of_measurement": "kWh",
        "currency": "CAD"
    }
    usage_point = UsagePoint.default_usage_point()
    config = MeterReadingConfig.from_mapping(mapping, usage_point)
    assert config.id == "meter1"
    assert config.sensor_device_class == sensor.SensorDeviceClass.ENERGY
    assert config.unit_of_measurement == "kWh"
    assert config.currency == "CAD"
    assert isinstance(config, MeterReadingConfig)


def test_meter_reading_config_from_model():
    from homeassistant.components import sensor
    from custom_components.green_button.model import UsagePoint
    usage_point = UsagePoint.default_usage_point()
    meter_reading = usage_point.meter_readings[0]
    config = MeterReadingConfig.from_model(usage_point, meter_reading)
    assert config.id == meter_reading.id
    assert config.sensor_device_class == sensor.SensorDeviceClass.ENERGY
    assert config.unit_of_measurement == meter_reading.reading_type.unit_of_measurement
    assert config.currency == meter_reading.reading_type.currency
    assert config.initial_meter_reading is None
