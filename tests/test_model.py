"""Unit tests for model.py in green_button integration."""
import datetime
from custom_components.green_button.model import (
    IntervalReading,
    IntervalBlock,
    ReadingType,
    MeterReading,
    UsagePoint,
)


def test_interval_reading_properties():
    reading_type = ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0)
    duration = datetime.timedelta(hours=1)
    reading = IntervalReading(
        reading_type=reading_type,
        cost=100,
        start=start,
        duration=duration,
        value=42,
        powerOfTenMultiplier=-3,
    )
    assert reading.end == start + duration
    assert reading.value == 42


def test_interval_block_ordering():
    reading_type = ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0)
    duration = datetime.timedelta(hours=1)
    readings = [
        IntervalReading(reading_type, 100, start, duration, 42),
        IntervalReading(reading_type, 200, start + duration, duration, 43),
    ]
    block = IntervalBlock(
        id="block1",
        reading_type=reading_type,
        start=start,
        duration=duration * 2,
        interval_readings=readings[::-1],  # Unsorted
    )
    assert block.interval_readings[0].value == 42
    newest = block.get_newest_interval_reading()
    assert newest is not None
    assert newest.value == 43
    assert block.end == start + duration * 2


def test_meter_reading_get_newest():
    reading_type = ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )
    start = datetime.datetime(2024, 1, 1, 0, 0)
    duration = datetime.timedelta(hours=1)
    readings = [
        IntervalReading(reading_type, 100, start, duration, 42),
        IntervalReading(reading_type, 200, start + duration, duration, 43),
    ]
    block = IntervalBlock(
        id="block1",
        reading_type=reading_type,
        start=start,
        duration=duration * 2,
        interval_readings=readings,
    )
    meter = MeterReading(
        id="meter1",
        reading_type=reading_type,
        interval_blocks=[block],
    )
    newest = meter.get_newest_interval_reading()
    assert newest is not None
    assert newest.value == 43


def test_usage_point_get_meter_reading_by_id():
    reading_type = ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )
    meter = MeterReading(
        id="meter1",
        reading_type=reading_type,
        interval_blocks=[],
    )
    usage_point = UsagePoint(
        id="up1",
        sensor_device_class=None,
        meter_readings=[meter],
    )
    assert usage_point.get_meter_reading_by_id("meter1") == meter
    assert usage_point.get_meter_reading_by_id("notfound") is None


def test_usage_point_default_usage_point():
    up = UsagePoint.default_usage_point()
    assert up.id == "default_usage_point"
    meter_readings = list(up.meter_readings)
    assert len(meter_readings) == 1
    assert meter_readings[0].id == "default_energy_meter"
