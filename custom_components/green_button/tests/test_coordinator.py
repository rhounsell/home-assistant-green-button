"""Regression tests for merging Green Button source documents."""

# ruff: noqa: SLF001, TID251

from datetime import UTC, datetime, timedelta

from custom_components.green_button import model
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry


def _interval_block(
    reading_type: model.ReadingType,
    block_id: str,
    start: datetime,
) -> model.IntervalBlock:
    reading = model.IntervalReading(reading_type, 0, start, timedelta(hours=1), 1000)
    return model.IntervalBlock(
        block_id, reading_type, start, timedelta(hours=1), [reading]
    )


def test_merge_keeps_new_meter_readings_and_usage_summaries(
    hass: HomeAssistant,
) -> None:
    """A source document adding both collections does not lose either one."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    coordinator = GreenButtonCoordinator(hass, entry)
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    first_start = datetime(2026, 7, 1, tzinfo=UTC)
    second_start = first_start + timedelta(hours=1)
    existing_meter = model.MeterReading(
        "meter", reading_type, [_interval_block(reading_type, "first", first_start)]
    )
    coordinator.usage_points = [
        model.UsagePoint("usage-point", SensorDeviceClass.ENERGY, [existing_meter])
    ]

    added_meter = model.MeterReading(
        "meter", reading_type, [_interval_block(reading_type, "second", second_start)]
    )
    summary = model.UsageSummary(
        "summary", first_start, timedelta(days=30), 20.0, "CAD", 100.0
    )
    coordinator._merge_usage_points(
        [
            model.UsagePoint(
                "usage-point", SensorDeviceClass.ENERGY, [added_meter], [summary]
            )
        ]
    )

    merged = coordinator.usage_points[0]
    assert [block.id for block in merged.meter_readings[0].interval_blocks] == [
        "first",
        "second",
    ]
    assert list(merged.usage_summaries) == [summary]
