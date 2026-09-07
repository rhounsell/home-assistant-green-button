"""Regression tests for merging Green Button source documents."""

# ruff: noqa: SLF001, TID251

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from custom_components.green_button import model
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator
from custom_components.green_button.xml_storage import async_get_xml_storage
import pytest

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


def _hourly_block(
    reading_type: model.ReadingType,
    block_id: str,
    start: datetime,
    values: list[int],
) -> model.IntervalBlock:
    """Create an hourly block with a value for every reading."""
    readings = [
        model.IntervalReading(
            reading_type,
            index,
            start + timedelta(hours=index),
            timedelta(hours=1),
            value,
        )
        for index, value in enumerate(values)
    ]
    return model.IntervalBlock(
        block_id,
        reading_type,
        start,
        timedelta(hours=len(readings)),
        readings,
    )


def _merge_meter_readings(
    hass: HomeAssistant,
    existing_meter: model.MeterReading,
    new_meter: model.MeterReading,
) -> model.MeterReading:
    """Merge one meter reading into another through the coordinator."""
    coordinator = GreenButtonCoordinator(
        hass, MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    )
    coordinator.usage_points = [
        model.UsagePoint("usage-point", SensorDeviceClass.ENERGY, [existing_meter])
    ]
    coordinator._merge_usage_points(
        [model.UsagePoint("usage-point", SensorDeviceClass.ENERGY, [new_meter])]
    )
    return coordinator.usage_points[0].meter_readings[0]


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


def test_merge_uses_new_values_for_corrected_intervals(
    hass: HomeAssistant,
) -> None:
    """A later document replaces both usage and cost for the same interval."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    existing_meter = model.MeterReading(
        "meter",
        reading_type,
        [_hourly_block(reading_type, "existing", start, [1000, 1000])],
    )
    corrected_reading = model.IntervalReading(
        reading_type, 200, start, timedelta(hours=1), 2000
    )
    corrected_meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "corrected",
                reading_type,
                start,
                timedelta(hours=1),
                [corrected_reading],
            )
        ],
    )

    merged = _merge_meter_readings(hass, existing_meter, corrected_meter)
    readings = [
        reading
        for block in merged.interval_blocks
        for reading in block.interval_readings
    ]

    assert [(reading.value, reading.cost) for reading in readings] == [
        (2000, 200),
        (1000, 1),
    ]


@pytest.mark.parametrize(
    ("initial_values", "new_values"),
    [
        pytest.param([1000] * 23, [1000] * 24, id="short-then-long"),
        pytest.param([1000] * 24, [1000] * 23, id="long-then-short"),
    ],
)
def test_merge_reconciles_differently_sized_blocks(
    hass: HomeAssistant,
    initial_values: list[int],
    new_values: list[int],
) -> None:
    """Block packaging cannot duplicate identical hourly intervals."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 8, 9, tzinfo=UTC)
    existing_meter = model.MeterReading(
        "meter",
        reading_type,
        [_hourly_block(reading_type, "initial", start, initial_values)],
    )
    new_meter = model.MeterReading(
        "meter",
        reading_type,
        [_hourly_block(reading_type, "new", start, new_values)],
    )

    merged = _merge_meter_readings(hass, existing_meter, new_meter)
    readings = [
        reading
        for block in merged.interval_blocks
        for reading in block.interval_readings
    ]

    assert len(readings) == 24
    assert {reading.start for reading in readings} == {
        start + timedelta(hours=index) for index in range(24)
    }


def test_merge_rejects_ambiguous_interval_overlap(
    hass: HomeAssistant,
) -> None:
    """A different-length overlap preserves prior coverage without duplication."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    existing_reading = model.IntervalReading(
        reading_type, 0, start, timedelta(hours=2), 2000
    )
    existing_meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "existing",
                reading_type,
                start,
                timedelta(hours=2),
                [existing_reading],
            )
        ],
    )
    new_meter = model.MeterReading(
        "meter",
        reading_type,
        [_interval_block(reading_type, "ambiguous", start)],
    )

    merged = _merge_meter_readings(hass, existing_meter, new_meter)
    readings = [
        reading
        for block in merged.interval_blocks
        for reading in block.interval_readings
    ]

    assert readings == [existing_reading]


def test_merge_rejects_ambiguous_usage_summary_overlap(
    hass: HomeAssistant,
) -> None:
    """A later overlapping statement cannot allocate the same gas days twice."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    coordinator = GreenButtonCoordinator(hass, entry)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    existing = model.UsageSummary(
        "existing", start, timedelta(days=30), 20.0, "CAD", 100.0
    )
    overlapping = model.UsageSummary(
        "overlapping",
        start + timedelta(days=15),
        timedelta(days=30),
        20.0,
        "CAD",
        100.0,
    )
    coordinator.usage_points = [
        model.UsagePoint("gas", SensorDeviceClass.GAS, [], [existing])
    ]

    coordinator._merge_usage_points(
        [model.UsagePoint("gas", SensorDeviceClass.GAS, [], [overlapping])]
    )

    assert list(coordinator.usage_points[0].usage_summaries) == [existing]


async def test_stored_reconstruction_matches_normal_canonical_merge(
    hass: HomeAssistant,
) -> None:
    """Recalculation rebuilds repeated and corrected XML through normal merging."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    entry.add_to_hass(hass)
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml("<first />", "electricity")
    await storage.async_add_xml("<correction />", "electricity")
    await storage.async_add_xml("<correction />", "electricity")
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    initial = model.UsagePoint(
        "electricity",
        SensorDeviceClass.ENERGY,
        [
            model.MeterReading(
                "meter",
                reading_type,
                [_hourly_block(reading_type, "initial", start, [1000, 1000])],
            )
        ],
    )
    correction = model.UsagePoint(
        "electricity",
        SensorDeviceClass.ENERGY,
        [
            model.MeterReading(
                "meter",
                reading_type,
                [_hourly_block(reading_type, "correction", start, [2000, 1000])],
            )
        ],
    )
    normal = GreenButtonCoordinator(hass, entry)
    normal._merge_usage_points([initial])
    normal._merge_usage_points([correction])
    reconstructed = GreenButtonCoordinator(hass, entry)

    with patch(
        "custom_components.green_button.coordinator.espi.parse_xml",
        side_effect=[[initial], [correction], [correction]],
    ):
        rebuilt = await reconstructed.async_reconstruct_stored_usage_points()

    assert rebuilt == normal.usage_points
