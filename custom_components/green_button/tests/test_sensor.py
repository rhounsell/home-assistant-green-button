"""Regression tests for Green Button stream discovery and identity."""

# ruff: noqa: SLF001, TID251

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from custom_components.green_button import model, sensor, statistics
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from tests.common import MockConfigEntry


def _usage_point(
    usage_point_id: str, meter_reading_id: str, value: int
) -> model.UsagePoint:
    """Build one electricity stream with a caller-controlled full provider ID."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 9, 1, tzinfo=UTC)
    reading = model.IntervalReading(reading_type, 0, start, timedelta(hours=1), value)
    meter_reading = model.MeterReading(
        meter_reading_id,
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, start, timedelta(hours=1), [reading]
            )
        ],
    )
    return model.UsagePoint(usage_point_id, SensorDeviceClass.ENERGY, [meter_reading])


def _gas_usage_point() -> model.UsagePoint:
    """Build one gas stream with both daily intervals and a billing summary."""
    usage_point = _usage_point("/UsagePoint/gas", "/MeterReading/1", 1000)
    return model.UsagePoint(
        usage_point.id,
        SensorDeviceClass.GAS,
        usage_point.meter_readings,
        [
            model.UsageSummary(
                "summary", datetime(2026, 9, 1, tzinfo=UTC), timedelta(days=1), 1, "CAD"
            )
        ],
    )


async def test_stream_entities_are_added_once_per_discovery(
    hass: HomeAssistant,
) -> None:
    """Same path suffixes and arrival order cannot hide or replace a stream."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    entry.add_to_hass(hass)
    coordinator = GreenButtonCoordinator(hass, entry)
    first = _usage_point("/UsagePoint/alpha", "/MeterReading/1", 1000)
    second = _usage_point("/UsagePoint/beta", "/MeterReading/1", 2000)
    coordinator.usage_points = [second, first]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"coordinator": coordinator}
    added: list[SensorEntity] = []
    add_calls: list[list[SensorEntity]] = []

    def async_add_entities(entities: list[SensorEntity]) -> None:
        added.extend(entities)
        add_calls.append(entities)

    await sensor.async_setup_entry(hass, entry, async_add_entities)

    assert len(added) == 4
    assert [len(entities) for entities in add_calls] == [4]
    assert len({entity.unique_id for entity in added}) == 4
    assert {entity._meter_reading_id for entity in added} == {"/MeterReading/1"}
    assert {entity._usage_point_id for entity in added} == {
        "/UsagePoint/alpha",
        "/UsagePoint/beta",
    }

    first_discovery = {entity.unique_id for entity in added}
    coordinator.usage_points = [first, second]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    assert {entity.unique_id for entity in added} == first_discovery
    assert [len(entities) for entities in add_calls] == [4]

    third = _usage_point("/UsagePoint/gamma", "/MeterReading/1", 3000)
    coordinator.usage_points = [third, second, first]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    assert len(added) == 6
    assert len({entity.unique_id for entity in added}) == 6
    assert [len(entities) for entities in add_calls] == [4, 2]


async def test_unambiguous_legacy_stream_migrates_entity_and_external_series(
    hass: HomeAssistant,
) -> None:
    """An old suffix-based entity keeps its registry entry and external history."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    entry.add_to_hass(hass)
    coordinator = GreenButtonCoordinator(hass, entry)
    usage_point = _usage_point("/UsagePoint/alpha", "/MeterReading/1", 1000)
    coordinator.usage_points = [usage_point]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"coordinator": coordinator}
    meter_reading = usage_point.meter_readings[0]
    legacy_unique_id = sensor.GreenButtonSensor(coordinator, meter_reading.id).unique_id
    registry = er.async_get(hass)
    legacy_entry = registry.async_get_or_create(
        "sensor", DOMAIN, legacy_unique_id, config_entry=entry
    )
    added: list[SensorEntity] = []

    def async_add_entities(entities: list[SensorEntity]) -> None:
        added.extend(entities)

    with patch.object(statistics, "rename_external_statistic") as rename_statistics:
        await sensor.async_setup_entry(hass, entry, async_add_entities)

    usage_sensor = next(
        entity for entity in added if isinstance(entity, sensor.GreenButtonSensor)
    )
    assert (
        registry.async_get(legacy_entry.entity_id).unique_id == usage_sensor.unique_id
    )
    rename_statistics.assert_any_call(hass, legacy_unique_id, usage_sensor.unique_id)


async def test_gas_allocation_mode_change_keeps_single_stream_identity(
    hass: HomeAssistant,
) -> None:
    """Daily and monthly allocation use the same entity pair for one gas stream."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test-entry",
        options={"gas_usage_allocation": "daily_readings"},
    )
    entry.add_to_hass(hass)
    coordinator = GreenButtonCoordinator(hass, entry)
    usage_point = _gas_usage_point()
    coordinator.usage_points = [usage_point]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"coordinator": coordinator}
    added: list[SensorEntity] = []

    def async_add_entities(entities: list[SensorEntity]) -> None:
        added.extend(entities)

    await sensor.async_setup_entry(hass, entry, async_add_entities)
    daily_ids = {entity.unique_id for entity in added}
    assert {entity._meter_reading_id for entity in added} == {"/MeterReading/1"}

    hass.config_entries.async_update_entry(
        entry, options={"gas_usage_allocation": "monthly_increment"}
    )
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})

    assert len(added) == 2
    assert {entity.unique_id for entity in added} == daily_ids
    assert {entity._meter_reading_id for entity in added} == {"/MeterReading/1"}
