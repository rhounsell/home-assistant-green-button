"""Recorder ownership regression tests.

Run with core fixtures:
PYTHONPATH=.:config uv run --no-sync pytest -p tests.conftest \
    config/custom_components/green_button/tests/test_statistics.py
"""

# ruff: noqa: TID251

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from custom_components.green_button import model, statistics
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator
from custom_components.green_button.sensor import (
    GreenButtonCostSensor,
    GreenButtonGasCostSensor,
    GreenButtonGasSensor,
    GreenButtonSensor,
    GreenButtonStatisticsSensor,
)
from custom_components.green_button.statistic_ids import statistic_id_from_unique_id
from freezegun.api import FrozenDateTimeFactory
import pytest

from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType
from homeassistant.components.sensor import DATA_COMPONENT
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry
from tests.components.recorder.common import (
    async_wait_recording_done,
    do_adhoc_statistics,
    statistics_during_period,
)

HISTORICAL = datetime(2026, 7, 1, tzinfo=UTC)


async def _energy(hass: HomeAssistant, entity: GreenButtonStatisticsSensor) -> None:
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    reading = model.IntervalReading(
        reading_type, 125, HISTORICAL, timedelta(hours=1), 1000
    )
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, HISTORICAL, timedelta(hours=1), [reading]
            )
        ],
    )
    await statistics.update_statistics(
        hass, entity, statistics.DefaultDataExtractor(), meter
    )


async def _cost(hass: HomeAssistant, entity: GreenButtonStatisticsSensor) -> None:
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    reading = model.IntervalReading(
        reading_type, 125, HISTORICAL, timedelta(hours=1), 1000
    )
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, HISTORICAL, timedelta(hours=1), [reading]
            )
        ],
    )
    await statistics.update_cost_statistics(
        hass, entity, statistics.CostDataExtractor(-3), meter
    )


async def _gas(hass: HomeAssistant, entity: GreenButtonStatisticsSensor) -> None:
    reading_type = model.ReadingType("type", 7, "CAD", -3, "m³", 86400)
    reading = model.IntervalReading(
        reading_type, 0, HISTORICAL, timedelta(days=1), 1000
    )
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, HISTORICAL, timedelta(days=1), [reading]
            )
        ],
    )
    await statistics.update_gas_statistics(hass, entity, meter)


async def _gas_cost(hass: HomeAssistant, entity: GreenButtonStatisticsSensor) -> None:
    summary = model.UsageSummary(
        "summary", HISTORICAL, timedelta(days=30), 2.5, "CAD", 1
    )
    await statistics.update_gas_cost_statistics(
        hass, entity, None, [summary], "monthly_increment", -3
    )


@pytest.mark.usefixtures("recorder_mock")
@pytest.mark.parametrize(
    ("sensor_cls", "importer", "unit_class", "total"),
    [
        pytest.param(GreenButtonSensor, _energy, "energy", 1.0, id="electricity"),
        pytest.param(GreenButtonCostSensor, _cost, None, 0.125, id="electricity-cost"),
        pytest.param(GreenButtonGasSensor, _gas, "volume", 1.0, id="gas"),
        pytest.param(GreenButtonGasCostSensor, _gas_cost, None, 2.5, id="gas-cost"),
    ],
)
async def test_external_statistics_own_history(
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    sensor_cls: type[GreenButtonStatisticsSensor],
    importer: Callable[[HomeAssistant, GreenButtonStatisticsSensor], Awaitable[None]],
    unit_class: str | None,
    total: float,
) -> None:
    """Imports preserve legacy history and sensor compilation cannot extend them."""
    now = datetime(2026, 9, 6, 12, tzinfo=UTC)
    freezer.move_to(now)
    assert await async_setup_component(hass, "sensor", {})
    entry = MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    coordinator = GreenButtonCoordinator(hass, entry)
    coordinator.async_set_updated_data({"usage_points": []})
    entity = sensor_cls(coordinator, "meter")
    entity.entity_id = "sensor.imported_display"
    await hass.data[DATA_COMPONENT].async_add_entities([entity])

    # Retain legacy history even if its source XML is no longer available.
    legacy = StatisticData(start=HISTORICAL - timedelta(days=30), state=17, sum=17)
    recorder_statistics.async_import_statistics(
        hass,
        {
            "statistic_id": entity.entity_id,
            "source": "recorder",
            "name": "Legacy",
            "unit_of_measurement": entity.native_unit_of_measurement,
            "unit_class": unit_class,
            "has_sum": True,
            "mean_type": StatisticMeanType.NONE,
        },
        [legacy],
    )
    await async_wait_recording_done(hass)
    legacy_before = statistics_during_period(
        hass, HISTORICAL - timedelta(days=31), statistic_ids={entity.entity_id}
    )
    await importer(hass, entity)
    await async_wait_recording_done(hass)

    external_id = entity.long_term_statistics_id
    before = statistics_during_period(hass, HISTORICAL, statistic_ids={external_id})
    assert before[external_id][-1]["sum"] == pytest.approx(total)
    assert (
        recorder_statistics.get_metadata(hass, statistic_ids={external_id})[
            external_id
        ][1]["source"]
        == DOMAIN
    )
    assert (
        recorder_statistics.get_metadata(hass, statistic_ids={external_id})[
            external_id
        ][1]["unit_class"]
        == unit_class
    )
    state = hass.states.get(entity.entity_id)
    assert state is not None
    assert "state_class" not in state.attributes
    assert state.attributes["statistic_id"] == external_id

    # Compile a complete hour after a large display-state change.
    hass.states.async_set(entity.entity_id, 999999, dict(state.attributes))
    await async_wait_recording_done(hass)
    freezer.move_to(now + timedelta(hours=1))
    for minute in range(0, 60, 5):
        do_adhoc_statistics(hass, start=now + timedelta(minutes=minute))
    await async_wait_recording_done(hass)
    assert (
        statistics_during_period(hass, HISTORICAL, statistic_ids={external_id})
        == before
    )
    assert (
        statistics_during_period(
            hass, HISTORICAL - timedelta(days=31), statistic_ids={entity.entity_id}
        )
        == legacy_before
    )
    assert (
        statistics_during_period(
            hass, now, period="5minute", statistic_ids={entity.entity_id, external_id}
        )
        == {}
    )

    # A re-created or renamed display entity resolves to the same external series.
    replacement = sensor_cls(GreenButtonCoordinator(hass, entry), "meter")
    replacement.entity_id = "sensor.renamed_display"
    assert replacement.long_term_statistics_id == external_id
    await importer(hass, replacement)
    await async_wait_recording_done(hass)
    after = statistics_during_period(hass, HISTORICAL, statistic_ids={external_id})
    assert len(after[external_id]) == len(before[external_id])
    assert after[external_id][-1]["sum"] == pytest.approx(total)


def test_external_ids_do_not_collide() -> None:
    """Keep distinct case and punctuation in stable identity hashing."""
    ids = {
        statistic_id_from_unique_id(value)
        for value in (
            "entry_A",
            "entry_a",
            "entry-a",
            "entry_A_cost",
            "entry_A_gas",
            "entry_A_gas_cost",
        )
    }
    assert len(ids) == 6
    assert all(recorder_statistics.valid_statistic_id(value) for value in ids)
