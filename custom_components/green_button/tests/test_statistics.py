"""Recorder ownership regression tests.

Run with core fixtures:
PYTHONPATH=.:config uv run --no-sync pytest -p tests.conftest \
    config/custom_components/green_button/tests/test_statistics.py
"""

# ruff: noqa: SLF001, TID251

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
import decimal
from unittest.mock import AsyncMock, Mock, patch

from custom_components.green_button import (
    allocation,
    model,
    scaling,
    sensor,
    statistics,
)
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
from homeassistant.components.sensor import DATA_COMPONENT, SensorDeviceClass
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry
from tests.components.recorder.common import (
    async_wait_recording_done,
    do_adhoc_statistics,
    statistics_during_period,
)

HISTORICAL = datetime(2026, 7, 1, tzinfo=UTC)


class _StatisticsEntity:
    """Minimal entity used by pure statistics-generation tests."""

    entity_id = "sensor.imported_display"
    long_term_statistics_id = "green_button:test"
    name = "Imported display"
    native_unit_of_measurement = "kWh"


def _partial_hour_meter(
    include_completion: bool, power_of_ten_multiplier: int | None = 0
) -> model.MeterReading:
    """Create a 2.5-hour interval and an optional trailing half-hour."""
    reading_type = model.ReadingType(
        "type", 1, "CAD", power_of_ten_multiplier, "Wh", 3600
    )
    long_reading = model.IntervalReading(
        reading_type, 250, HISTORICAL, timedelta(hours=2, minutes=30), 2500
    )
    blocks = [
        model.IntervalBlock(
            "long",
            reading_type,
            HISTORICAL,
            timedelta(hours=2, minutes=30),
            [long_reading],
        )
    ]
    if include_completion:
        completion_start = HISTORICAL + timedelta(hours=2, minutes=30)
        completion = model.IntervalReading(
            reading_type, 50, completion_start, timedelta(minutes=30), 500
        )
        blocks.append(
            model.IntervalBlock(
                "completion",
                reading_type,
                completion_start,
                timedelta(minutes=30),
                [completion],
            )
        )
    return model.MeterReading("meter", reading_type, blocks)


def _assert_hourly_statistics(
    records: list[StatisticData],
    expected_hours: int,
) -> None:
    """Assert complete hours receive one unit and a running cumulative sum."""
    assert [record["start"] for record in records] == [
        HISTORICAL + timedelta(hours=hour) for hour in range(expected_hours)
    ]
    assert [record["state"] for record in records] == pytest.approx(
        [1.0] * expected_hours
    )
    assert [record["sum"] for record in records] == pytest.approx(
        list(range(1, expected_hours + 1))
    )


def test_source_multiplier_precedes_configured_cost_fallback() -> None:
    """A declared source multiplier wins; the fallback covers only its absence."""
    start = HISTORICAL
    declared_type = model.ReadingType("type", 1, "CAD", -3, "Wh", 3600)
    missing_type = model.ReadingType("type", 1, "CAD", None, "Wh", 3600)
    declared = model.IntervalReading(declared_type, 800, start, timedelta(hours=1), 1)
    missing = model.IntervalReading(missing_type, 800, start, timedelta(hours=1), 1)
    declared_summary = model.UsageSummary(
        "declared",
        start,
        timedelta(days=1),
        800,
        "CAD",
        consumption_m3=800,
        power_of_ten_multiplier=-3,
        consumption_power_of_ten_multiplier=-3,
    )
    missing_summary = model.UsageSummary(
        "missing", start, timedelta(days=1), 800, "CAD"
    )

    assert float(scaling.interval_cost(declared, -5)) == pytest.approx(0.8)
    assert float(scaling.interval_cost(missing, -3)) == pytest.approx(0.8)
    assert float(scaling.usage_summary_cost(declared_summary, -5)) == pytest.approx(0.8)
    assert float(scaling.usage_summary_cost(missing_summary, -3)) == pytest.approx(0.8)
    assert float(scaling.usage_summary_consumption(declared_summary)) == pytest.approx(
        0.8
    )


@pytest.mark.parametrize(
    ("time_zone", "start", "duration", "value", "expected"),
    [
        pytest.param(
            "America/Toronto",
            datetime(2026, 1, 2, 5, tzinfo=UTC),
            timedelta(days=1),
            24,
            [("2026-01-02", 24.0)],
            id="negative-utc-offset",
        ),
        pytest.param(
            "Asia/Tokyo",
            datetime(2026, 1, 1, 15, tzinfo=UTC),
            timedelta(days=1),
            24,
            [("2026-01-02", 24.0)],
            id="positive-utc-offset",
        ),
        pytest.param(
            "America/Toronto",
            datetime(2026, 1, 2, 4, tzinfo=UTC),
            timedelta(hours=2),
            2,
            [("2026-01-01", 1.0), ("2026-01-02", 1.0)],
            id="local-midnight-overlap",
        ),
        pytest.param(
            "America/Toronto",
            datetime(2026, 2, 1, 4, tzinfo=UTC),
            timedelta(hours=2),
            2,
            [("2026-01-31", 1.0), ("2026-02-01", 1.0)],
            id="month-boundary",
        ),
        pytest.param(
            "America/Toronto",
            datetime(2026, 3, 8, 5, tzinfo=UTC),
            timedelta(hours=23),
            23,
            [("2026-03-08", 23.0)],
            id="dst-short-day",
        ),
    ],
)
async def test_gas_daily_totals_use_local_dates_and_interval_overlap(
    hass: HomeAssistant,
    time_zone: str,
    start: datetime,
    duration: timedelta,
    value: int,
    expected: list[tuple[str, float]],
) -> None:
    """Gas intervals use local days and their physical duration on each day."""
    await hass.config.async_set_time_zone(time_zone)
    reading_type = model.ReadingType("type", 7, "CAD", 0, "m³", 3600)
    reading = model.IntervalReading(reading_type, 0, start, duration, value)

    totals = allocation.local_day_values(
        [reading],
        scaling.interval_value,
        statistics._billing_timezone(),
    )

    assert [day.isoformat() for day in totals] == [day for day, _ in expected]
    assert list(totals.values()) == pytest.approx([total for _, total in expected])


async def test_gas_cost_allocation_clips_to_billing_period_boundaries(
    hass: HomeAssistant,
) -> None:
    """A partial local day receives only its physical share of the billed cost."""
    await hass.config.async_set_time_zone("America/Toronto")
    reading_type = model.ReadingType("type", 7, "CAD", 0, "m³", 3600)
    reading = model.IntervalReading(
        reading_type, 0, datetime(2026, 1, 2, 4, tzinfo=UTC), timedelta(hours=2), 2
    )
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, reading.start, reading.duration, [reading]
            )
        ],
    )
    summary = model.UsageSummary(
        "summary",
        datetime(2026, 1, 2, 4, 30, tzinfo=UTC),
        timedelta(hours=1),
        100,
        "CAD",
        power_of_ten_multiplier=0,
    )

    with patch.object(
        statistics, "_async_replace_statistics", new_callable=AsyncMock
    ) as replace_statistics:
        await statistics._async_update_gas_cost_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            meter,
            [summary],
            merge_with_existing=False,
        )

    records = replace_statistics.await_args.args[2]
    assert [record["start"].isoformat() for record in records] == [
        "2026-01-01T00:00:00-05:00",
        "2026-01-02T00:00:00-05:00",
    ]
    assert [record["state"] for record in records] == pytest.approx([50.0, 50.0])


async def test_summary_only_gas_cost_uses_local_billing_end_date(
    hass: HomeAssistant,
) -> None:
    """Summary-only monthly cost records use the local billing end date."""
    await hass.config.async_set_time_zone("America/Toronto")
    summary = model.UsageSummary(
        "summary",
        datetime(2026, 1, 31, 5, tzinfo=UTC),
        timedelta(days=1),
        100,
        "CAD",
        power_of_ten_multiplier=0,
    )

    with patch.object(
        statistics, "_async_replace_statistics", new_callable=AsyncMock
    ) as replace_statistics:
        await statistics._async_update_gas_cost_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            None,
            [summary],
            allocation_mode="monthly_increment",
            merge_with_existing=False,
        )

    records = replace_statistics.await_args.args[2]
    assert records[0]["start"].isoformat() == "2026-02-01T00:00:00-05:00"


async def test_summary_only_gas_usage_uses_local_billing_end_date(
    hass: HomeAssistant,
) -> None:
    """Summary-only monthly usage records use the local billing end date."""
    await hass.config.async_set_time_zone("America/Toronto")
    summary = model.UsageSummary(
        "summary",
        datetime(2026, 1, 31, 5, tzinfo=UTC),
        timedelta(days=1),
        0,
        "CAD",
        consumption_m3=10,
    )
    assert statistics.gas_usage_total(None, [summary], "monthly_increment") == 10
    recorder = Mock()
    recorder.async_add_executor_job = AsyncMock(return_value=None)

    with (
        patch.object(statistics.recorder_helper, "get_instance", return_value=recorder),
        patch.object(
            statistics, "_async_replace_statistics", new_callable=AsyncMock
        ) as replace_statistics,
    ):
        await statistics._async_update_gas_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            None,
            [summary],
            allocation_mode="monthly_increment",
        )

    records = replace_statistics.await_args.args[2]
    assert records[0]["start"].isoformat() == "2026-02-01T00:00:00-05:00"
    assert records[0]["state"] == pytest.approx(10.0)


async def test_monthly_gas_usage_and_daily_cost_share_available_readings(
    hass: HomeAssistant,
) -> None:
    """Monthly usage can use summaries while costs use the same daily readings."""
    await hass.config.async_set_time_zone("America/Toronto")
    reading_type = model.ReadingType("type", 7, "CAD", 0, "m³", 86400)
    start = datetime(2026, 1, 1, 5, tzinfo=UTC)
    readings = [
        model.IntervalReading(reading_type, 0, start, timedelta(days=1), 4),
        model.IntervalReading(
            reading_type, 0, start + timedelta(days=1), timedelta(days=1), 6
        ),
    ]
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, start, timedelta(days=2), readings
            )
        ],
    )
    summary = model.UsageSummary(
        "summary", start, timedelta(days=2), 100, "CAD", consumption_m3=None
    )
    recorder = Mock()
    recorder.async_add_executor_job = AsyncMock(return_value=None)

    with (
        patch.object(statistics.recorder_helper, "get_instance", return_value=recorder),
        patch.object(
            statistics, "_async_replace_statistics", new_callable=AsyncMock
        ) as replace_statistics,
    ):
        await statistics._async_update_gas_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            meter,
            [summary],
            allocation_mode="monthly_increment",
        )
        usage_records = replace_statistics.await_args.args[2]
        await statistics._async_update_gas_cost_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            meter,
            [summary],
            allocation_mode="pro_rate_daily",
            gas_cost_multiplier=0,
            merge_with_existing=False,
        )
        cost_records = replace_statistics.await_args.args[2]

    assert [record["state"] for record in usage_records] == pytest.approx([10.0])
    assert [record["state"] for record in cost_records] == pytest.approx([40.0, 60.0])


async def test_monthly_gas_usage_accepts_unrepresented_long_reading(
    hass: HomeAssistant,
) -> None:
    """A multi-day gas reading can provide a billing increment without a summary."""
    reading_type = model.ReadingType("type", 7, "CAD", 0, "m³", 604800)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    reading = model.IntervalReading(reading_type, 0, start, timedelta(days=6), 12)
    meter = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, start, reading.duration, [reading]
            )
        ],
    )
    recorder = Mock()
    recorder.async_add_executor_job = AsyncMock(return_value=None)

    with (
        patch.object(statistics.recorder_helper, "get_instance", return_value=recorder),
        patch.object(
            statistics, "_async_replace_statistics", new_callable=AsyncMock
        ) as replace_statistics,
    ):
        await statistics._async_update_gas_statistics(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            meter,
            allocation_mode="monthly_increment",
        )

    assert [record["state"] for record in replace_statistics.await_args.args[2]] == [
        12.0
    ]


async def test_energy_statistics_split_trimmed_multi_hour_intervals(
    hass: HomeAssistant,
) -> None:
    """A trailing partial hour cannot move earlier energy into its last hour."""
    entity = _StatisticsEntity()
    with patch.object(
        statistics, "_get_all_existing_statistics", new_callable=AsyncMock
    ) as get_existing:
        get_existing.return_value = []
        initial = await statistics._generate_statistics_data(
            hass,
            entity,  # type: ignore[arg-type]
            statistics.DefaultDataExtractor(),
            _partial_hour_meter(False, None),
        )
        get_existing.return_value = initial
        completed = await statistics._generate_statistics_data(
            hass,
            entity,  # type: ignore[arg-type]
            statistics.DefaultDataExtractor(),
            _partial_hour_meter(True, None),
        )

    _assert_hourly_statistics(initial, 2)
    _assert_hourly_statistics(completed, 3)


async def test_cost_statistics_split_trimmed_multi_hour_intervals(
    hass: HomeAssistant,
) -> None:
    """A trailing partial hour cannot move earlier cost into its last hour."""
    entity = _StatisticsEntity()
    with patch.object(
        statistics, "_get_all_existing_statistics", new_callable=AsyncMock
    ) as get_existing:
        get_existing.return_value = []
        initial = await statistics._generate_statistics_data_cost(
            hass,
            entity,  # type: ignore[arg-type]
            statistics.CostDataExtractor(-2),
            _partial_hour_meter(False, None),
        )
        get_existing.return_value = initial
        completed = await statistics._generate_statistics_data_cost(
            hass,
            entity,  # type: ignore[arg-type]
            statistics.CostDataExtractor(-2),
            _partial_hour_meter(True, None),
        )

    _assert_hourly_statistics(initial, 2)
    _assert_hourly_statistics(completed, 3)


async def test_electricity_display_totals_match_complete_hour_statistics(
    hass: HomeAssistant,
) -> None:
    """A trailing partial interval cannot be included only in the display total."""
    meter_reading = _partial_hour_meter(False, None)
    entity = _StatisticsEntity()

    with patch.object(
        statistics,
        "_get_all_existing_statistics",
        new_callable=AsyncMock,
        return_value=[],
    ):
        usage_records = await statistics._generate_statistics_data(
            hass, entity, statistics.DefaultDataExtractor(), meter_reading
        )
        cost_records = await statistics._generate_statistics_data_cost(
            hass, entity, statistics.CostDataExtractor(-2), meter_reading
        )

    assert sensor._electricity_usage_total(meter_reading) == pytest.approx(
        usage_records[-1]["sum"]
    )
    assert sensor._electricity_cost_total(meter_reading, -2) == pytest.approx(
        cost_records[-1]["sum"]
    )


async def test_electricity_allocation_runs_in_the_executor(
    hass: HomeAssistant,
) -> None:
    """Historical interval allocation does not block Home Assistant's event loop."""
    original_executor_job = hass.async_add_executor_job
    executor_jobs: list[object] = []

    async def async_add_executor_job(
        job: Callable[..., object], *args: object
    ) -> object:
        executor_jobs.append(job)
        return await original_executor_job(job, *args)

    with (
        patch.object(
            hass, "async_add_executor_job", side_effect=async_add_executor_job
        ),
        patch.object(
            statistics,
            "_get_all_existing_statistics",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        await statistics._generate_statistics_data(
            hass,
            _StatisticsEntity(),  # type: ignore[arg-type]
            statistics.DefaultDataExtractor(),
            _partial_hour_meter(False),
        )

    assert executor_jobs == [statistics._electricity_usage_values]


def test_prorated_gas_cost_marks_incomplete_consumption_as_an_estimate() -> None:
    """A partial billing period assigns its bill to measured days by policy."""
    reading_type = model.ReadingType("type", 7, "CAD", 0, "m³", 86400)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    reading = model.IntervalReading(reading_type, 0, start, timedelta(days=1), 5)

    costs, estimated = allocation.prorated_cost_by_day(
        [reading],
        [(start, start + timedelta(days=2), decimal.Decimal("10"))],
        scaling.interval_value,
        UTC,
    )

    assert estimated
    assert costs == {start.date(): decimal.Decimal("10")}


@pytest.mark.usefixtures("recorder_mock")
async def test_statistics_read_failure_preserves_existing_history(
    hass: HomeAssistant,
) -> None:
    """A failed read cannot turn existing history into an empty replacement."""
    entity = _StatisticsEntity()
    existing = StatisticData(start=HISTORICAL, state=1.0, sum=1.0)
    recorder_statistics.async_add_external_statistics(
        hass,
        {
            "statistic_id": entity.long_term_statistics_id,
            "source": DOMAIN,
            "name": entity.name,
            "unit_of_measurement": entity.native_unit_of_measurement,
            "unit_class": "energy",
            "has_sum": True,
            "mean_type": StatisticMeanType.NONE,
        },
        [existing],
    )
    await async_wait_recording_done(hass)
    before = statistics_during_period(
        hass,
        HISTORICAL - timedelta(hours=1),
        statistic_ids={entity.long_term_statistics_id},
    )

    with (
        patch.object(
            statistics,
            "_get_all_existing_statistics",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ),
        pytest.raises(RuntimeError, match="database unavailable"),
    ):
        await statistics.update_statistics(
            hass,
            entity,  # type: ignore[arg-type]
            statistics.DefaultDataExtractor(),
            _partial_hour_meter(False),
        )

    assert (
        statistics_during_period(
            hass,
            HISTORICAL - timedelta(hours=1),
            statistic_ids={entity.long_term_statistics_id},
        )
        == before
    )


@pytest.mark.parametrize(
    ("records", "error"),
    [
        pytest.param(
            [
                StatisticData(
                    start=HISTORICAL + timedelta(minutes=1), state=1.0, sum=1.0
                )
            ],
            "Invalid statistic timestamp",
            id="invalid-timestamp",
        ),
        pytest.param([], "with no records", id="empty-replacement"),
    ],
)
async def test_statistics_validation_failure_does_not_queue_replacement(
    hass: HomeAssistant, records: list[StatisticData], error: str
) -> None:
    """Reject an invalid replacement before it reaches the recorder."""
    with patch.object(
        statistics._ReplaceStatisticsTask,
        "queue_task",
        new_callable=AsyncMock,
    ) as queue_task:
        with pytest.raises(ValueError, match=error):
            await statistics._async_replace_statistics(
                hass,
                statistics.create_metadata(_StatisticsEntity()),  # type: ignore[arg-type]
                records,
            )

    queue_task.assert_not_awaited()


async def test_unchanged_statistics_skip_recorder_replacement(
    hass: HomeAssistant,
) -> None:
    """Equivalent normalized source records do not rewrite an external series."""
    metadata = statistics.create_metadata(_StatisticsEntity())  # type: ignore[arg-type]
    records = [StatisticData(start=HISTORICAL, state=1.0, sum=1.0)]

    with patch.object(
        statistics._ReplaceStatisticsTask, "queue_task", new_callable=AsyncMock
    ) as queue_task:
        assert await statistics._async_replace_statistics(hass, metadata, records)
        assert not await statistics._async_replace_statistics(hass, metadata, records)

    queue_task.assert_awaited_once_with(hass, metadata, records)


@pytest.mark.usefixtures("recorder_mock")
async def test_statistics_write_failure_rolls_back_replacement(
    hass: HomeAssistant,
) -> None:
    """A recorder write failure preserves the series that was being replaced."""
    entity = _StatisticsEntity()
    existing = StatisticData(start=HISTORICAL, state=1.0, sum=1.0)
    metadata = statistics.create_metadata(entity)  # type: ignore[arg-type]
    recorder_statistics.async_add_external_statistics(hass, metadata, [existing])
    await async_wait_recording_done(hass)
    before = statistics_during_period(
        hass,
        HISTORICAL - timedelta(hours=1),
        statistic_ids={entity.long_term_statistics_id},
    )

    replacement = StatisticData(
        start=HISTORICAL + timedelta(hours=1), state=2.0, sum=3.0
    )
    with (
        patch.object(
            recorder_statistics,
            "_import_statistics_with_session",
            side_effect=RuntimeError("write failure"),
        ),
        pytest.raises(RuntimeError, match="write failure"),
    ):
        await statistics._async_replace_statistics(hass, metadata, [replacement])

    assert (
        statistics_during_period(
            hass,
            HISTORICAL - timedelta(hours=1),
            statistic_ids={entity.long_term_statistics_id},
        )
        == before
    )


async def test_statistics_writers_serialize_a_series(
    hass: HomeAssistant,
) -> None:
    """A later writer waits until the current series replacement finishes."""
    entity = _StatisticsEntity()
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def update(*_args: object) -> None:
        calls.append("start")
        started.set()
        await release.wait()
        calls.append("end")

    with patch.object(statistics, "_async_update_statistics", side_effect=update):
        first = hass.async_create_task(
            statistics.update_statistics(
                hass,
                entity,  # type: ignore[arg-type]
                statistics.DefaultDataExtractor(),
                _partial_hour_meter(False),
            )
        )
        await started.wait()
        second = hass.async_create_task(
            statistics.update_statistics(
                hass,
                entity,  # type: ignore[arg-type]
                statistics.DefaultDataExtractor(),
                _partial_hour_meter(False),
            )
        )
        await asyncio.sleep(0)
        assert calls == ["start"]
        release.set()
        await asyncio.gather(first, second)

    assert calls == ["start", "end", "start", "end"]


async def test_statistics_tasks_are_cancelled_on_unload(
    hass: HomeAssistant,
) -> None:
    """Tracked sensor jobs are cancelled and drained by config-entry unload."""
    started = asyncio.Event()

    async def wait_forever() -> None:
        started.set()
        await asyncio.Event().wait()

    task = statistics.async_schedule_statistics_update(hass, "entry", wait_forever())
    await started.wait()
    await statistics.async_cancel_statistics_tasks(hass, "entry")

    assert task.cancelled()


async def test_clear_statistics_task_propagates_recorder_failure(
    hass: HomeAssistant,
) -> None:
    """A recorder clear error resolves the queued task's waiting future."""
    future: asyncio.Future[None] = hass.loop.create_future()
    task = statistics._ClearStatisticsTask(hass, "green_button:test", future)

    with patch.object(
        statistics.statistics,
        "clear_statistics",
        side_effect=RuntimeError("database failed"),
    ):
        task.run(Mock())

    with pytest.raises(RuntimeError, match="database failed"):
        await future


async def test_queued_recorder_task_ignores_future_cancelled_during_unload(
    hass: HomeAssistant,
) -> None:
    """A queued recorder completion cannot overwrite an unload cancellation."""
    recorder = Mock()
    queued: list[statistics.tasks.RecorderTask] = []
    recorder.queue_task.side_effect = queued.append

    with patch.object(
        statistics.recorder_helper, "get_instance", return_value=recorder
    ):
        update_task = statistics.async_schedule_statistics_update(
            hass,
            "entry",
            statistics.clear_statistic(hass, "green_button:test"),
        )
        await asyncio.sleep(0)
        await statistics.async_cancel_statistics_tasks(hass, "entry")

    assert update_task.cancelled()
    assert len(queued) == 1
    queued[0].run(Mock())


async def _energy(hass: HomeAssistant, entity: GreenButtonStatisticsSensor) -> None:
    reading_type = model.ReadingType("type", 1, "CAD", None, "Wh", 3600)
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
    reading_type = model.ReadingType("type", 1, "CAD", None, "Wh", 3600)
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
        "summary", HISTORICAL, timedelta(days=30), 2.5, "CAD", 1, 0
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


def test_cost_display_uses_source_multiplier_before_configured_fallback(
    hass: HomeAssistant,
) -> None:
    """The displayed total follows the same source-first scaling as statistics."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test-entry",
        options={"electricity_cost_power_of_ten_multiplier": -3},
    )
    start = HISTORICAL

    def total_for(multiplier: int | None) -> float:
        coordinator = GreenButtonCoordinator(hass, entry)
        reading_type = model.ReadingType("type", 1, "CAD", multiplier, "Wh", 3600)
        reading = model.IntervalReading(
            reading_type, 800, start, timedelta(hours=1), 1000
        )
        meter = model.MeterReading(
            "meter",
            reading_type,
            [
                model.IntervalBlock(
                    "block", reading_type, start, timedelta(hours=1), [reading]
                )
            ],
        )
        coordinator._merge_usage_points(
            [model.UsagePoint("electricity", SensorDeviceClass.ENERGY, [meter])]
        )
        coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
        return float(GreenButtonCostSensor(coordinator, "meter").native_value)

    assert total_for(-5) == pytest.approx(0.008)
    assert total_for(None) == pytest.approx(0.8)
