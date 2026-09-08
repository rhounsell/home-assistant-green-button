"""A module defining calculators for statistics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
import dataclasses
import datetime
import decimal
import hashlib
import logging
import math
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, final

from homeassistant.components.recorder import (
    db_schema as recorder_db_schema,
    statistics,
    tasks,
)
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType
from homeassistant.components.recorder.models.statistics import StatisticMetaData
from homeassistant.core import HomeAssistant
from homeassistant.helpers import recorder as recorder_helper
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, VolumeConverter

from . import allocation, model, scaling
from .const import (
    DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    DOMAIN,
)
from .statistic_ids import statistic_id_from_unique_id

if TYPE_CHECKING:
    from homeassistant.components.recorder.core import Recorder


class GreenButtonEntity(Protocol):
    """Protocol for Green Button entities that support statistics."""

    @property
    def entity_id(self) -> str:
        """Return the entity ID."""
        ...

    @property
    def name(self) -> str:
        """Return the entity name."""
        ...

    @property
    def long_term_statistics_id(self) -> str:
        """Return the statistic ID."""
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the native unit of measurement."""
        ...

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update the entity's state and statistics."""
        ...


_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

_DATA_STATISTICS_LOCKS = f"{DOMAIN}_statistics_locks"
_DATA_STATISTICS_TASKS = f"{DOMAIN}_statistics_tasks"
_DATA_STATISTICS_FINGERPRINTS = f"{DOMAIN}_statistics_fingerprints"

# Notes on statistic behavior:
#
# Sensor platform stat computations:
#
# Stats computed from the sensor state history. The values of `sum`, `state`,
# and `last_reset` in a stat record are the values at the *end* of the record
# period (non-inclusive). IOW, if a sensor value changed exactly at the end of
# the period, it won't be noticed in that period.
#
# What happens when the sensor resets in the middle of a stat record?
#   - When `last_reset` changes, it's assumed that the point in time when it
#     changes (not the value of `last_reset`) is the new zero point.
#   - If both `last_reset` and `state` changed at the same time, then it's
#     assumed that the reset happened first. The state at that time is
#     considered an additional sum.
#
# Recorder integration stat computations:
#
# Samples are stored with a start and end range.
#
# Code assumes that the values of `sum`, `min`, `max`, and `mean` are the same
# across the entire period. IOW, it assumes that they are the values at the
# *start* of the record period. Indeed, the UI seems to assume the same. It
# seems to be to be the opposite of the Sensor platform.
#
# However, code for compiling the hourly statistic reads the sum from the last
# 5m entry, which matches the Sensor platform. See
# `_compile_hourly_statistics()`.
#
# I guess the assumption is that the values stored was reached at some point in
# the period, and we don't know when, so we assume all points are the same as
# the end. This method over-estimates the value of the sample. Seems like a bug
# to me...
#
# General notes:
#   - Statistics are collected every 5 minutes and records the changes since
#     the last 5m.
#   - Works best when the first stat has a sum of 0, because the UI's "Adjust
#     Statistics" page can't modify the sum of the first stat.
#
# `statistics_during_period()`
#   - Returns all stats whose start time is within the range (non-inclusive
#     end). See `_statistics_during_period_stmt()`.
#   - Also returns the newest stat whose start time is before the range. See
#     `_statistics_at_time()`.
#
# `statistic_during_period()`
#   - Returns the change in stats within the period (non-inclusive end). See
#     `_get_newest_sum_statistic_in_sub_period()`.
#   - If the start time falls within a stat record period (non-inclusive
#     *start*), that record is considered the oldest. If the start time is equal
#     to a record start time, then the previous record is considered the oldest.
#     See `_get_oldest_sum_statistic_in_sub_period()`.
#   -
#


def _queue_task(
    hass: HomeAssistant, task_ctor: Callable[[asyncio.Future[T]], tasks.RecorderTask]
) -> asyncio.Future[T]:
    future = asyncio.get_event_loop().create_future()
    recorder_helper.get_instance(hass).queue_task(task_ctor(future))
    #   RDH recorder_util.get_instance(hass).queue_task(task_ctor(future))
    return future


def _complete_future(future: asyncio.Future[T], value: T) -> None:
    """Set a result unless the waiter has already completed or cancelled."""

    def _set_result() -> None:
        if not future.done():
            future.set_result(value)

    future.get_loop().call_soon_threadsafe(_set_result)


def _complete_future_exception(future: asyncio.Future[T], err: BaseException) -> None:
    """Set an exception unless the waiter has already completed or cancelled."""

    def _set_exception() -> None:
        if not future.done():
            future.set_exception(err)

    future.get_loop().call_soon_threadsafe(_set_exception)


def _statistics_lock(hass: HomeAssistant, statistic_id: str) -> asyncio.Lock:
    """Return the lock serializing one statistics series."""
    locks = cast(
        dict[str, asyncio.Lock],
        hass.data.setdefault(_DATA_STATISTICS_LOCKS, {}),
    )
    return locks.setdefault(statistic_id, asyncio.Lock())


def async_schedule_statistics_update(
    hass: HomeAssistant,
    entry_id: str,
    coro: Coroutine[Any, Any, None],
) -> asyncio.Task[None]:
    """Schedule a statistics update that can be cancelled during unload."""
    task = hass.async_create_task(coro)
    tasks_by_entry = cast(
        dict[str, set[asyncio.Task[None]]],
        hass.data.setdefault(_DATA_STATISTICS_TASKS, {}),
    )
    entry_tasks = tasks_by_entry.setdefault(entry_id, set())
    entry_tasks.add(task)

    def _discard(completed: asyncio.Task[None]) -> None:
        entry_tasks.discard(completed)
        if not entry_tasks:
            tasks_by_entry.pop(entry_id, None)

    task.add_done_callback(_discard)
    return task


async def async_cancel_statistics_tasks(hass: HomeAssistant, entry_id: str) -> None:
    """Cancel and drain statistics updates belonging to an unloaded entry."""
    tasks_by_entry = cast(
        dict[str, set[asyncio.Task[None]]],
        hass.data.get(_DATA_STATISTICS_TASKS, {}),
    )
    entry_tasks = tasks_by_entry.pop(entry_id, set())
    for task in entry_tasks:
        task.cancel()
    if entry_tasks:
        await asyncio.gather(*entry_tasks, return_exceptions=True)


def _validated_statistics(
    metadata: StatisticMetaData,
    records: list[StatisticData],
) -> list[StatisticData]:
    """Validate a complete replacement before scheduling any mutation."""
    statistic_id = metadata["statistic_id"]
    if metadata["source"] != DOMAIN or not statistic_id.startswith(f"{DOMAIN}:"):
        raise ValueError(f"Invalid external statistic metadata for {statistic_id}")
    if not records:
        raise ValueError(f"Refusing to replace {statistic_id} with no records")

    validated: list[StatisticData] = []
    previous_start: datetime.datetime | None = None
    for record in records:
        start = record["start"]
        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or start.minute
            or start.second
            or start.microsecond
        ):
            raise ValueError(f"Invalid statistic timestamp for {statistic_id}: {start}")
        if previous_start is not None and start <= previous_start:
            raise ValueError(f"Statistics for {statistic_id} are not strictly ordered")
        if not all(math.isfinite(float(record[key])) for key in ("state", "sum")):
            raise ValueError(
                f"Statistics for {statistic_id} contain a non-finite value"
            )
        validated.append({**record, "start": start.astimezone(datetime.UTC)})
        previous_start = start
    return validated


@final
@dataclasses.dataclass(frozen=False)
class _ReplaceStatisticsTask(tasks.RecorderTask):
    """Replace a series in one recorder transaction."""

    metadata: StatisticMetaData
    records: list[StatisticData]
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        statistic_id = self.metadata["statistic_id"]
        try:
            with recorder_helper.session_scope(
                session=instance.get_session()
            ) as session:
                instance.statistics_meta_manager.delete(session, [statistic_id])
                statistics._import_statistics_with_session(  # noqa: SLF001
                    instance,
                    session,
                    self.metadata,
                    self.records,
                    recorder_db_schema.Statistics,
                )
        except Exception as err:
            _complete_future_exception(self.future, err)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        metadata: StatisticMetaData,
        records: list[StatisticData],
    ) -> asyncio.Future[None]:
        """Queue an atomic replacement and return its completion future."""

        def ctor(future: asyncio.Future[None]) -> _ReplaceStatisticsTask:
            return cls(metadata=metadata, records=records, future=future)

        return _queue_task(hass, ctor)


async def _async_replace_statistics(
    hass: HomeAssistant,
    metadata: StatisticMetaData,
    records: list[StatisticData],
) -> bool:
    """Validate and atomically replace an external series when it changed."""
    validated = _validated_statistics(metadata, records)
    digest = hashlib.sha256()
    for record in validated:
        digest.update(
            (
                f"{record['start'].isoformat()}\\0{record['state']:.17g}"
                f"\\0{record['sum']:.17g}\\n"
            ).encode()
        )
    fingerprints = cast(
        dict[str, str],
        hass.data.setdefault(_DATA_STATISTICS_FINGERPRINTS, {}),
    )
    statistic_id = metadata["statistic_id"]
    fingerprint = digest.hexdigest()
    if fingerprints.get(statistic_id) == fingerprint:
        _LOGGER.debug("Skipping unchanged Green Button statistic %s", statistic_id)
        return False
    _LOGGER.info(
        "Replacing %d records for %s from %s through %s",
        len(validated),
        statistic_id,
        validated[0]["start"],
        validated[-1]["start"],
    )
    await _ReplaceStatisticsTask.queue_task(hass, metadata, validated)
    fingerprints[statistic_id] = fingerprint
    return True


@final
@dataclasses.dataclass(frozen=False)
class _ClearStatisticsTask(tasks.RecorderTask):
    """Clear one external series and settle its waiting caller."""

    hass: HomeAssistant
    statistic_id: str
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        try:
            statistics.clear_statistics(
                instance=instance, statistic_ids=[self.statistic_id]
            )
        except Exception as err:
            _complete_future_exception(self.future, err)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(cls, hass: HomeAssistant, statistic_id: str) -> asyncio.Future[None]:
        """Queue the clear operation and return its completion future."""

        def ctor(future: asyncio.Future[None]) -> _ClearStatisticsTask:
            return cls(hass=hass, statistic_id=statistic_id, future=future)

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _RenameExternalStatisticTask(tasks.RecorderTask):
    """Rename an external Green Button series with its entity identity."""

    old_statistic_id: str
    new_statistic_id: str

    def run(self, instance: Recorder) -> None:
        """Rename only metadata owned by this integration."""
        with recorder_helper.session_scope(session=instance.get_session()) as session:
            instance.statistics_meta_manager.update_statistic_id(
                session, DOMAIN, self.old_statistic_id, self.new_statistic_id
            )


def rename_external_statistic(
    hass: HomeAssistant, old_unique_id: str, new_unique_id: str
) -> None:
    """Queue migration of an external series after an entity-ID migration."""
    recorder_helper.get_instance(hass).queue_task(
        _RenameExternalStatisticTask(
            statistic_id_from_unique_id(old_unique_id),
            statistic_id_from_unique_id(new_unique_id),
        )
    )


async def _get_all_existing_statistics(
    hass: HomeAssistant,
    statistic_id: str,
) -> list[StatisticData]:
    """Retrieve all existing hourly statistics for a statistic_id.

    Returns a list of StatisticData dictionaries sorted by start time.
    """
    rec = recorder_helper.get_instance(hass)

    def _get_stats() -> dict[str, list[Any]]:
        return statistics.statistics_during_period(
            hass=hass,
            start_time=datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
            end_time=datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc),
            statistic_ids={statistic_id},
            period="hour",
            types={"sum", "state"},
            units=None,
        )

    raw_stats = await rec.async_add_executor_job(_get_stats)
    stats_list = (raw_stats or {}).get(statistic_id, [])
    result: list[StatisticData] = []
    for stat in stats_list:
        stat_dict = cast(dict[str, Any], stat)
        start_val = stat_dict["start"]
        start_dt = (
            start_val
            if isinstance(start_val, datetime.datetime)
            else datetime.datetime.fromtimestamp(start_val, tz=datetime.timezone.utc)
        )
        result.append(
            {
                "start": start_dt,
                "state": float(stat_dict.get("state", 0.0)),
                "sum": float(stat_dict.get("sum", 0.0)),
            }
        )

    result.sort(key=lambda statistic: statistic["start"])
    return result


def _convert_to_kwh(value: float, source_unit: Any) -> float:
    """Convert energy value from source unit to kWh.

    Args:
        value: The energy value in the source unit
        source_unit: The source unit (e.g., UnitOfEnergy.WATT_HOUR or string like "Wh")

    Returns:
        The energy value in kWh
    """
    return float(allocation.energy_to_kwh(decimal.Decimal(str(value)), source_unit))


class DataExtractor(Protocol):
    """A protocol for an instance that can extract data from an IntervalReading."""

    def get_native_value(
        self, interval_reading: model.IntervalReading
    ) -> decimal.Decimal:
        """Get the native value from the IntervalReading."""
        ...


class DefaultDataExtractor:
    """Default implementation of DataExtractor."""

    def get_native_value(
        self, interval_reading: model.IntervalReading
    ) -> decimal.Decimal:
        """Get the native value from the IntervalReading."""
        if interval_reading.value is None:
            return decimal.Decimal(0)

        return scaling.interval_value(interval_reading)


class CostDataExtractor:
    """DataExtractor that pulls monetary cost from IntervalReading.

    Uses the source multiplier when declared, otherwise the configured fallback.
    """

    def __init__(
        self,
        cost_power_of_ten_multiplier: int = DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER,
    ) -> None:
        """Initialise the extractor with the given multiplier."""
        self._multiplier = cost_power_of_ten_multiplier
        _LOGGER.info(
            "CostDataExtractor initialized with multiplier: %d (10^%d)",
            cost_power_of_ten_multiplier,
            cost_power_of_ten_multiplier,
        )

    def get_native_value(
        self, interval_reading: model.IntervalReading
    ) -> decimal.Decimal:
        """
        Calculate the native value with source-first power-of-ten scaling.

        Args:
            interval_reading (model.IntervalReading): The interval reading object containing cost information.

        Returns:
            decimal.Decimal: The scaled monetary value.
        """
        return scaling.interval_cost(interval_reading, self._multiplier)


def create_metadata(entity: GreenButtonEntity) -> StatisticMetaData:
    """Create the statistic metadata for the entity."""
    return {
        "mean_type": StatisticMeanType.NONE,
        "has_sum": True,
        "name": f"{entity.name} ({entity.entity_id}, imported)",
        "source": DOMAIN,
        "statistic_id": entity.long_term_statistics_id,
        "unit_of_measurement": entity.native_unit_of_measurement,
        "unit_class": {
            "kWh": EnergyConverter.UNIT_CLASS,
            "m³": VolumeConverter.UNIT_CLASS,
        }.get(entity.native_unit_of_measurement),
    }


def _electricity_usage_values(
    meter_reading: model.MeterReading, data_extractor: DataExtractor
) -> dict[datetime.datetime, decimal.Decimal]:
    """Build complete-hour energy values without accessing Home Assistant."""
    return allocation.hourly_values(
        allocation.interval_readings(meter_reading),
        lambda reading: allocation.energy_to_kwh(
            data_extractor.get_native_value(reading),
            reading.reading_type.unit_of_measurement,
        ),
    )


def _electricity_cost_values(
    meter_reading: model.MeterReading, data_extractor: DataExtractor
) -> dict[datetime.datetime, decimal.Decimal]:
    """Build complete-hour cost values without accessing Home Assistant."""
    return allocation.hourly_values(
        allocation.interval_readings(meter_reading), data_extractor.get_native_value
    )


async def _generate_statistics_data(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
) -> list[StatisticData]:
    """Generate statistics data aggregated to full hours with out-of-order import support.

    This function handles imports in any order by:
    1. Getting all existing statistics
    2. Generating new statistics from the meter reading
    3. Merging them intelligently, recalculating sums as needed
    """
    values = await hass.async_add_executor_job(
        _electricity_usage_values, meter_reading, data_extractor
    )
    if not values:
        _LOGGER.info(
            "No complete hourly statistics generated for entity %s",
            entity.entity_id,
        )
        return []
    existing_stats = await _get_all_existing_statistics(
        hass,
        entity.long_term_statistics_id,
    )
    merged_stats = [
        cast(StatisticData, record)
        for record in allocation.merge_records(existing_stats, values)
    ]
    _LOGGER.debug(
        "Generated %d complete hourly statistics for entity %s",
        len(values),
        entity.entity_id,
    )
    return merged_stats


async def _generate_statistics_data_cost(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
    merge_with_existing: bool = True,
) -> list[StatisticData]:
    """Generate hourly cost statistics with out-of-order import support.

    Mirrors the energy statistics generation but uses monetary cost per interval
    without applying energy unit conversions.

    Args:
        hass: Home Assistant instance
        entity: The entity to generate statistics for
        data_extractor: The data extractor to use
        meter_reading: The meter reading to process
        merge_with_existing: If True, merge with existing statistics. If False, regenerate all from scratch.
    """
    values = await hass.async_add_executor_job(
        _electricity_cost_values, meter_reading, data_extractor
    )
    if not values:
        _LOGGER.info(
            "No complete hourly cost statistics generated for entity %s",
            entity.entity_id,
        )
        return []

    # Get all existing statistics for this entity (only if merging)
    if merge_with_existing:
        existing_stats = await _get_all_existing_statistics(
            hass,
            entity.long_term_statistics_id,
        )

        # Merge new statistics with existing ones, handling out-of-order imports
        merged_stats = [
            cast(StatisticData, record)
            for record in allocation.merge_records(existing_stats, values)
        ]

        # Log summary of what was processed
        _LOGGER.info(
            "Generated %d hourly cost statistics for entity %s (existing: %d, merged result: %d)",
            len(values),
            entity.entity_id,
            len(existing_stats),
            len(merged_stats),
        )
    else:
        # Recalculation mode: don't merge, just calculate running sum from scratch
        _LOGGER.info(
            "Recalculation mode: Regenerating ALL cost statistics from meter reading (not merging with existing)"
        )
        merged_stats = [
            cast(StatisticData, record)
            for record in allocation.cumulative_records(values)
        ]
        _LOGGER.info(
            "Generated %d hourly cost statistics for entity %s (recalculation mode)",
            len(merged_stats),
            entity.entity_id,
        )

    if merged_stats:
        _LOGGER.info(
            "Cost statistics range: %s (sum=%.2f) to %s (sum=%.2f)",
            merged_stats[0]["start"],
            merged_stats[0].get("sum", 0.0),
            merged_stats[-1]["start"],
            merged_stats[-1].get("sum", 0.0),
        )

    return merged_stats


async def _async_update_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
    merge_with_existing: bool = True,
) -> None:
    """Update the cost statistics for an entry to match the MeterReading.

    Args:
        hass: Home Assistant instance
        entity: The entity to update
        data_extractor: The data extractor to use
        meter_reading: The meter reading to process
        merge_with_existing: If True, merge with existing statistics. If False, regenerate all from scratch.
    """
    metadata = create_metadata(entity)
    _LOGGER.info(
        "Starting cost statistics generation for entity %s, meter reading %s (merge_with_existing=%s)",
        entity.entity_id,
        meter_reading.id,
        merge_with_existing,
    )
    statistics_data = await _generate_statistics_data_cost(
        hass, entity, data_extractor, meter_reading, merge_with_existing
    )

    _LOGGER.info(
        "Generated %d cost statistics records for entity %s",
        len(statistics_data),
        entity.entity_id,
    )

    if not statistics_data:
        _LOGGER.warning(
            "No cost statistics data generated for entity %s", entity.entity_id
        )
        return

    await _async_replace_statistics(hass, metadata, statistics_data)
    _LOGGER.info(
        "Replaced %d cost records for entity %s",
        len(statistics_data),
        entity.entity_id,
    )


async def _async_update_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
) -> None:
    """Update the statistics for an entry to match the MeterReading.

    This method imports historical statistics data properly with out-of-order support.
    """
    # Create metadata for the statistics
    metadata = create_metadata(entity)

    # Generate statistics data from meter reading
    _LOGGER.debug(
        "Starting statistics generation for entity %s, meter reading %s",
        entity.entity_id,
        meter_reading.id,
    )
    statistics_data = await _generate_statistics_data(
        hass, entity, data_extractor, meter_reading
    )

    _LOGGER.info(
        "Generated %d statistics records for entity %s",
        len(statistics_data),
        entity.entity_id,
    )

    if not statistics_data:
        _LOGGER.warning(
            "No statistics data generated for entity %s",
            entity.entity_id,
        )
        return

    await _async_replace_statistics(hass, metadata, statistics_data)
    _LOGGER.info(
        "Replaced %d statistics records for entity %s",
        len(statistics_data),
        entity.entity_id,
    )


async def clear_statistic(hass: HomeAssistant, statistic_id: str) -> None:
    """Clear all statistics with the specified ID."""
    await _ClearStatisticsTask.queue_task(hass=hass, statistic_id=statistic_id)
    fingerprints = cast(
        dict[str, str], hass.data.get(_DATA_STATISTICS_FINGERPRINTS, {})
    )
    fingerprints.pop(statistic_id, None)


# -------------------- GAS (m³) DAILY STATISTICS --------------------


def _billing_timezone() -> datetime.tzinfo:
    """Return the configured Home Assistant timezone for billing dates."""
    return dt_util.get_default_time_zone()


def _gas_daily_totals(
    readings: Sequence[model.IntervalReading],
    time_zone: datetime.tzinfo,
    range_start: datetime.datetime | None = None,
    range_end: datetime.datetime | None = None,
) -> dict[datetime.date, decimal.Decimal]:
    """Allocate gas interval overlap across local calendar days."""
    return allocation.local_day_values(
        readings,
        scaling.interval_value,
        time_zone,
        range_start,
        range_end,
    )


def _is_gas_billing_period_reading(reading: model.IntervalReading) -> bool:
    """Return whether a reading represents more than one daily gas interval."""
    return (
        reading.reading_type.interval_length > 86400
        or reading.duration > datetime.timedelta(days=1)
    )


def gas_usage_values(
    meter_reading: model.MeterReading | None,
    usage_summaries: Sequence[model.UsageSummary],
    allocation_mode: str,
    time_zone: datetime.tzinfo,
) -> dict[datetime.datetime, decimal.Decimal]:
    """Return normalized source increments for one gas usage series."""
    readings = allocation.interval_readings(meter_reading)
    if allocation_mode != "monthly_increment":
        return {
            datetime.datetime.combine(day, datetime.time.min, tzinfo=time_zone): value
            for day, value in allocation.local_day_values(
                readings, scaling.interval_value, time_zone
            ).items()
        }

    values: dict[datetime.datetime, decimal.Decimal] = {}
    periods: list[
        tuple[datetime.datetime, datetime.datetime, decimal.Decimal | None]
    ] = [
        (
            summary.start,
            summary.start + summary.duration,
            (
                scaling.usage_summary_consumption(summary)
                if summary.consumption_m3 is not None
                else None
            ),
        )
        for summary in usage_summaries
    ]
    for reading in readings:
        if not _is_gas_billing_period_reading(reading) or any(
            reading.start < summary.start + summary.duration
            and summary.start < reading.end
            for summary in usage_summaries
        ):
            continue
        periods.append((reading.start, reading.end, scaling.interval_value(reading)))

    for start, end, consumption in periods:
        if consumption is None:
            daily_values = allocation.local_day_values(
                readings, scaling.interval_value, time_zone, start, end
            )
            consumption = (
                sum(daily_values.values(), decimal.Decimal(0)) if daily_values else None
            )
        if consumption is None or consumption < 0:
            continue
        record_start = datetime.datetime.combine(
            end.astimezone(time_zone).date(), datetime.time.min, tzinfo=time_zone
        )
        values[record_start] = (
            values.get(record_start, decimal.Decimal(0)) + consumption
        )
    return values


def gas_usage_total(
    meter_reading: model.MeterReading | None,
    usage_summaries: Sequence[model.UsageSummary],
    allocation_mode: str,
) -> float:
    """Return the display total represented by normalized gas source increments."""
    return float(
        sum(
            gas_usage_values(
                meter_reading,
                usage_summaries,
                allocation_mode,
                _billing_timezone(),
            ).values(),
            decimal.Decimal(0),
        )
    )


async def _generate_daily_m3_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading,
) -> list[StatisticData]:
    """Generate daily statistics for gas consumption (m³) with out-of-order import support.

    We emit one hourly record per day at 00:00 with the day's total m³ as state.
    """
    time_zone = _billing_timezone()
    values = await hass.async_add_executor_job(
        gas_usage_values, meter_reading, [], "daily_readings", time_zone
    )
    if not values:
        return []

    # Get all existing statistics for this entity
    existing_stats = await _get_all_existing_statistics(
        hass,
        entity.long_term_statistics_id,
    )

    return [
        cast(StatisticData, record)
        for record in allocation.merge_records(existing_stats, values)
    ]


async def _async_update_gas_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading | None,
    usage_summaries: list[model.UsageSummary] | None = None,
    allocation_mode: str = "daily_readings",
) -> None:
    """Import gas m³ statistics.

    Modes:
    - daily_readings: one record per day at 00:00 with that day's m³ (default)
    - monthly_increment: one record per UsageSummary at end-of-period day with total m³
      (uses UsageSummary.consumption_m3 when available)

    Args:
        hass: Home Assistant instance
        entity: Gas sensor entity
        meter_reading: MeterReading with daily interval data (optional for monthly_increment mode)
        usage_summaries: List of UsageSummary objects for billing periods
        allocation_mode: "daily_readings" or "monthly_increment"
    """
    metadata = create_metadata(entity)

    if allocation_mode == "monthly_increment":
        summaries = usage_summaries or []
        _LOGGER.info(
            "Gas %s: monthly_increment mode - processing %d usage summaries",
            entity.entity_id,
            len(summaries),
        )
        time_zone = _billing_timezone()
        values = await hass.async_add_executor_job(
            gas_usage_values,
            meter_reading,
            summaries,
            allocation_mode,
            time_zone,
        )

        if not values:
            _LOGGER.warning(
                "Gas %s: No gas usage records generated - all periods had no consumption data",
                entity.entity_id,
            )
            return

        existing_stats = await _get_all_existing_statistics(
            hass, entity.long_term_statistics_id
        )
        records = [
            cast(StatisticData, record)
            for record in allocation.merge_records(existing_stats, values)
        ]

        await _async_replace_statistics(hass, metadata, records)
        _LOGGER.info(
            "Replaced %d gas usage records for %s (total: %.1f m³)",
            len(records),
            entity.entity_id,
            records[-1].get("sum", 0.0),
        )

        return

    # Default: daily readings mode requires a MeterReading
    if not meter_reading:
        _LOGGER.warning(
            "Cannot generate daily gas statistics for %s - no meter reading data available. "
            "Consider using monthly_increment mode if only UsageSummaries are available.",
            entity.entity_id,
        )
        return

    data = await _generate_daily_m3_statistics(hass, entity, meter_reading)
    if not data:
        _LOGGER.info("No gas statistics to import for %s", entity.entity_id)
        return

    await _async_replace_statistics(hass, metadata, data)
    _LOGGER.info("Replaced %d gas daily records for %s", len(data), entity.entity_id)


async def _async_update_gas_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading | None,
    usage_summaries: list[model.UsageSummary],
    allocation_mode: str = "pro_rate_daily",
    gas_cost_multiplier: int = DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    merge_with_existing: bool = True,
) -> None:
    """Import pro-rated daily gas costs based on UsageSummary totals and daily m³.

    For each billing period, distribute total_cost across days proportionally
    to daily consumption in m³. Emit one hourly record per day at 00:00.

    Args:
        hass: Home Assistant instance
        entity: The gas cost sensor entity
        meter_reading: MeterReading containing interval data (optional for monthly mode)
        usage_summaries: List of UsageSummary with billing totals
        allocation_mode: Either "pro_rate_daily" or "monthly_increment"
        gas_cost_multiplier: Fallback power-of-ten multiplier for gas costs.
        merge_with_existing: If True, merge with existing statistics. If False, regenerate all from scratch.
    """
    metadata = create_metadata(entity)

    _LOGGER.info(
        "Starting gas cost statistics generation for entity %s (merge_with_existing=%s)",
        entity.entity_id,
        merge_with_existing,
    )

    if allocation_mode == "monthly_increment":
        # One increment per usage summary at the period end (00:00 of end day)
        if not usage_summaries:
            _LOGGER.info(
                "No usage summaries for monthly gas cost on %s", entity.entity_id
            )
            return
        time_zone = _billing_timezone()

        values: dict[datetime.datetime, decimal.Decimal] = {}
        for us in usage_summaries:
            period_end = us.start + us.duration
            # Monthly increments belong to the local billing end date, not the
            # last covered day, even when the period ends at local midnight.
            rec_start = datetime.datetime.combine(
                period_end.astimezone(time_zone).date(),
                datetime.time.min,
                tzinfo=time_zone,
            )
            values[rec_start] = values.get(rec_start, decimal.Decimal(0)) + (
                scaling.usage_summary_cost(us, gas_cost_multiplier)
            )

        if not values:
            return

        # Get all existing statistics for this entity (only if merging)
        if merge_with_existing:
            existing_stats = await _get_all_existing_statistics(
                hass,
                entity.long_term_statistics_id,
            )

            records = [
                cast(StatisticData, record)
                for record in allocation.merge_records(existing_stats, values)
            ]
        else:
            # Recalculation mode: don't merge, just calculate running sum from scratch
            _LOGGER.info(
                "Recalculation mode: Regenerating ALL gas cost statistics from usage summaries (not merging with existing)"
            )
            records = [
                cast(StatisticData, record)
                for record in allocation.cumulative_records(values)
            ]

    else:
        # Pro-rate daily across billing period days proportional to m³
        # Build daily m³ map (same as in gas stats)
        if not meter_reading:
            _LOGGER.warning(
                "Gas Cost Sensor %s: Cannot use pro_rate_daily mode without MeterReading (daily readings). "
                "Use monthly_increment mode for UsageSummary-only data.",
                entity.entity_id,
            )
            return
        readings = allocation.interval_readings(meter_reading)
        if not readings:
            _LOGGER.info(
                "No gas readings for cost distribution on %s", entity.entity_id
            )
            return
        time_zone = _billing_timezone()
        if not usage_summaries:
            _LOGGER.info("No gas usage summaries provided for %s", entity.entity_id)
            return

        periods = [
            (
                summary.start,
                summary.start + summary.duration,
                scaling.usage_summary_cost(summary, gas_cost_multiplier),
            )
            for summary in usage_summaries
        ]
        daily_cost, estimated = await hass.async_add_executor_job(
            allocation.prorated_cost_by_day,
            readings,
            periods,
            scaling.interval_value,
            time_zone,
        )
        if estimated:
            _LOGGER.info(
                "Gas daily cost for %s is estimated from available consumption coverage",
                entity.entity_id,
            )

        if not daily_cost:
            _LOGGER.info("No daily cost allocations computed for %s", entity.entity_id)
            return

        values = {
            datetime.datetime.combine(day, datetime.time.min, tzinfo=time_zone): value
            for day, value in daily_cost.items()
        }

        # Get all existing statistics for this entity (only if merging)
        if merge_with_existing:
            existing_stats = await _get_all_existing_statistics(
                hass,
                entity.long_term_statistics_id,
            )

            records = [
                cast(StatisticData, record)
                for record in allocation.merge_records(existing_stats, values)
            ]
        else:
            # Recalculation mode: don't merge, just calculate running sum from scratch
            _LOGGER.info(
                "Recalculation mode: Regenerating ALL gas cost statistics from meter reading (not merging with existing)"
            )
            records = [
                cast(StatisticData, record)
                for record in allocation.cumulative_records(values)
            ]

    if not records:
        return

    await _async_replace_statistics(hass, metadata, records)
    _LOGGER.info("Replaced %d gas cost records for %s", len(records), entity.entity_id)


async def update_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
) -> None:
    """Serialize and replace electricity usage statistics."""
    async with _statistics_lock(hass, entity.long_term_statistics_id):
        await _async_update_statistics(hass, entity, data_extractor, meter_reading)


async def update_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
    merge_with_existing: bool = True,
) -> None:
    """Serialize and replace electricity cost statistics."""
    async with _statistics_lock(hass, entity.long_term_statistics_id):
        await _async_update_cost_statistics(
            hass, entity, data_extractor, meter_reading, merge_with_existing
        )


async def update_gas_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading | None,
    usage_summaries: list[model.UsageSummary] | None = None,
    allocation_mode: str = "daily_readings",
) -> None:
    """Serialize and replace gas usage statistics."""
    async with _statistics_lock(hass, entity.long_term_statistics_id):
        await _async_update_gas_statistics(
            hass, entity, meter_reading, usage_summaries, allocation_mode
        )


async def update_gas_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading | None,
    usage_summaries: list[model.UsageSummary],
    allocation_mode: str = "pro_rate_daily",
    gas_cost_multiplier: int = DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER,
    merge_with_existing: bool = True,
) -> None:
    """Serialize and replace gas cost statistics."""
    async with _statistics_lock(hass, entity.long_term_statistics_id):
        await _async_update_gas_cost_statistics(
            hass,
            entity,
            meter_reading,
            usage_summaries,
            allocation_mode,
            gas_cost_multiplier,
            merge_with_existing,
        )
