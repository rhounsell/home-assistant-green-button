"""A module defining calculators for statistics."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import decimal
import logging
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import final
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import TYPE_CHECKING

from homeassistant import exceptions
from homeassistant.components.recorder import db_schema as recorder_db_schema
from homeassistant.components.recorder import statistics
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType
from homeassistant.components.recorder.models.statistics import StatisticMetaData
from homeassistant.components.recorder import tasks
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers import recorder as recorder_helper

from . import model

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


@final
@dataclasses.dataclass(frozen=True)
class _SensorStatRecord:
    timestamp: datetime.datetime
    last_reset: datetime.datetime | None
    state: decimal.Decimal
    sum: decimal.Decimal

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> _SensorStatRecord:
        """Create a stat record from the raw database record."""
        return _SensorStatRecord(
            timestamp=record["end"],
            last_reset=record.get("last_reset"),
            state=decimal.Decimal(record["state"]),
            sum=decimal.Decimal(record["sum"]),
        )

    def to_statistics_data(
        self, period: Literal["5minute", "hour"]
    ) -> StatisticData:
        """Create a StatisticData from this record."""
        return StatisticData(
            start=self.timestamp - _to_time_delta(period),
            last_reset=self.last_reset,
            state=float(self.state),
            sum=float(self.sum),
        )


@final
@dataclasses.dataclass(frozen=True)
class _StatisticSamples:
    prev_sum_before_end: float | None
    samples: list[_SensorStatRecord]

    def get_total_change(self) -> float:
        """Return the total change of the sum if these samples are applied."""
        if not self.samples:
            return 0.0
        if self.prev_sum_before_end is None:
            return float(self.samples[-1].sum)
        return float(self.samples[-1].sum) - self.prev_sum_before_end


@final
@dataclasses.dataclass(frozen=True)
class _MergedIntervalBlock:
    ids: list[str]
    reading_type: model.ReadingType
    start: datetime.datetime
    duration: datetime.timedelta
    interval_readings: list[model.IntervalReading]

    @property
    def end(self) -> datetime.datetime:
        """Return the block's interval's end time."""
        return self.start + self.duration

    @classmethod
    def create(cls, interval_blocks: list[model.IntervalBlock]) -> _MergedIntervalBlock:
        """Create a new merge block."""
        if not interval_blocks:
            raise ValueError("interval_blocks cannot be empty.")
        return cls(
            ids=[block.id for block in interval_blocks],
            reading_type=interval_blocks[0].reading_type,
            start=interval_blocks[0].start,
            duration=interval_blocks[-1].end - interval_blocks[0].start,
            interval_readings=[
                reading
                for block in interval_blocks
                for reading in block.interval_readings
            ],
        )


def _merge_interval_blocks(
    interval_blocks: Sequence[model.IntervalBlock],
) -> list[_MergedIntervalBlock]:
    res: list[_MergedIntervalBlock] = []
    merged_blocks: list[model.IntervalBlock] = []
    for curr_block in interval_blocks:
        if not merged_blocks:
            merged_blocks.append(curr_block)
            continue
        prev_block = merged_blocks[-1]
        if prev_block.end == curr_block.start:
            merged_blocks.append(curr_block)
            continue
        res.append(_MergedIntervalBlock.create(merged_blocks))
        merged_blocks = [curr_block]
    if merged_blocks:
        res.append(_MergedIntervalBlock.create(merged_blocks))
    return res


def _to_table(
    period: Literal["5minute", "hour"],
) -> type[recorder_db_schema.StatisticsShortTerm | recorder_db_schema.Statistics]:
    if period == "5minute":
        return recorder_db_schema.StatisticsShortTerm
    if period == "hour":
        return recorder_db_schema.Statistics


def _to_time_delta(period: Literal["5minute", "hour"]) -> datetime.timedelta:
    return _to_table(period).duration


def _round_down(
    datetime_val: datetime.datetime, period: Literal["5minute", "hour"]
) -> datetime.datetime:
    if period == "5minute":
        return datetime_val.replace(
            minute=datetime_val.minute - (datetime_val.minute % 5),
            second=0,
            microsecond=0,
        )
    if period == "hour":
        return datetime_val.replace(
            minute=0,
            second=0,
            microsecond=0,
        )


def _round_up(
    datetime_val: datetime.datetime, period: Literal["5minute", "hour"]
) -> datetime.datetime:
    return _round_down(
        datetime_val=datetime_val
        + _to_time_delta(period)
        - datetime.timedelta.resolution,
        period=period,
    )


def _is_aligned(
    datetime_val: datetime.datetime, period: Literal["5minute", "hour"]
) -> bool:
    return datetime_val == _round_down(datetime_val, period)


def _adjust_for_end_time(
    datetime_val: datetime.datetime, period: Literal["5minute", "hour"]
) -> datetime.datetime:
    """Return the start time of the stat record that would contain the datetime.

    If the datetime is on a period boundary, then return the previous period
    boundary. This is useful for treating stat records as representing the state
    of the world at the record's end time when query methods compare ranges
    against the start time (like the current state of the query methods).
    """
    return _round_down(datetime_val - datetime.timedelta.resolution, period)


def _queue_task(
    hass: HomeAssistant, task_ctor: Callable[[asyncio.Future[T]], tasks.RecorderTask]
) -> asyncio.Future[T]:
    future = asyncio.get_event_loop().create_future()
    recorder_helper.get_instance(hass).queue_task(task_ctor(future))
#   RDH recorder_util.get_instance(hass).queue_task(task_ctor(future))
    return future


def _complete_future(future: asyncio.Future[T], value: T) -> None:
    future.get_loop().call_soon_threadsafe(future.set_result, value)


async def _get_all_existing_statistics(
    hass: HomeAssistant,
    statistic_id: str,
) -> list[StatisticData]:
    """Retrieve all existing hourly statistics for a statistic_id.
    
    Returns a list of StatisticData dictionaries sorted by start time.
    """
    try:
        rec = recorder_helper.get_instance(hass)
        
        # Define a wrapper to call with keyword arguments
        # Use a very wide date range to get all statistics
        def _get_stats() -> dict[str, list[Any]]:
            return statistics.statistics_during_period(
                hass=hass,
                start_time=datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),  # earliest possible date
                end_time=datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc),  # far future date
                statistic_ids={statistic_id},
                period="hour",
                types={"sum", "state"},
                units=None,
            )
        
        # Get all statistics
        raw_stats = await rec.async_add_executor_job(_get_stats)
        
        stats_list = raw_stats.get(statistic_id, [])
        if not stats_list:
            return []
        
        # Convert to StatisticData format
        result: list[StatisticData] = []
        for stat in stats_list:
            stat_dict = cast(dict[str, Any], stat)
            # Convert start time - it may be a datetime or a timestamp float
            start_val = stat_dict["start"]
            if isinstance(start_val, datetime.datetime):
                start_dt = start_val
            else:
                # It's a Unix timestamp (float)
                start_dt = datetime.datetime.fromtimestamp(start_val, tz=datetime.timezone.utc)
            
            stat_data: StatisticData = {
                "start": start_dt,
                "state": float(stat_dict.get("state", 0.0)),
                "sum": float(stat_dict.get("sum", 0.0)),
            }
            result.append(stat_data)
        
        # Sort by start time
        result.sort(key=lambda s: s["start"])
        return result
    except Exception:
        _LOGGER.warning(
            "Could not retrieve existing statistics for %s",
            statistic_id,
            exc_info=True,
        )
        return []


def _find_last_statistic_before(
    existing_stats: list[StatisticData],
    target_time: datetime.datetime,
) -> StatisticData | None:
    """Find the last statistic with start time before target_time.
    
    Returns None if no such statistic exists.
    """
    for stat in reversed(existing_stats):
        if stat["start"] < target_time:
            return stat
    return None


def _merge_statistics_with_out_of_order_support(
    existing_stats: list[StatisticData],
    new_stats: list[StatisticData],
    statistic_id: str,
) -> list[StatisticData]:
    """Merge new statistics into existing statistics, handling out-of-order imports.
    
    Algorithm:
    1. Determine the date range of new data
    2. Find the last existing statistic before the new data (baseline)
    3. Remove existing statistics that overlap with the new data range (they get replaced)
    4. Add new data with cumulative sums starting from baseline
    5. Preserve statistics after the new data range, recalculating their sums
       with the updated baseline (important for restart scenarios where stored XML
       may not have the latest imported data)
    
    Args:
        existing_stats: List of existing StatisticData sorted by start time
        new_stats: List of new StatisticData to merge, sorted by start time
        statistic_id: The ID of the statistic being merged
        
    Returns:
        Complete list of StatisticData that should be imported
    """
    if not new_stats:
        return []
    
    if not existing_stats:
        # No existing data, just return new stats with cumulative sums starting from 0
        result: list[StatisticData] = []
        cumulative = 0.0
        for stat in new_stats:
            cumulative += stat.get("state", 0.0)
            stat_data: StatisticData = {
                "start": stat["start"],
                "state": float(stat.get("state", 0.0)),
                "sum": cumulative,
            }
            result.append(stat_data)
        return result
    
    # Find the range of new data
    first_new_start = new_stats[0]["start"]
    last_new_start = new_stats[-1]["start"]
    
    _LOGGER.debug(
        "Merging statistics for %s: new data range %s to %s",
        statistic_id,
        first_new_start,
        last_new_start,
    )
    
    # Find the baseline: last existing statistic before the new data
    baseline_stat = _find_last_statistic_before(existing_stats, first_new_start)
    baseline_sum = baseline_stat.get("sum", 0.0) if baseline_stat else 0.0
    
    _LOGGER.debug(
        "Baseline sum before new data for %s: %.3f (from %s)",
        statistic_id,
        baseline_sum,
        baseline_stat["start"] if baseline_stat else "start",
    )
    
    # Separate existing stats into:
    # 1. Stats before the new data (keep as-is)
    # 2. Stats that overlap or are within the new data range (discard, will be replaced)
    # 3. Stats after the new data (need recalculation with new baseline)
    
    stats_before = []
    stats_after = []
    
    for stat in existing_stats:
        if stat["start"] < first_new_start:
            stats_before.append(stat)
        elif stat["start"] > last_new_start:
            stats_after.append(stat)
        # Stats in between are discarded (will be replaced by new_stats)
    
    _LOGGER.debug(
        "Existing statistics for %s: %d before, %d after new data range",
        statistic_id,
        len(stats_before),
        len(stats_after),
    )
    
    # Build the result list
    result = []
    
    # 1. Add all stats before the new data (unchanged)
    result.extend(stats_before)
    
    # 2. Add new statistics with cumulative sums starting from baseline
    cumulative = baseline_sum
    for stat in new_stats:
        cumulative += stat.get("state", 0.0)
        stat_data: StatisticData = {
            "start": stat["start"],
            "state": float(stat.get("state", 0.0)),
            "sum": cumulative,
        }
        result.append(stat_data)
    
    # 3. Preserve and recalculate statistics after the new data range
    # This handles the case where we're importing older data and need to maintain
    # newer statistics that were previously imported (e.g., after restart when
    # stored XML might not have the latest data).
    if stats_after:
        _LOGGER.info(
            "Preserving %d existing statistics for %s that are after the new data range (after %s). "
            "Recalculating their sums with new baseline.",
            len(stats_after),
            statistic_id,
            last_new_start,
        )
        # The last cumulative sum from new data becomes the baseline for stats_after
        for stat in stats_after:
            cumulative += stat.get("state", 0.0)
            stat_data: StatisticData = {
                "start": stat["start"],
                "state": float(stat.get("state", 0.0)),
                "sum": cumulative,
            }
            result.append(stat_data)
    
    return result


def _convert_to_kwh(value: float, source_unit: Any) -> float:
    """Convert energy value from source unit to kWh.
    
    Args:
        value: The energy value in the source unit
        source_unit: The source unit (e.g., UnitOfEnergy.WATT_HOUR or string like "Wh")
    
    Returns:
        The energy value in kWh
    """
    # Normalize to string code when possible
    unit_str = None
    try:
        if isinstance(source_unit, str):
            unit_str = source_unit.lower()
        else:
            # Enum value from UnitOfEnergy
            unit_str = str(source_unit).lower()
    except Exception:  # pragma: no cover - defensive
        unit_str = None

    if source_unit == UnitOfEnergy.WATT_HOUR or unit_str in {"watt-hour", "wh"}:
        # Convert Wh to kWh
        return value / 1000.0
    elif source_unit == UnitOfEnergy.KILO_WATT_HOUR or unit_str in {"kilowatt-hour", "kwh"}:
        # Already in kWh
        return value
    elif source_unit == UnitOfEnergy.MEGA_WATT_HOUR or unit_str in {"megawatt-hour", "mwh"}:
        # Convert MWh to kWh
        return value * 1000.0
    else:
        # Unknown unit, assume it's already in the correct unit
        _LOGGER.warning(
            "Unknown energy unit '%s', assuming value is already in kWh",
            source_unit,
        )
        return value


class _StatsDao:
    def __init__(
        self,
        hass: HomeAssistant,
        statistic_id: str,
    ) -> None:
        self._hass = hass
        self._statistic_id = statistic_id

    def statistics_during_period_from_end_time(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        period: Literal["5minute", "hour"],
    ) -> list[_SensorStatRecord]:
        """Return the stats whose end time lies in the range (non-inclusive)."""
        # We adjust the range by subtracting the resolution and rounding down so
        # that the results return exactly the stat records whose end time lie in the
        # range.
        raw_data = statistics.statistics_during_period(
            hass=self._hass,
            start_time=_adjust_for_end_time(start, period),
            end_time=_adjust_for_end_time(end, period),
            statistic_ids={self._statistic_id},
            period=period,
            types={"sum", "state"},
            units=None,
        ).get(self._statistic_id, [])
        if not raw_data:
            return []

        data = [_SensorStatRecord.from_dict(cast(dict[str, Any], record)) for record in raw_data]
        # Remove the head if the stat is before the requested range. This can
        # happen because `statistics_during_period` will attempt always attempt
        # to append the most recent stat record that starts before the requested
        # start time. It does this even if that record's end time is also before
        # the requested start time. Since we clamp the start time to the period,
        # this can only happen if there is a gap in data (e.g., if HASS is not
        # running when it should have collected that data point).
        if data[0].timestamp < start:
            return data[1:]
        return data

    def compute_sum_before(self, timestamp: datetime.datetime) -> _SensorStatRecord:
        """Compute the sum statistics before the specified time."""
        # We need to round up because the end time is non-inclusive.
        sample_datetime = _round_up(
            timestamp + datetime.timedelta.resolution, "5minute"
        )
        sum_before = statistics.statistic_during_period(
            hass=self._hass,
            start_time=None,
            end_time=sample_datetime,
            statistic_id=self._statistic_id,
            types={"change"},
            units=None,
        ).get("change")
        if sum_before is None:
            sum_before = 0
        sum_decimal = decimal.Decimal(sum_before)
        return _SensorStatRecord(
            timestamp=sample_datetime,
            last_reset=None,
            state=sum_decimal,
            sum=sum_decimal,
        )


class _ComputeUpdatedPeriodStatisticsTask(tasks.RecorderTask):
    def __init__(
        self,
        hass: HomeAssistant,
        statistic_id: str,
        data_extractor: DataExtractor,
        interval_block: _MergedIntervalBlock,
        period: Literal["5minute", "hour"],
        future: asyncio.Future[_StatisticSamples],
    ) -> None:
        self._hass = hass
        self._statistic_id = statistic_id
        self._data_extractor = data_extractor
        self._interval_block = interval_block
        self._period: Literal["5minute", "hour"] = period
        self._future = future

    def _statistics_during_period_from_end_time(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> list[_SensorStatRecord]:
        """Return the stats whose end time lies in the range (non-inclusive)."""
        # We adjust the range by subtracting the resolution and rounding down so
        # that the results return exactly the stat records whose end time lie in the
        # range.
        raw_data = statistics.statistics_during_period(
            hass=self._hass,
            start_time=_adjust_for_end_time(start, self._period),
            end_time=_adjust_for_end_time(end, self._period),
            statistic_ids={self._statistic_id},
            period=self._period,
            types={"sum", "state"},
            units=None,
        ).get(self._statistic_id, [])
        if not raw_data:
            return []

        data = [_SensorStatRecord.from_dict(cast(dict[str, Any], record)) for record in raw_data]
        # Remove the head if the stat is before the requested range. This can
        # happen because `statistics_during_period` will attempt always attempt
        # to append the most recent stat record that starts before the requested
        # start time. It does this even if that record's end time is also before
        # the requested start time. Since we clamp the start time to the period,
        # this can only happen if there is a gap in data (e.g., if HASS is not
        # running when it should have collected that data point).
        if data[0].timestamp < start:
            return data[1:]
        return data

    def _compute_sum_before_old(
        self, timestamp: datetime.datetime
    ) -> _SensorStatRecord:
        # We need to round up because the end time is non-inclusive.
        sample_datetime = _round_up(
            timestamp + datetime.timedelta.resolution, "5minute"
        )
        sum_before = statistics.statistic_during_period(
            hass=self._hass,
            start_time=None,
            end_time=sample_datetime,
            statistic_id=self._statistic_id,
            types={"change"},
            units=None,
        ).get("change")
        if sum_before is None:
            sum_before = 0
        sum_decimal = decimal.Decimal(sum_before)
        return _SensorStatRecord(
            timestamp=sample_datetime,
            last_reset=None,
            state=sum_decimal,
            sum=sum_decimal,
        )

    def _compute_sum_before(self, timestamp: datetime.datetime) -> float:
        # We need to round up because the end time is non-inclusive.
        sample_datetime = _round_up(
            timestamp + datetime.timedelta.resolution, "5minute"
        )
        sum_before = statistics.statistic_during_period(
            hass=self._hass,
            start_time=None,
            end_time=sample_datetime,
            statistic_id=self._statistic_id,
            types={"change"},
            units=None,
        ).get("change")
        if sum_before is None:
            sum_before = 0.0
        return sum_before

    def _read_stats_and_generate_samples(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> tuple[datetime.timedelta, list[datetime.datetime]]:
        sample_period = _to_time_delta(self._period)
        if self._period == "hour":
            res = []
            sample_time = _round_up(start, self._period)
            # data_idx = 0
            while sample_time < end:
                res.append(sample_time)
                sample_time += sample_period
            return (sample_period, res)
        if self._period == "5minute":
            data = self._statistics_during_period_from_end_time(start, end)
            return (sample_period, [sample.timestamp for sample in data])
        # Fallback for unexpected period values
        return (sample_period, [])

    def _compute_samples(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> _StatisticSamples:
        sample_period, sample_datetimes = self._read_stats_and_generate_samples(
            start=start, end=end
        )
        sum_before_start = self._compute_sum_before(start)
        prev_sum_before_end = None
        if sample_datetimes:
            prev_sum_before_end = self._compute_sum_before(sample_datetimes[-1])
        _LOGGER.debug(
            "[%s] Computing %s samples. Samples to compute: %d. Sum before start: %s. Prev sum before end: %s",
            self._statistic_id,
            self._period,
            len(sample_datetimes),
            sum_before_start,
            prev_sum_before_end,
        )

        reading_idx = 0
        curr_sum = decimal.Decimal(sum_before_start)
        reset_time = None
        res = []
        for i, sample_datetime in enumerate(sample_datetimes):
            if i > 0 and i % 10000 == 0:
                _LOGGER.debug(
                    "[%s] Finished computing %d samples", self._statistic_id, i
                )
            prev_sample_datetime = sample_datetime - sample_period
            curr_value = None
            while reading_idx < len(self._interval_block.interval_readings):
                reading = self._interval_block.interval_readings[reading_idx]
                if sample_datetime <= reading.start:
                    # Sample is fully before the reading.
                    break
                reading_value = self._data_extractor.get_native_value(reading)
                reading_period = reading.end - reading.start
                scale = decimal.Decimal(
                    (sample_datetime - reading.start) / reading_period
                )
                scale = min(scale, decimal.Decimal(1))
                reset_time = reading.start
                curr_value = scale * reading_value
                if prev_sample_datetime <= reading.start:
                    curr_sum += curr_value
                else:
                    prev_value_scale = decimal.Decimal(
                        (prev_sample_datetime - reading.start) / reading_period
                    )
                    prev_value_scale = min(prev_value_scale, decimal.Decimal(1))
                    curr_sum += curr_value - (prev_value_scale * reading_value)
                if sample_datetime < reading.end:
                    break
                reading_idx += 1

            if curr_value is not None:
                prev_sample = _SensorStatRecord(
                    timestamp=sample_datetime,
                    last_reset=reset_time,
                    state=curr_value,
                    sum=curr_sum,
                )
                res.append(prev_sample)
        stat_samples = _StatisticSamples(
            prev_sum_before_end=prev_sum_before_end, samples=res
        )
        if res:
            _LOGGER.debug(
                "[%s] Computed %d %s samples. Total change: %s. Latest sample:\n%s",
                self._statistic_id,
                len(res),
                self._period,
                stat_samples.get_total_change(),
                res[-1],
            )
        else:
            _LOGGER.debug(
                "[%s] No %s samples computed", self._statistic_id, self._period
            )
        return stat_samples

    def run(self, instance: Recorder) -> None:
        start = self._interval_block.start
        end = self._interval_block.end
        samples = self._compute_samples(start=start, end=end)
        _complete_future(self._future, samples)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        statistic_id: str,
        data_extractor: DataExtractor,
        interval_block: _MergedIntervalBlock,
        period: Literal["5minute", "hour"],
    ) -> asyncio.Future[_StatisticSamples]:
        """Queue the task and return a future that completes when the task completes."""

        def ctor(
            future: asyncio.Future[_StatisticSamples],
        ) -> _ComputeUpdatedPeriodStatisticsTask:
            return cls(
                hass=hass,
                statistic_id=statistic_id,
                data_extractor=data_extractor,
                interval_block=interval_block,
                period=period,
                future=future,
            )

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _ImportStatisticsTask(tasks.RecorderTask):
    hass: HomeAssistant
    entity: GreenButtonEntity
    samples: list[StatisticData]
    table: type[recorder_db_schema.StatisticsShortTerm | recorder_db_schema.Statistics]
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        statistic_id = self.entity.long_term_statistics_id
        _LOGGER.debug(
            "[%s] Importing %d statistics samples to table '%s'",
            statistic_id,
            len(self.samples),
            self.table.__tablename__,
        )
        metadata = statistics.get_metadata(self.hass, statistic_ids={statistic_id}).get(
            statistic_id, (0, None)
        )[1]
        if metadata is None:
            metadata = create_metadata(self.entity)
        success = statistics.import_statistics(
            instance, metadata, self.samples, self.table
        )
        if not success:
            recorder_helper.get_instance(self.hass).queue_task(self)
            #  RDH recorder_util.get_instance(self.hass).queue_task(self)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        entity: GreenButtonEntity,
        samples: list[StatisticData],
        table: type[recorder_db_schema.Statistics | recorder_db_schema.StatisticsShortTerm],
    ) -> asyncio.Future[None]:
        """Queue the task and return a future that completes when the task completes."""

        def ctor(future: asyncio.Future[None]) -> _ImportStatisticsTask:
            return cls(
                hass=hass,
                entity=entity,
                samples=samples,
                table=table,
                future=future,
            )

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _AdjustStatisticsTask(tasks.RecorderTask):
    _MIN_CHANGE = decimal.Decimal(10) ** -10

    hass: HomeAssistant
    statistic_id: str
    start_time: datetime.datetime
    unit_of_measurement: str
    sum_adjustment: float
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        _LOGGER.debug(
            "[%s] Adjusting statistics after '%s' by %s %s",
            self.statistic_id,
            self.start_time,
            self.sum_adjustment,
            self.unit_of_measurement,
        )
        success = statistics.adjust_statistics(
            instance,
            self.statistic_id,
            self.start_time,
            float(self.sum_adjustment),
            self.unit_of_measurement,
        )
        if not success:
            recorder_helper.get_instance(self.hass).queue_task(self)
            #  RDH recorder_util.get_instance(self.hass).queue_task(self)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        statistic_id: str,
        start_time: datetime.datetime,
        unit_of_measurement: str,
        sum_adjustment: float,
    ) -> asyncio.Future[None]:
        """Queue the task and return a Future that completes when the task is done."""

        def ctor(future: asyncio.Future[None]) -> _AdjustStatisticsTask:
            return cls(
                hass=hass,
                statistic_id=statistic_id,
                start_time=start_time,
                unit_of_measurement=unit_of_measurement,
                sum_adjustment=sum_adjustment,
                future=future,
            )

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _ClearStatisticsTask(tasks.RecorderTask):
    hass: HomeAssistant
    statistic_id: str
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        _LOGGER.debug("[%s] Clearing statistics", self.statistic_id)
        statistics.clear_statistics(
            instance=instance, statistic_ids=[self.statistic_id]
        )
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        statistic_id: str,
    ) -> asyncio.Future[None]:
        """Queue the task and return a Future that completes when the task is done."""

        def ctor(future: asyncio.Future[None]) -> _ClearStatisticsTask:
            return cls(hass=hass, statistic_id=statistic_id, future=future)

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _TruncateStatisticsAfterTask(tasks.RecorderTask):
    """Recorder task to delete statistics at and after a cutoff time.

    This removes trailing statistics rows for a statistic_id to prevent
    leftover future bars from previous imports from appearing in charts.
    """

    hass: HomeAssistant
    statistic_id: str
    cutoff_start: datetime.datetime
    table: type[recorder_db_schema.StatisticsShortTerm | recorder_db_schema.Statistics]
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        _LOGGER.info(
            "[%s] Truncating statistics in table '%s' at and after %s",
            self.statistic_id,
            self.table.__tablename__,
            self.cutoff_start,
        )
        try:
            # Use recorder session to delete rows at and after the cutoff
            with recorder_helper.session_scope(session=instance.get_session()) as session:
                # Find metadata_id for the statistic_id
                meta = (
                    session.query(recorder_db_schema.StatisticsMeta)
                    .filter(
                        recorder_db_schema.StatisticsMeta.statistic_id
                        == self.statistic_id
                    )
                    .one_or_none()
                )
                if meta is not None:
                    (
                        session.query(self.table)
                        .filter(self.table.metadata_id == meta.id)
                        .filter(self.table.start >= self.cutoff_start)
                        .delete(synchronize_session=False)
                    )
                else:
                    _LOGGER.debug(
                        "[%s] No metadata found when truncating; nothing to delete",
                        self.statistic_id,
                    )
        except Exception:
            # Re-queue if recorder is not ready
            recorder_helper.get_instance(self.hass).queue_task(self)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        statistic_id: str,
        cutoff_start: datetime.datetime,
        table: type[recorder_db_schema.StatisticsShortTerm | recorder_db_schema.Statistics],
    ) -> asyncio.Future[None]:
        """Queue the task and return a Future that completes when the truncation is done."""

        def ctor(future: asyncio.Future[None]) -> _TruncateStatisticsAfterTask:
            return cls(
                hass=hass,
                statistic_id=statistic_id,
                cutoff_start=cutoff_start,
                table=table,
                future=future,
            )

        return _queue_task(hass, ctor)


class _UpdateStatisticsTask:
    def __init__(
        self,
        hass: HomeAssistant,
        stats_dao: _StatsDao,
        entity: GreenButtonEntity,
        data_extractor: DataExtractor,
        meter_reading: model.MeterReading,
    ) -> None:
        self._hass = hass
        self._stats_dao = stats_dao
        self._entity = entity
        self._data_extractor = data_extractor
        self._meter_reading = meter_reading

    @property
    def _statistic_id(self) -> str:
        return self._entity.long_term_statistics_id

    async def _update_statistics(
        self, interval_block: _MergedIntervalBlock, period: Literal["5minute", "hour"]
    ) -> _StatisticSamples:
        samples = await _ComputeUpdatedPeriodStatisticsTask.queue_task(
            hass=self._hass,
            statistic_id=self._statistic_id,
            data_extractor=self._data_extractor,
            interval_block=interval_block,
            period=period,
        )
        await _ImportStatisticsTask.queue_task(
            hass=self._hass,
            entity=self._entity,
            samples=[sample.to_statistics_data(period) for sample in samples.samples],
            table=_to_table(period),
        )
        # NOTE: Removed _AdjustStatisticsTask call - it was causing corruption by applying
        # adjustments to ALL future statistics, including non-existent dates.
        # The async_import_statistics API handles proper merging without needing manual adjustments.
        return samples

    async def _update_for_interval_block(
        self, interval_block: _MergedIntervalBlock
    ) -> None:
        _LOGGER.info(
            "[%s] Processing %d IntervalReadings for merged IntervalBlock from '%s' to '%s'",
            self._statistic_id,
            len(interval_block.interval_readings),
            interval_block.start,
            interval_block.end,
        )
        await self._update_statistics(interval_block, "hour")
        await self._update_statistics(interval_block, "5minute")

    async def __call__(self) -> None:
        _LOGGER.info("[%s] Updating statistics for entity", self._statistic_id)
        merged_blocks = _merge_interval_blocks(self._meter_reading.interval_blocks)
        for block in merged_blocks:
            if not _is_aligned(block.end, "hour"):
                raise UnalignedIntervalBlocksError(
                    f"Merged IntervalBlock not aligned at end time. Block ID: {repr(block.ids[-1])}. Block end: '{block.end}'"
                )
        for block in merged_blocks:
            await self._update_for_interval_block(block)
        _LOGGER.info("[%s] Statistics update complete", self._statistic_id)

    @classmethod
    def create(
        cls,
        hass: HomeAssistant,
        entity: GreenButtonEntity,
        data_extractor: DataExtractor,
        meter_reading: model.MeterReading,
    ) -> _UpdateStatisticsTask:
        """Create a new task."""
        return _UpdateStatisticsTask(
            hass=hass,
            stats_dao=_StatsDao(hass, entity.long_term_statistics_id),
            entity=entity,
            data_extractor=data_extractor,
            meter_reading=meter_reading,
        )


class UnalignedIntervalBlocksError(exceptions.HomeAssistantError):
    """An error raised when a MeterReading contains unaligned readings.

    Unaligned readings cannot be stored so has the potential to cause data
    corruption.
    """


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

        # Apply power of ten multiplier
        power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
        value = interval_reading.value * (10**power_multiplier)
        return decimal.Decimal(value)


class CostDataExtractor:
    """DataExtractor that pulls monetary cost from IntervalReading.

    Applies the ReadingType power_of_ten_multiplier to the cost value.
    For example, with multiplier -3, a cost of 90 becomes 0.09 (in major currency units).
    """

    def get_native_value(
        self, interval_reading: model.IntervalReading
    ) -> decimal.Decimal:
        """
        Calculates the native value for a given interval reading by applying the power of ten multiplier to the cost.

        Args:
            interval_reading (model.IntervalReading): The interval reading object containing cost and reading type information.

        Returns:
            decimal.Decimal: The calculated native value as a decimal, representing the cost adjusted by the power of ten multiplier.
        """
        cost = interval_reading.cost if interval_reading.cost is not None else 0
        power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
        return decimal.Decimal(cost * (10**power_multiplier))


def create_metadata(entity: GreenButtonEntity) -> StatisticMetaData:
    """Create the statistic metadata for the entity."""
    return {
        "mean_type": StatisticMeanType.NONE,
        "has_sum": True,
        "name": entity.name,
        "source": "recorder",  # Must be "recorder" - HA validates this
        "statistic_id": entity.long_term_statistics_id,
        "unit_of_measurement": entity.native_unit_of_measurement,
        "unit_class": None,
    }


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
    # Collect all readings first
    all_readings = [
        interval_reading
        for interval_block in meter_reading.interval_blocks
        for interval_reading in interval_block.interval_readings
    ]

    if not all_readings:
        return []

    # Sort readings by start time to ensure chronological order
    all_readings.sort(key=lambda r: r.start)

    # Determine cutoff at the end of the last FULL hour covered by the data
    # Any intervals ending after this cutoff are considered part of a partial hour and skipped
    last_end: datetime.datetime = max(r.end for r in all_readings)
    cutoff_end: datetime.datetime = last_end.replace(minute=0, second=0, microsecond=0)

    # If the last interval ends exactly on an hour boundary, we can include it
    # Otherwise, we skip the trailing partial hour
    include_trailing_hour = last_end == cutoff_end

    # Build hourly buckets with coverage tracking (seconds covered in the hour)
    hourly_kwh: dict[datetime.datetime, decimal.Decimal] = {}
    hourly_coverage_seconds: dict[datetime.datetime, int] = {}

    # Determine source unit from ReadingType (default to Wh if missing)
    source_unit = (
        all_readings[0].reading_type.unit_of_measurement
        if all_readings and getattr(all_readings[0].reading_type, "unit_of_measurement", None)
        else UnitOfEnergy.WATT_HOUR
    )

    for reading in all_readings:
        # Skip intervals that end after the cutoff if we don't include trailing hour
        if not include_trailing_hour and reading.end > cutoff_end:
            # If the reading overlaps the cutoff boundary, trim to cutoff
            if reading.start < cutoff_end < reading.end:
                # Split proportionally: keep portion up to cutoff
                total_seconds = (reading.end - reading.start).total_seconds()
                kept_seconds = (cutoff_end - reading.start).total_seconds()
                if total_seconds > 0:
                    proportion = decimal.Decimal(kept_seconds / total_seconds)
                else:
                    proportion = decimal.Decimal(0)
                base_value = data_extractor.get_native_value(reading)
                value_kwh = decimal.Decimal(
                    _convert_to_kwh(float(base_value), source_unit)
                )
                kept_kwh = value_kwh * proportion
                # Bucket kept portion into hour of cutoff_end - 1 hour
                hour_start = (cutoff_end - datetime.timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
                hourly_kwh[hour_start] = hourly_kwh.get(hour_start, decimal.Decimal(0)) + kept_kwh
                hourly_coverage_seconds[hour_start] = hourly_coverage_seconds.get(hour_start, 0) + int(kept_seconds)
            # Skip the remainder
            continue

        # Potentially split a reading that spans multiple hours
        curr_start = reading.start
        curr_end = reading.end
        base_value = data_extractor.get_native_value(reading)
        value_kwh_total = decimal.Decimal(
            _convert_to_kwh(float(base_value), source_unit)
        )
        total_seconds = (curr_end - curr_start).total_seconds()
        if total_seconds <= 0:
            continue

        # Iterate across hours, splitting proportionally
        while curr_start < curr_end:
            hour_start = curr_start.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + datetime.timedelta(hours=1)
            segment_end = min(curr_end, hour_end)
            seg_seconds = (segment_end - curr_start).total_seconds()
            proportion = decimal.Decimal(seg_seconds / total_seconds)
            seg_kwh = value_kwh_total * proportion

            hourly_kwh[hour_start] = hourly_kwh.get(hour_start, decimal.Decimal(0)) + seg_kwh
            hourly_coverage_seconds[hour_start] = hourly_coverage_seconds.get(hour_start, 0) + int(seg_seconds)

            curr_start = segment_end

    # Build new statistics for FULLY covered hours only (3600s)
    hour_keys_sorted = sorted(hourly_kwh.keys())
    if not hour_keys_sorted:
        return []

    new_statistics_data: list[StatisticData] = []
    skipped_count = 0
    for hour_start in hour_keys_sorted:
        coverage = hourly_coverage_seconds.get(hour_start, 0)
        if coverage < 3600:
            _LOGGER.debug(
                "Skipping partial hour starting %s (covered %ds)",
                hour_start,
                coverage,
            )
            skipped_count += 1
            continue
        hour_kwh = float(hourly_kwh.get(hour_start, decimal.Decimal(0)))
        stat_record: StatisticData = {
            "start": hour_start,
            "state": hour_kwh,
            "sum": 0.0,  # Will be calculated during merge
        }
        new_statistics_data.append(stat_record)

    if not new_statistics_data:
        _LOGGER.info(
            "No complete hourly statistics generated for entity %s (skipped %d partial hours)",
            entity.entity_id,
            skipped_count,
        )
        return []

    # Get all existing statistics for this entity
    existing_stats = await _get_all_existing_statistics(
        hass,
        entity.long_term_statistics_id,
    )

    # Merge new statistics with existing ones, handling out-of-order imports
    merged_stats = _merge_statistics_with_out_of_order_support(
        existing_stats,
        new_statistics_data,
        entity.long_term_statistics_id,
    )

    # Log summary of what was processed
    _LOGGER.debug(
        "Generated %d hourly statistics for entity %s (skipped %d partial hours, existing: %d, merged result: %d)",
        len(new_statistics_data),
        entity.entity_id,
        skipped_count,
        len(existing_stats),
        len(merged_stats),
    )
    if merged_stats:
        _LOGGER.info(
            "Statistics range: %s (sum=%.3f) to %s (sum=%.3f)",
            merged_stats[0]["start"],
            merged_stats[0].get("sum", 0.0),
            merged_stats[-1]["start"],
            merged_stats[-1].get("sum", 0.0),
        )

    return merged_stats


async def _generate_statistics_data_cost(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
) -> list[StatisticData]:
    """Generate hourly cost statistics with out-of-order import support.

    Mirrors the energy statistics generation but uses monetary cost per interval
    without applying energy unit conversions.
    """
    # Collect all readings first
    all_readings = [
        interval_reading
        for interval_block in meter_reading.interval_blocks
        for interval_reading in interval_block.interval_readings
    ]

    if not all_readings:
        _LOGGER.warning(
            "No interval readings found in meter reading %s for cost statistics",
            meter_reading.id,
        )
        return []

    _LOGGER.debug(
        "Cost statistics: Collected %d interval readings for meter reading %s",
        len(all_readings),
        meter_reading.id,
    )

    # Sort readings by start time
    all_readings.sort(key=lambda r: r.start)

    # Determine cutoff at the end of the last FULL hour covered by the data
    last_end: datetime.datetime = max(r.end for r in all_readings)
    cutoff_end: datetime.datetime = last_end.replace(minute=0, second=0, microsecond=0)
    include_trailing_hour = last_end == cutoff_end
    
    _LOGGER.debug(
        "Cost statistics: Last end time = %s, cutoff_end = %s, include_trailing_hour = %s",
        last_end,
        cutoff_end,
        include_trailing_hour,
    )

    # Build hourly buckets with coverage tracking
    hourly_cost: dict[datetime.datetime, decimal.Decimal] = {}
    hourly_coverage_seconds: dict[datetime.datetime, int] = {}

    for reading in all_readings:
        # Skip intervals that end after the cutoff if not including trailing hour
        if not include_trailing_hour and reading.end > cutoff_end:
            if reading.start < cutoff_end < reading.end:
                # Keep portion up to cutoff
                total_seconds = (reading.end - reading.start).total_seconds()
                kept_seconds = (cutoff_end - reading.start).total_seconds()
                if total_seconds > 0:
                    proportion = decimal.Decimal(kept_seconds / total_seconds)
                else:
                    proportion = decimal.Decimal(0)
                base_value = data_extractor.get_native_value(reading)
                kept_val = base_value * proportion
                hour_start = (cutoff_end - datetime.timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
                hourly_cost[hour_start] = hourly_cost.get(
                    hour_start, decimal.Decimal(0)
                ) + kept_val
                hourly_coverage_seconds[hour_start] = hourly_coverage_seconds.get(
                    hour_start, 0
                ) + int(kept_seconds)
            continue

        # Split a reading that may span multiple hours proportionally
        curr_start = reading.start
        curr_end = reading.end
        base_value_total = data_extractor.get_native_value(reading)
        total_seconds = (curr_end - curr_start).total_seconds()
        if total_seconds <= 0:
            continue

        while curr_start < curr_end:
            hour_start = curr_start.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + datetime.timedelta(hours=1)
            segment_end = min(curr_end, hour_end)
            seg_seconds = (segment_end - curr_start).total_seconds()
            proportion = decimal.Decimal(seg_seconds / total_seconds)
            seg_val = base_value_total * proportion

            hourly_cost[hour_start] = hourly_cost.get(
                hour_start, decimal.Decimal(0)
            ) + seg_val
            hourly_coverage_seconds[hour_start] = hourly_coverage_seconds.get(
                hour_start, 0
            ) + int(seg_seconds)

            curr_start = segment_end

    # Build new statistics for FULLY covered hours only (3600s)
    hour_keys_sorted = sorted(hourly_cost.keys())
    if not hour_keys_sorted:
        _LOGGER.warning(
            "Cost statistics: No hourly buckets created for entity %s",
            entity.entity_id,
        )
        return []

    _LOGGER.debug(
        "Cost statistics: Created %d hourly cost buckets before filtering",
        len(hour_keys_sorted),
    )

    new_statistics_data: list[StatisticData] = []
    skipped_count = 0
    for hour_start in hour_keys_sorted:
        coverage = hourly_coverage_seconds.get(hour_start, 0)
        if coverage < 3600:
            _LOGGER.debug(
                "Cost statistics: Skipping partial hour starting %s (covered %ds)",
                hour_start,
                coverage,
            )
            skipped_count += 1
            continue
        hour_val = float(hourly_cost.get(hour_start, decimal.Decimal(0)))
        stat_record: StatisticData = {
            "start": hour_start,
            "state": hour_val,
            "sum": 0.0,  # Will be calculated during merge
        }
        new_statistics_data.append(stat_record)

    _LOGGER.info(
        "Cost statistics: Generated %d complete hourly records (skipped %d partial hours) for entity %s",
        len(new_statistics_data),
        skipped_count,
        entity.entity_id,
    )

    if not new_statistics_data:
        _LOGGER.info(
            "No complete hourly cost statistics generated for entity %s (skipped %d partial hours)",
            entity.entity_id,
            skipped_count,
        )
        return []

    # Get all existing statistics for this entity
    existing_stats = await _get_all_existing_statistics(
        hass,
        entity.long_term_statistics_id,
    )

    # Merge new statistics with existing ones, handling out-of-order imports
    merged_stats = _merge_statistics_with_out_of_order_support(
        existing_stats,
        new_statistics_data,
        entity.long_term_statistics_id,
    )

    # Log summary of what was processed
    _LOGGER.info(
        "Generated %d hourly cost statistics for entity %s (skipped %d partial hours, existing: %d, merged result: %d)",
        len(new_statistics_data),
        entity.entity_id,
        skipped_count,
        len(existing_stats),
        len(merged_stats),
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


async def update_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    data_extractor: DataExtractor,
    meter_reading: model.MeterReading,
) -> None:
    """Update the cost statistics for an entry to match the MeterReading."""
    metadata = create_metadata(entity)
    _LOGGER.info(
        "Starting cost statistics generation for entity %s, meter reading %s",
        entity.entity_id,
        meter_reading.id,
    )
    statistics_data = await _generate_statistics_data_cost(
        hass, entity, data_extractor, meter_reading
    )

    _LOGGER.info(
        "Generated %d cost statistics records for entity %s",
        len(statistics_data),
        entity.entity_id,
    )

    if statistics_data:
        # Log first and last records for debugging
        first_record = statistics_data[0]
        last_record = statistics_data[-1]
        _LOGGER.info(
            "Cost statistics range: %s (sum=%s) to %s (sum=%s)",
            first_record["start"],
            first_record.get("sum"),
            last_record["start"],
            last_record.get("sum"),
        )

        # Clear all existing statistics and reimport with the merged data
        # This ensures the database is consistent with our calculated sums
        try:
            await _ClearStatisticsTask.queue_task(
                hass=hass,
                statistic_id=entity.long_term_statistics_id,
            )
            _LOGGER.info(
                "Cleared existing cost statistics for entity %s before reimporting",
                entity.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "Failed to clear cost statistics for entity %s",
                entity.entity_id,
            )

        # Import cost statistics using the proper Home Assistant API
        try:
            async_import_statistics(hass, metadata, statistics_data)
            _LOGGER.info(
                "✅ Imported %d cost records for entity %s",
                len(statistics_data),
                entity.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "❌ Failed to import cost statistics for entity %s",
                entity.entity_id,
            )


async def update_statistics(
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

    if statistics_data:
        # Log first and last records for debugging
        first_record = statistics_data[0]
        last_record = statistics_data[-1]
        _LOGGER.info(
            "Statistics range: %s (sum=%s) to %s (sum=%s)",
            first_record["start"],
            first_record.get("sum"),
            last_record["start"],
            last_record.get("sum"),
        )

        # Clear all existing statistics and reimport with the merged data
        # This ensures the database is consistent with our calculated sums
        try:
            await _ClearStatisticsTask.queue_task(
                hass=hass,
                statistic_id=entity.long_term_statistics_id,
            )
            _LOGGER.info(
                "Cleared existing statistics for entity %s before reimporting",
                entity.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "Failed to clear statistics for entity %s",
                entity.entity_id,
            )

        # Import historical statistics using the proper Home Assistant API
        try:
            async_import_statistics(hass, metadata, statistics_data)
            _LOGGER.debug(
                "✅ Successfully called async_import_statistics with %d records for entity %s",
                len(statistics_data),
                entity.entity_id,
            )

            # Log some sample records for debugging
            if statistics_data:
                first = statistics_data[0]
                last = statistics_data[-1]
                _LOGGER.info(
                    "📊 Statistics range: %s (state=%.3f, sum=%.3f) → %s (state=%.3f, sum=%.3f)",
                    first["start"].isoformat(),
                    first.get("state", 0.0),
                    first.get("sum", 0.0),
                    last["start"].isoformat(),
                    last.get("state", 0.0),
                    last.get("sum", 0.0),
                )
        except Exception:
            _LOGGER.exception(
                "❌ Failed to import statistics for entity %s",
                entity.entity_id,
            )
    else:
        _LOGGER.warning(
            "No statistics data generated for entity %s",
            entity.entity_id,
        )


async def clear_statistic(hass: HomeAssistant, statistic_id: str) -> None:
    """Clear all statistics with the specified ID."""
    await _ClearStatisticsTask.queue_task(hass=hass, statistic_id=statistic_id)


# -------------------- GAS (m³) DAILY STATISTICS --------------------

async def _generate_daily_m3_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading,
) -> list[StatisticData]:
    """Generate daily statistics for gas consumption (m³) with out-of-order import support.

    We emit one hourly record per day at 00:00 with the day's total m³ as state.
    """
    # Flatten all interval readings
    readings = [
        r
        for block in meter_reading.interval_blocks
        for r in block.interval_readings
    ]
    if not readings:
        return []

    # Sort readings by start
    readings.sort(key=lambda r: r.start)
    # Expect daily intervals; compute daily totals in native m³
    daily_totals: dict[datetime.date, float] = {}
    for rd in readings:
        # Apply multiplier
        val = float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
        day = rd.start.date()
        daily_totals[day] = daily_totals.get(day, 0.0) + val

    if not daily_totals:
        return []

    # Build new statistics data
    new_statistics_data: list[StatisticData] = []
    for day in sorted(daily_totals.keys()):
        day_val = daily_totals[day]
        start = datetime.datetime.combine(day, datetime.time.min, tzinfo=readings[0].start.tzinfo)
        new_statistics_data.append({
            "start": start,
            "state": day_val,
            "sum": 0.0,  # Will be calculated during merge
        })

    if not new_statistics_data:
        return []

    # Get all existing statistics for this entity
    existing_stats = await _get_all_existing_statistics(
        hass,
        entity.long_term_statistics_id,
    )

    # Merge new statistics with existing ones, handling out-of-order imports
    merged_stats = _merge_statistics_with_out_of_order_support(
        existing_stats,
        new_statistics_data,
        entity.long_term_statistics_id,
    )

    return merged_stats


async def update_gas_statistics(
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
        # Clear existing statistics to avoid residual daily bars or negative corrections
        try:
            await clear_statistic(hass, entity.long_term_statistics_id)
        except Exception:
            _LOGGER.exception("Failed to clear existing gas usage stats for %s", entity.entity_id)

        # Determine tzinfo from readings if present
        if meter_reading and meter_reading.interval_blocks:
            readings = [r for b in meter_reading.interval_blocks for r in b.interval_readings]
            tzinfo = readings[0].start.tzinfo if readings else datetime.timezone.utc
        else:
            # No meter reading available - use UTC as default
            tzinfo = datetime.timezone.utc
            readings = []

        # Build a list of billing periods from both UsageSummaries and long IntervalReadings
        # This handles the case where Enbridge provides:
        # - UsageSummary for previous finalized billing period
        # - IntervalReading for current billing period (not yet finalized)
        periods_to_process: list[tuple[datetime.datetime, datetime.datetime, float | None, str]] = []

        # Add all UsageSummaries
        for us in summaries:
            period_start = us.start
            period_end = us.start + us.duration
            consumption_m3 = float(us.consumption_m3) if (hasattr(us, "consumption_m3") and us.consumption_m3 is not None) else None
            source = f"UsageSummary:{us.id}"
            periods_to_process.append((period_start, period_end, consumption_m3, source))

        # Check for long IntervalReadings (>7 days) that might represent billing periods
        # not yet in UsageSummary
        MIN_BILLING_PERIOD_DAYS = 7
        for rd in readings:
            duration_days = rd.duration.total_seconds() / 86400
            if duration_days >= MIN_BILLING_PERIOD_DAYS:
                rd_start = rd.start
                rd_end = rd.start + rd.duration
                # Check if this period overlaps significantly with any UsageSummary
                overlaps = False
                for us in summaries:
                    us_start = us.start
                    us_end = us.start + us.duration
                    # Check for significant overlap (>50% of either period)
                    overlap_start = max(rd_start, us_start)
                    overlap_end = min(rd_end, us_end)
                    if overlap_start < overlap_end:
                        overlap_days = (overlap_end - overlap_start).total_seconds() / 86400
                        if overlap_days > min(duration_days, (us_end - us_start).total_seconds() / 86400) * 0.5:
                            overlaps = True
                            break

                if not overlaps:
                    # This is a billing-period-length reading not covered by UsageSummary
                    consumption_m3 = float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
                    source = f"IntervalReading:{rd_start.isoformat()}"
                    periods_to_process.append((rd_start, rd_end, consumption_m3, source))
                    _LOGGER.info(
                        "Found billing period from IntervalReading not in UsageSummary: %s to %s (%.1f m³)",
                        rd_start.date(), rd_end.date(), consumption_m3
                    )

        if not periods_to_process:
            _LOGGER.info("No billing periods available for monthly gas usage on %s", entity.entity_id)
            return

        # Sort by period end
        periods_to_process.sort(key=lambda p: p[1])
        records: list[StatisticData] = []
        first_start: datetime.datetime | None = None
        existing_sum = 0.0

        for period_start, period_end, consumption_m3, source in periods_to_process:
            # Place the increment at 00:00 of the period end date (the day the period ends)
            rec_start = datetime.datetime.combine(period_end.date(), datetime.time.min, tzinfo=tzinfo)
            if first_start is None:
                first_start = rec_start
                # Existing cumulative before first record
                try:
                    rec = recorder_helper.get_instance(hass)
                    existing_stats = await rec.async_add_executor_job(
                        statistics.statistic_during_period,
                        hass,
                        None,
                        first_start,
                        entity.long_term_statistics_id,
                        {"change"},
                        None,
                    )
                    if existing_stats and existing_stats.get("change") is not None:
                        existing_sum = float(existing_stats["change"])  # type: ignore[assignment]
                except Exception:
                    _LOGGER.debug("Unable to query existing sum for gas usage %s", entity.entity_id, exc_info=True)

            # Use the consumption_m3 if available, otherwise fallback to summing daily readings
            period_m3 = consumption_m3
            if period_m3 is None:
                # Fallback: sum any daily readings within this period if present
                period_days: list[datetime.date] = []
                day = period_start.date()
                end_day = (period_end - datetime.timedelta(seconds=1)).date()
                while day <= end_day:
                    period_days.append(day)
                    day = day + datetime.timedelta(days=1)
                # Build a daily map from readings
                daily_m3: dict[datetime.date, float] = {}
                for rd in readings:
                    d = rd.start.date()
                    if d < period_days[0] or d > period_days[-1]:
                        continue
                    val = float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
                    daily_m3[d] = daily_m3.get(d, 0.0) + val
                total = sum(daily_m3.values())
                period_m3 = total if total > 0 else None

            if period_m3 is None or period_m3 <= 0:
                continue

            records.append({"start": rec_start, "state": period_m3, "sum": 0.0})

        if not records:
            _LOGGER.warning(
                "Gas %s: No gas usage records generated - all periods had no consumption data",
                entity.entity_id,
            )
            return

        # Apply cumulative
        cumulative = existing_sum
        for recd in records:
            cumulative += recd.get("state", 0.0)
            recd["sum"] = cumulative

        # Truncate and import: choose earliest cutoff between our first record and
        # the earliest reading day (to ensure any previously-imported daily m³ are removed)
        cutoff = records[0]["start"]
        try:
            if readings:
                earliest_day = min(r.start.date() for r in readings)
                tzinfo_cut = readings[0].start.tzinfo
                earliest_midnight = datetime.datetime.combine(earliest_day, datetime.time.min, tzinfo=tzinfo_cut)
                if earliest_midnight < cutoff:
                    cutoff = earliest_midnight
            await _TruncateStatisticsAfterTask.queue_task(
                hass=hass,
                statistic_id=entity.long_term_statistics_id,
                cutoff_start=cutoff,
                table=recorder_db_schema.Statistics,
            )
        except Exception:
            _LOGGER.exception("Failed truncating gas stats for %s", entity.entity_id)

        try:
            async_import_statistics(hass, metadata, records)
            _LOGGER.info(
                "Imported %d gas usage records for %s (total: %.1f m³)",
                len(records),
                entity.entity_id,
                records[-1].get("sum", 0.0),
            )
        except Exception:
            _LOGGER.exception("Failed to import gas stats for %s", entity.entity_id)

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

    # Clear all existing statistics and reimport with the merged data
    try:
        await _ClearStatisticsTask.queue_task(
            hass=hass,
            statistic_id=entity.long_term_statistics_id,
        )
        _LOGGER.info(
            "Cleared existing gas statistics for entity %s before reimporting",
            entity.entity_id,
        )
    except Exception:
        _LOGGER.exception("Failed to clear gas stats for %s", entity.entity_id)

    try:
        async_import_statistics(hass, metadata, data)
        _LOGGER.info("Imported %d gas daily records for %s", len(data), entity.entity_id)
    except Exception:
        _LOGGER.exception("Failed to import gas stats for %s", entity.entity_id)


async def update_gas_cost_statistics(
    hass: HomeAssistant,
    entity: GreenButtonEntity,
    meter_reading: model.MeterReading | None,
    usage_summaries: list[model.UsageSummary],
    allocation_mode: str = "pro_rate_daily",
) -> None:
    """Import pro-rated daily gas costs based on UsageSummary totals and daily m³.

    For each billing period, distribute total_cost across days proportionally
    to daily consumption in m³. Emit one hourly record per day at 00:00.
    """
    metadata = create_metadata(entity)
    
    if allocation_mode == "monthly_increment":
        # One increment per usage summary at the period end (00:00 of end day)
        if not usage_summaries:
            _LOGGER.info("No usage summaries for monthly gas cost on %s", entity.entity_id)
            return
        # Determine tzinfo from readings if available; otherwise UTC
        readings = []
        if meter_reading:
            readings = [r for b in meter_reading.interval_blocks for r in b.interval_readings]
        tzinfo = readings[0].start.tzinfo if readings else datetime.timezone.utc

        # Build records sorted by period end
        summaries_sorted = sorted(usage_summaries, key=lambda s: s.start + s.duration)
        new_statistics_data: list[StatisticData] = []
        for us in summaries_sorted:
            period_end = us.start + us.duration
            # Place the increment at 00:00 of the period end date (the day the period ends)
            rec_start = datetime.datetime.combine(period_end.date(), datetime.time.min, tzinfo=tzinfo)
            new_statistics_data.append({
                "start": rec_start,
                "state": float(us.total_cost),
                "sum": 0.0,  # Will be calculated during merge
            })

        if not new_statistics_data:
            return

        # Get all existing statistics for this entity
        existing_stats = await _get_all_existing_statistics(
            hass,
            entity.long_term_statistics_id,
        )

        # Merge new statistics with existing ones, handling out-of-order imports
        records = _merge_statistics_with_out_of_order_support(
            existing_stats,
            new_statistics_data,
            entity.long_term_statistics_id,
        )

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
        readings = [
            r
            for block in meter_reading.interval_blocks
            for r in block.interval_readings
        ]
        readings.sort(key=lambda r: r.start)
        if not readings:
            _LOGGER.info("No gas readings for cost distribution on %s", entity.entity_id)
            return
        tzinfo = readings[0].start.tzinfo
        daily_m3: dict[datetime.date, float] = {}
        for rd in readings:
            val = float(rd.value) * (10 ** rd.reading_type.power_of_ten_multiplier)
            day = rd.start.date()
            daily_m3[day] = daily_m3.get(day, 0.0) + val

        if not usage_summaries:
            _LOGGER.info("No gas usage summaries provided for %s", entity.entity_id)
            return

        # Build daily cost allocations
        daily_cost: dict[datetime.date, float] = {}
        for us in usage_summaries:
            period_start = us.start
            period_end = us.start + us.duration
            # Collect days in period
            day = period_start.date()
            end_day = (period_end - datetime.timedelta(seconds=1)).date()
            period_days: list[datetime.date] = []
            while day <= end_day:
                period_days.append(day)
                day = day + datetime.timedelta(days=1)
            # Sum m3 in period
            total_m3 = sum(daily_m3.get(d, 0.0) for d in period_days)
            if total_m3 <= 0:
                # Even split if no consumption data
                per_day = float(us.total_cost) / max(1, len(period_days))
                for d in period_days:
                    daily_cost[d] = daily_cost.get(d, 0.0) + per_day
            else:
                for d in period_days:
                    frac = daily_m3.get(d, 0.0) / total_m3
                    daily_cost[d] = daily_cost.get(d, 0.0) + (float(us.total_cost) * frac)

        if not daily_cost:
            _LOGGER.info("No daily cost allocations computed for %s", entity.entity_id)
            return

        # Build new statistics data
        new_statistics_data = []
        for d in sorted(daily_cost.keys()):
            val = daily_cost[d]
            start = datetime.datetime.combine(d, datetime.time.min, tzinfo=tzinfo)
            new_statistics_data.append({
                "start": start,
                "state": val,
                "sum": 0.0,  # Will be calculated during merge
            })

        # Get all existing statistics for this entity
        existing_stats = await _get_all_existing_statistics(
            hass,
            entity.long_term_statistics_id,
        )

        # Merge new statistics with existing ones, handling out-of-order imports
        records = _merge_statistics_with_out_of_order_support(
            existing_stats,
            new_statistics_data,
            entity.long_term_statistics_id,
        )

    if not records:
        return

    # Clear all existing statistics and reimport with the merged data
    try:
        await _ClearStatisticsTask.queue_task(
            hass=hass,
            statistic_id=entity.long_term_statistics_id,
        )
        _LOGGER.info(
            "Cleared existing gas cost statistics for entity %s before reimporting",
            entity.entity_id,
        )
    except Exception:
        _LOGGER.exception("Failed to clear gas cost stats for %s", entity.entity_id)

    try:
        async_import_statistics(hass, metadata, records)
        _LOGGER.info("Imported %d gas cost records for %s", len(records), entity.entity_id)
    except Exception:
        _LOGGER.exception("Failed to import gas cost stats for %s", entity.entity_id)
