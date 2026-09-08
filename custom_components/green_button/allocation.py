"""Pure interval allocation helpers for Green Button historical statistics."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import datetime
import decimal
from typing import Any

from . import model

Decimal = decimal.Decimal
ZERO = Decimal(0)
SECONDS_PER_HOUR = Decimal(3600)


def decimal_seconds(duration: datetime.timedelta) -> Decimal:
    """Return a duration in seconds without converting through float."""
    return Decimal(duration.days * 86400 + duration.seconds) + Decimal(
        duration.microseconds
    ) / Decimal(1_000_000)


def interval_readings(
    meter_reading: model.MeterReading | None,
) -> list[model.IntervalReading]:
    """Flatten and chronologically order a meter stream."""
    if meter_reading is None:
        return []
    return sorted(
        (
            reading
            for block in meter_reading.interval_blocks
            for reading in block.interval_readings
        ),
        key=lambda reading: reading.start,
    )


def energy_to_kwh(value: Decimal, source_unit: Any) -> Decimal:
    """Convert a source energy value to kWh while retaining Decimal precision."""
    unit = str(source_unit).lower()
    if unit in {"wh", "watt-hour"}:
        return value / Decimal(1000)
    if unit in {"kwh", "kilowatt-hour"}:
        return value
    if unit in {"mwh", "megawatt-hour"}:
        return value * Decimal(1000)
    raise ValueError(f"Unsupported energy unit: {source_unit!r}")


def hourly_values(
    readings: Sequence[model.IntervalReading],
    value_for_reading: Callable[[model.IntervalReading], Decimal],
) -> dict[datetime.datetime, Decimal]:
    """Allocate complete UTC hours from intervals using exact duration ratios.

    The final incomplete hour is intentionally held until later source data
    completes it. A zero-valued measurement remains a represented hour.
    """
    if not readings:
        return {}

    last_end = max(reading.end for reading in readings)
    cutoff = last_end.replace(minute=0, second=0, microsecond=0)
    include_cutoff = last_end == cutoff
    values: dict[datetime.datetime, Decimal] = {}
    coverage: dict[datetime.datetime, Decimal] = {}

    for reading in readings:
        total_seconds = decimal_seconds(reading.duration)
        if total_seconds <= ZERO:
            continue
        cursor = reading.start
        end = reading.end if include_cutoff else min(reading.end, cutoff)
        if cursor >= end:
            continue
        value = value_for_reading(reading)
        while cursor < end:
            hour = cursor.replace(minute=0, second=0, microsecond=0)
            segment_end = min(end, hour + datetime.timedelta(hours=1))
            segment_seconds = decimal_seconds(segment_end - cursor)
            values[hour] = (
                values.get(hour, ZERO) + value * segment_seconds / total_seconds
            )
            coverage[hour] = coverage.get(hour, ZERO) + segment_seconds
            cursor = segment_end

    return {
        hour: values[hour]
        for hour in sorted(values)
        if coverage[hour] >= SECONDS_PER_HOUR
    }


def local_day_values(
    readings: Iterable[model.IntervalReading],
    value_for_reading: Callable[[model.IntervalReading], Decimal],
    time_zone: datetime.tzinfo,
    range_start: datetime.datetime | None = None,
    range_end: datetime.datetime | None = None,
) -> dict[datetime.date, Decimal]:
    """Allocate interval values to local days by physical overlap duration."""
    totals: dict[datetime.date, Decimal] = {}
    for reading in readings:
        total_seconds = decimal_seconds(reading.duration)
        if total_seconds <= ZERO:
            continue
        start = max(reading.start, range_start) if range_start else reading.start
        end = min(reading.end, range_end) if range_end else reading.end
        if start >= end:
            continue
        value = value_for_reading(reading)
        cursor = start
        while cursor < end:
            local_cursor = cursor.astimezone(time_zone)
            next_midnight = datetime.datetime.combine(
                local_cursor.date() + datetime.timedelta(days=1),
                datetime.time.min,
                tzinfo=time_zone,
            ).astimezone(datetime.UTC)
            segment_end = min(end, next_midnight)
            segment_seconds = decimal_seconds(segment_end - cursor)
            day = local_cursor.date()
            totals[day] = (
                totals.get(day, ZERO) + value * segment_seconds / total_seconds
            )
            cursor = segment_end
    return totals


def local_days(
    start: datetime.datetime, end: datetime.datetime, time_zone: datetime.tzinfo
) -> list[datetime.date]:
    """Return each local calendar date with physical overlap in an interval."""
    if start >= end:
        return []
    first = start.astimezone(time_zone).date()
    last = (end - datetime.timedelta.resolution).astimezone(time_zone).date()
    return [
        first + datetime.timedelta(days=offset)
        for offset in range((last - first).days + 1)
    ]


def prorated_cost_by_day(
    readings: Sequence[model.IntervalReading],
    periods: Iterable[tuple[datetime.datetime, datetime.datetime, Decimal]],
    value_for_reading: Callable[[model.IntervalReading], Decimal],
    time_zone: datetime.tzinfo,
) -> tuple[dict[datetime.date, Decimal], bool]:
    """Allocate billing totals by available measured consumption.

    A period with gaps allocates its whole bill to measured days. That is an
    estimate, represented by the returned flag. A period without any positive
    consumption is evenly estimated over its local calendar days.
    """
    costs: dict[datetime.date, Decimal] = {}
    estimated = False
    for start, end, cost in periods:
        days = local_days(start, end, time_zone)
        consumption = local_day_values(
            readings, value_for_reading, time_zone, start, end
        )
        total = sum(consumption.values(), ZERO)
        estimated |= len(consumption) != len(days)
        if total > ZERO:
            for day, value in consumption.items():
                costs[day] = costs.get(day, ZERO) + cost * value / total
        elif days:
            estimated = True
            share = cost / Decimal(len(days))
            for day in days:
                costs[day] = costs.get(day, ZERO) + share
    return costs, estimated


def cumulative_records(
    values: Mapping[datetime.datetime, Decimal],
    initial_sum: Decimal = ZERO,
) -> list[dict[str, float | datetime.datetime]]:
    """Serialize ordered Decimal increments to recorder-compatible records."""
    total = initial_sum
    records: list[dict[str, float | datetime.datetime]] = []
    for start in sorted(values):
        state = values[start]
        total += state
        records.append({"start": start, "state": float(state), "sum": float(total)})
    return records


def merge_records(
    existing: Sequence[Mapping[str, Any]],
    values: Mapping[datetime.datetime, Decimal],
) -> list[dict[str, float | datetime.datetime]]:
    """Replace source-owned suffix records and recalculate cumulative sums."""
    if not values:
        return []
    first_new_start = min(values)
    retained = [record for record in existing if record["start"] < first_new_start]
    baseline = Decimal(str(retained[-1].get("sum", 0))) if retained else ZERO
    records = [
        {
            "start": record["start"],
            "state": float(record.get("state", 0)),
            "sum": float(record.get("sum", 0)),
        }
        for record in retained
    ]
    records.extend(cumulative_records(values, baseline))
    return records
