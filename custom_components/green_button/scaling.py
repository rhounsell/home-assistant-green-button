"""Power-of-ten scaling helpers for Green Button measurements."""

from __future__ import annotations

import decimal
from typing import Any

from . import model


def configured_multiplier(config_entry: Any, key: str, default: int) -> int:
    """Return an option override, then entry data, then the integration default."""
    value = config_entry.options.get(key)
    if value is None:
        value = config_entry.data.get(key)
    return int(value) if value is not None else default


def resolve_multiplier(declared: int | None, fallback: int) -> int:
    """Prefer the XML multiplier and use configured fallback only when absent."""
    return declared if declared is not None else fallback


def interval_value(
    reading: model.IntervalReading, fallback_multiplier: int = 0
) -> decimal.Decimal:
    """Scale a raw interval value using its ReadingType multiplier."""
    multiplier = resolve_multiplier(
        reading.reading_type.power_of_ten_multiplier, fallback_multiplier
    )
    return decimal.Decimal(reading.value) * decimal.Decimal(10) ** multiplier


def interval_cost(
    reading: model.IntervalReading, fallback_multiplier: int
) -> decimal.Decimal:
    """Scale a raw interval cost using its ReadingType multiplier."""
    if reading.cost is None:
        raise ValueError("Interval cost is missing")
    multiplier = resolve_multiplier(
        reading.reading_type.power_of_ten_multiplier, fallback_multiplier
    )
    return decimal.Decimal(reading.cost) * decimal.Decimal(10) ** multiplier


def usage_summary_cost(
    summary: model.UsageSummary, fallback_multiplier: int
) -> decimal.Decimal:
    """Scale a raw usage-summary cost using its declared multiplier."""
    if summary.total_cost is None:
        raise ValueError("Usage summary cost is missing")
    multiplier = resolve_multiplier(
        summary.power_of_ten_multiplier, fallback_multiplier
    )
    return decimal.Decimal(str(summary.total_cost)) * decimal.Decimal(10) ** multiplier


def usage_summary_consumption(summary: model.UsageSummary) -> decimal.Decimal:
    """Scale a raw usage-summary volume using its declared multiplier."""
    if summary.consumption_m3 is None:
        raise ValueError("Usage summary consumption is missing")
    multiplier = resolve_multiplier(summary.consumption_power_of_ten_multiplier, 0)
    return (
        decimal.Decimal(str(summary.consumption_m3)) * decimal.Decimal(10) ** multiplier
    )
