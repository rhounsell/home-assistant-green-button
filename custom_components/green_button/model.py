"""Package containing data classes representing measurements.

Based on the Energy Services Provider Interface (ESPI) Atom feed defined by the
North American Energy Standards Board.
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
from collections.abc import Collection
from collections.abc import Sequence
from homeassistant.components import sensor
from typing import final


@final
@functools.total_ordering
@dataclasses.dataclass(frozen=True)
class IntervalReading:
    """An object representing a specific meter reading over some time interval."""

    reading_type: ReadingType
    cost: int
    start: datetime.datetime
    duration: datetime.timedelta
    value: int

    def __lt__(self, other: IntervalBlock) -> bool:
        """Return whether or not this reading's start time is before the other's."""
        return self.start < other.start

    @property
    def end(self) -> datetime.datetime:
        """Return the reading interval's end time."""
        return self.start + self.duration


@final
@functools.total_ordering
@dataclasses.dataclass(frozen=True)
class IntervalBlock:
    """A collection of IntervalReadings."""

    id: str
    reading_type: ReadingType
    start: datetime.datetime
    duration: datetime.timedelta
    interval_readings: list[IntervalReading]

    def __post_init__(self):
        """Post-process the data."""
        object.__setattr__(self, "interval_readings", sorted(self.interval_readings))

    def __lt__(self, other: IntervalBlock) -> bool:
        """Return whether or not this block's start time is before the other's."""
        return self.start < other.start

    @property
    def end(self) -> datetime.datetime:
        """Return the block's interval's end time."""
        return self.start + self.duration

    def get_newest_interval_reading(self) -> IntervalReading | None:
        """Return the most recent IntervalReading in the block."""
        if not self.interval_readings:
            return None
        return self.interval_readings[len(self.interval_readings) - 1]


@final
@dataclasses.dataclass(frozen=True)
class ReadingType:
    """A object describing metadata about the meter readings."""

    id: str
    currency: str
    power_of_ten_multiplier: int
    unit_of_measurement: str


@final
@dataclasses.dataclass(frozen=True)
class MeterReading:
    """A meter reading. Contains a collection of IntervalBlocks."""

    id: str
    reading_type: ReadingType
    interval_blocks: Sequence[IntervalBlock]

    def __post_init__(self):
        """Post-process the data."""
        object.__setattr__(self, "interval_blocks", sorted(self.interval_blocks))

    def get_newest_interval_reading(self) -> IntervalReading | None:
        """Return the most recent IntervalBlock."""
        if not self.interval_blocks:
            return None
        newest_interval_block = self.interval_blocks[len(self.interval_blocks) - 1]
        if newest_interval_block is not None:
            return newest_interval_block.get_newest_interval_reading()
        return None


@final
@dataclasses.dataclass(frozen=True)
class UsagePoint:
    """A usage location. Contains multiple MeterReadings."""

    id: str
    sensor_device_class: sensor.SensorDeviceClass
    meter_readings: Collection[MeterReading]

    def get_meter_reading_by_id(self, id_str: str) -> MeterReading | None:
        """Get a meter reading by its ID."""
        for meter_reading in self.meter_readings:
            if meter_reading.id == id_str:
                return meter_reading
        return None
