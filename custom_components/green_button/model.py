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
from typing import final

from homeassistant.components import sensor


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
    powerOfTenMultiplier: int

    def __lt__(self, other: "IntervalBlock") -> bool:
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

    def __lt__(self, other: "IntervalBlock") -> bool:
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
    commodity: int | None
    currency: str
    power_of_ten_multiplier: int
    unit_of_measurement: str
    interval_length: int


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
        return newest_interval_block.get_newest_interval_reading()


@final
@dataclasses.dataclass(frozen=True)
class UsageSummary:
    """A usage/billing summary (typically monthly for gas)."""

    id: str
    start: datetime.datetime
    duration: datetime.timedelta
    total_cost: float
    currency: str
    # Optional total consumption for the billing period in m³ (if provided)
    consumption_m3: float | None = None


@final
@dataclasses.dataclass(frozen=True)
class UsagePoint:
    """A usage location. Contains multiple MeterReadings."""

    id: str
    sensor_device_class: sensor.SensorDeviceClass
    meter_readings: Collection[MeterReading]
    usage_summaries: Collection[UsageSummary] = dataclasses.field(default_factory=list)

    def get_meter_reading_by_id(self, id_str: str) -> MeterReading | None:
        """Get a meter reading by its ID."""
        for meter_reading in self.meter_readings:
            if meter_reading.id == id_str:
                return meter_reading
        return None

    @classmethod
    def default_usage_point(cls) -> UsagePoint:
        """Create a default usage point for cases where no usage points are found.

        This is used when utilities like Hydro Ottawa provide only links to usage
        data without actual usage point definitions in the XML.
        """
        # Create a default reading type for energy consumption
        default_reading_type = ReadingType(
            id="default_energy_reading",
            commodity=1,
            currency="CAD",
            power_of_ten_multiplier=0,
            unit_of_measurement="kWh",
            interval_length=3600,
        )

        # Create a default meter reading with no interval blocks
        default_meter_reading = MeterReading(
            id="default_energy_meter",
            reading_type=default_reading_type,
            interval_blocks=[],
        )

        return cls(
            id="default_usage_point",
            sensor_device_class=sensor.SensorDeviceClass.ENERGY,
            meter_readings=[default_meter_reading],
            usage_summaries=[],
        )
