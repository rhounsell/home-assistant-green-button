"""Module containing parsers for the Energy Services Provider Interface (ESPI) Atom feed defined by the North American Energy Standards Board."""

import datetime
import logging
from collections.abc import Callable
from typing import Final
from typing import TypeVar
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as defusedET
from homeassistant.components import sensor
from homeassistant.const import UnitOfEnergy

from .. import model

T = TypeVar("T")

_NAMESPACE_MAP: Final = {
    "atom": "http://www.w3.org/2005/Atom",
    "espi": "http://naesb.org/espi",
}


_UOM_MAP: Final = {
    "72": UnitOfEnergy.WATT_HOUR,
}


_CURRENCY_MAP: Final = {
    "124": "CAD",  # Canadian Dollar
    "840": "USD",  # US Dollar
}


_SERVICE_KIND: Final = {
    # 0 - Electricity.
    "0": sensor.SensorDeviceClass.ENERGY,
}


class EspiXmlParseError(ValueError):
    """Error when parsing ESPI XML."""


def _pretty_print(elem: ET.Element) -> str:
    return ET.tostring(elem, encoding="unicode")


def _parse_child_text(elem: ET.Element, xpath: str, parser: Callable[[str], T]) -> T:
    matches = elem.findall(xpath, _NAMESPACE_MAP)
    if len(matches) != 1:
        raise EspiXmlParseError(
            f"No path '{xpath}' found for entry:\n{_pretty_print(elem)}"
        )

    text = matches[0].text
    if text is None:
        raise EspiXmlParseError(
            f"Invalid value None at path {xpath!r} of entry:\n{_pretty_print(elem)}"
        )

    try:
        return parser(text)
    except ValueError as ex:
        raise EspiXmlParseError(
            f"Invalid value {text!r} at path '{xpath}' of entry:\n{_pretty_print(elem)}"
        ) from ex
    except KeyError as ex:  # For Mappings.
        raise EspiXmlParseError(
            f"Invalid value {text!r} at path '{xpath}' of entry:\n{_pretty_print(elem)}"
        ) from ex


def _parse_child_elems(
    elem: ET.Element, xpath: str, parser: Callable[[ET.Element], T]
) -> list[T]:
    out = []
    for match in elem.findall(xpath, _NAMESPACE_MAP):
        try:
            out.append(parser(match))
        except ValueError as ex:
            raise EspiXmlParseError(
                f"Invalid child at path '{xpath}' of entry:\n{_pretty_print(elem)}\nChild:\n{_pretty_print(match)}"
            ) from ex
    return out


def _to_utc_datetime(timestamp: str) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(int(timestamp), datetime.timezone.utc)


def _to_timedelta(duration: str) -> datetime.timedelta:
    return datetime.timedelta(seconds=int(duration))


class GreenButtonFeed:
    """A wrapper around a Green Button atom Feed XML element."""

    def __init__(self, xml: ET.Element) -> None:
        """Create a new instance."""
        self._xml = xml

    def find_entries(self, entry_type_tag: str) -> list["EspiEntry"]:
        """Find all atom entries whose root data tag has the specified name."""
        return [
            EspiEntry(self, elem, entry_type_tag)
            for elem in self._xml.findall(
                f"./atom:entry/atom:content/espi:{entry_type_tag}/../..",
                _NAMESPACE_MAP,
            )
        ]

    def to_usage_points(self) -> list[model.UsagePoint]:
        """Parse the feed into UsagePoints."""
        out = []
        for usage_point in self.find_entries("UsagePoint"):
            out.append(usage_point.to_usage_point())

        if not out:
            # No explicit UsagePoint entries - create default usage point
            # and associate only energy consumed meter readings (flowDirection=1)
            out = [self._create_default_usage_point_with_consumed_energy()]
        return out

    def _create_default_usage_point_with_consumed_energy(self) -> model.UsagePoint:
        """Create a default usage point with only energy consumed data."""
        logger = logging.getLogger(__name__)

        # Get all reading types and filter for energy consumed (flowDirection=1)
        reading_type_entries = self.find_entries("ReadingType")
        consumed_energy_reading_types: list[tuple[EspiEntry, model.ReadingType]] = []

        logger.info(
            "Found %d ReadingType entries to process", len(reading_type_entries)
        )

        for rt_entry in reading_type_entries:
            try:
                # Check if this is energy consumed data (flowDirection=1)
                rt_href = rt_entry.find_self_href()
                logger.debug("Processing ReadingType: %s", rt_href)
                flow_direction = rt_entry.parse_child_text("espi:flowDirection", int)
                interval_length = rt_entry.parse_child_text("espi:intervalLength", int)

                if flow_direction == 1:  # Energy consumed
                    # Skip daily summaries (intervalLength >= 86400 seconds = 24 hours)
                    if interval_length >= 86400:
                        logger.info(
                            "Skipping daily summary ReadingType: %s (intervalLength=%d seconds)",
                            rt_href,
                            interval_length,
                        )
                        continue

                    reading_type = rt_entry.to_reading_type()
                    consumed_energy_reading_types.append((rt_entry, reading_type))
                    logger.info(
                        "Found energy consumed ReadingType: %s (flowDirection=%d, intervalLength=%d)",
                        reading_type.id,
                        flow_direction,
                        interval_length,
                    )
                else:
                    logger.info(
                        "Skipping ReadingType with flowDirection=%d (not consumed energy)",
                        flow_direction,
                    )
            except (ValueError, EspiXmlParseError) as ex:
                logger.warning("Failed to parse ReadingType entry: %s", ex)
                continue

        # Find meter readings that match consumed energy reading types
        meter_reading_entries = self.find_entries("MeterReading")
        consumed_meter_readings = []

        for mr_entry in meter_reading_entries:
            try:
                # Find the related ReadingType for this MeterReading
                related_rt_hrefs = mr_entry.find_related_hrefs()

                # Check if any of the related reading types are for consumed energy
                for rt_entry, reading_type in consumed_energy_reading_types:
                    rt_href = rt_entry.find_self_href()
                    if rt_href in related_rt_hrefs:
                        # This meter reading is for consumed energy
                        meter_reading = self._create_meter_reading_with_reading_type(
                            mr_entry, reading_type
                        )
                        consumed_meter_readings.append(meter_reading)
                        logger.info(
                            "Associated MeterReading %s with consumed energy ReadingType %s",
                            meter_reading.id,
                            reading_type.id,
                        )
                        break
            except (ValueError, EspiXmlParseError) as ex:
                logger.warning("Failed to process MeterReading entry: %s", ex)
                continue

        logger.info(
            "Created default usage point with %d consumed energy meter readings",
            len(consumed_meter_readings),
        )

        return model.UsagePoint(
            id="default_usage_point",
            sensor_device_class=sensor.SensorDeviceClass.ENERGY,
            meter_readings=consumed_meter_readings,
        )

    def _create_meter_reading_with_reading_type(
        self, mr_entry: "EspiEntry", reading_type: model.ReadingType
    ) -> model.MeterReading:
        """Create a MeterReading with the given ReadingType and associated IntervalBlocks."""
        logger = logging.getLogger(__name__)

        mr_href = mr_entry.find_self_href()
        mr_related_hrefs = mr_entry.find_related_hrefs()

        logger.debug("MeterReading %s has related hrefs: %s", mr_href, mr_related_hrefs)

        # Find related interval blocks for this meter reading
        interval_blocks = mr_entry.find_related_entries(
            "IntervalBlock",
            mr_entry.create_interval_block_parser(reading_type),
        )

        logger.info(
            "MeterReading %s found %d related IntervalBlocks",
            mr_href,
            len(interval_blocks),
        )

        # If no interval blocks found via direct relations, try alternative matching
        if not interval_blocks:
            logger.warning(
                "No IntervalBlocks found via direct relations, trying alternative matching"
            )
            interval_blocks = self._find_interval_blocks_for_meter_reading(
                mr_entry, reading_type
            )
            logger.info(
                "Alternative matching found %d IntervalBlocks for MeterReading %s",
                len(interval_blocks),
                mr_href,
            )

        return model.MeterReading(
            id=mr_href,
            reading_type=reading_type,
            interval_blocks=interval_blocks,
        )

    def _find_interval_blocks_for_meter_reading(
        self, mr_entry: "EspiEntry", reading_type: model.ReadingType
    ) -> list[model.IntervalBlock]:
        """Find IntervalBlocks that relate back to the given MeterReading."""
        logger = logging.getLogger(__name__)

        mr_href = mr_entry.find_self_href()
        interval_block_entries = self.find_entries("IntervalBlock")
        matching_blocks = []

        logger.debug(
            "Checking %d IntervalBlock entries for MeterReading %s",
            len(interval_block_entries),
            mr_href,
        )

        for ib_entry in interval_block_entries:
            try:
                ib_related_hrefs = ib_entry.find_related_hrefs()
                logger.debug(
                    "IntervalBlock %s has related hrefs: %s",
                    ib_entry.find_self_href(),
                    ib_related_hrefs,
                )

                # Check if this interval block relates back to our meter reading
                if mr_href in ib_related_hrefs:
                    parser = ib_entry.create_interval_block_parser(reading_type)
                    interval_block = parser(ib_entry)
                    matching_blocks.append(interval_block)
                    logger.debug(
                        "Matched IntervalBlock %s to MeterReading %s",
                        ib_entry.find_self_href(),
                        mr_href,
                    )

            except (ValueError, EspiXmlParseError) as ex:
                logger.warning("Failed to process IntervalBlock entry: %s", ex)
                continue

        return matching_blocks


class EspiEntry:
    """A wrapper around an atom Entry XML element."""

    def __init__(self, root: GreenButtonFeed, elem: ET.Element, type_tag: str) -> None:
        """Create a new instance."""
        self._root = root
        self._elem = elem
        self._type_tag = type_tag

    def _pretty_print(self) -> str:
        return _pretty_print(self._elem)

    def _find_link_hrefs(self, rel: str) -> list[str]:
        hrefs = []
        for link in self._elem.findall(f"./atom:link[@rel='{rel}']", _NAMESPACE_MAP):
            href = link.get("href")
            if href is not None:
                hrefs.append(href)
        return hrefs

    def find_self_href(self) -> str:
        """Find the entry's self HREF."""
        hrefs = self._find_link_hrefs("self")
        if not hrefs:
            raise EspiXmlParseError(f"No self link for entry:\n{self._pretty_print()}")
        return hrefs[0]

    def find_related_hrefs(self) -> list[str]:
        """Find the entry's related HREFs."""
        return self._find_link_hrefs("related")

    def parse_child_text(self, path: str, parser: Callable[[str], T]) -> T:
        """Parse the text of the element at the specified path."""
        xpath = f"./atom:content/espi:{self._type_tag}/{path}"
        return _parse_child_text(self._elem, xpath, parser)

    def parse_child_elems(
        self, path: str, parser: Callable[[ET.Element], T]
    ) -> list[T]:
        """Parse the *element* at the specified path."""
        xpath = f"./atom:content/espi:{self._type_tag}/{path}"
        return _parse_child_elems(self._elem, xpath, parser)

    def find_related_entries(
        self, related_entry_type_tag: str, parser: Callable[["EspiEntry"], T]
    ) -> list[T]:
        """Find all related entries whose root data tag has the specified name."""
        related_hrefs = self.find_related_hrefs()
        matches = []
        for related_entry in self._root.find_entries(related_entry_type_tag):
            related_entry_href = related_entry.find_self_href()
            if related_entry_href in related_hrefs:
                matches.append(parser(related_entry))
        return matches

    def find_first_related_entries(
        self, related_entry_type_tag: str, parser: Callable[["EspiEntry"], T]
    ) -> T:
        """Find the first related entry whose root data tag has the specified name."""
        matches = self.find_related_entries(related_entry_type_tag, parser)
        if not matches:
            raise EspiXmlParseError(
                f"No related entry with tag '{related_entry_type_tag}' found for entry:\n{self._pretty_print()}"
            )
        return matches[0]

    def create_interval_reading_parser(
        self, reading_type: model.ReadingType
    ) -> Callable[[ET.Element], model.IntervalReading]:
        """Create an IntervalReading parser for the ReadingType."""

        def parser(elem: ET.Element) -> model.IntervalReading:
            return model.IntervalReading(
                reading_type=reading_type,
                cost=_parse_child_text(elem, "./espi:cost", int),
                start=_parse_child_text(
                    elem, "./espi:timePeriod/espi:start", _to_utc_datetime
                ),
                duration=_parse_child_text(
                    elem, "./espi:timePeriod/espi:duration", _to_timedelta
                ),
                value=_parse_child_text(elem, "./espi:value", int),
            )

        return parser

    def create_interval_block_parser(
        self, reading_type: model.ReadingType
    ) -> Callable[["EspiEntry"], model.IntervalBlock]:
        """Create an IntervalBlock parser for the ReadingType."""

        def parser(entry: EspiEntry) -> model.IntervalBlock:
            return model.IntervalBlock(
                id=entry.find_self_href(),
                reading_type=reading_type,
                start=entry.parse_child_text(
                    "espi:interval/espi:start", _to_utc_datetime
                ),
                duration=entry.parse_child_text(
                    "espi:interval/espi:duration", _to_timedelta
                ),
                interval_readings=entry.parse_child_elems(
                    "espi:IntervalReading",
                    entry.create_interval_reading_parser(reading_type),
                ),
            )

        return parser

    def to_reading_type(self) -> model.ReadingType:
        """Parse this entry as a ReadingType."""
        return model.ReadingType(
            id=self.find_self_href(),
            power_of_ten_multiplier=self.parse_child_text(
                "espi:powerOfTenMultiplier", int
            ),
            unit_of_measurement=self.parse_child_text("espi:uom", _UOM_MAP.__getitem__),
            currency=self.parse_child_text("espi:currency", _CURRENCY_MAP.__getitem__),
            interval_length=self.parse_child_text("espi:intervalLength", int),
        )

    def to_meter_reading(self) -> model.MeterReading:
        """Parse this entry as a MeterReading."""
        reading_type = self.find_first_related_entries(
            "ReadingType",
            EspiEntry.to_reading_type,
        )
        return model.MeterReading(
            id=self.find_self_href(),
            reading_type=reading_type,
            interval_blocks=self.find_related_entries(
                "IntervalBlock",
                self.create_interval_block_parser(reading_type),
            ),
        )

    def to_usage_point(self) -> model.UsagePoint:
        """Parse this entry as a UsagePoint."""
        self_href = self.find_self_href()
        sensor_device_class = self.parse_child_text(
            "espi:ServiceCategory/espi:kind",
            _SERVICE_KIND.__getitem__,
        )
        return model.UsagePoint(
            id=self_href,
            sensor_device_class=sensor_device_class,
            meter_readings=self.find_related_entries(
                "MeterReading",
                EspiEntry.to_meter_reading,
            ),
        )


def parse_xml(value: str) -> list[model.UsagePoint]:
    """Parse an ESPI atom feed XML string."""
    logger = logging.getLogger(__name__)

    try:
        root = defusedET.fromstring(value)
    except ET.ParseError as ex:
        raise EspiXmlParseError("Invalid XML.") from ex
    else:
        feed = GreenButtonFeed(root)

        # Debug: Log what entries we found
        logger.info("Found %d UsagePoint entries", len(feed.find_entries("UsagePoint")))
        logger.info(
            "Found %d MeterReading entries", len(feed.find_entries("MeterReading"))
        )
        logger.info(
            "Found %d IntervalBlock entries", len(feed.find_entries("IntervalBlock"))
        )
        logger.info(
            "Found %d ReadingType entries", len(feed.find_entries("ReadingType"))
        )

        usage_points = feed.to_usage_points()

        # Debug: Log the parsed structure
        for i, up in enumerate(usage_points):
            logger.info(
                "UsagePoint %d: id=%s, device_class=%s",
                i,
                up.id,
                up.sensor_device_class,
            )
            logger.info("UsagePoint %d: %d meter readings", i, len(up.meter_readings))
            for j, mr in enumerate(up.meter_readings):
                logger.info(
                    "  MeterReading %d: id=%s, %d interval blocks",
                    j,
                    mr.id,
                    len(mr.interval_blocks),
                )
                for k, ib in enumerate(mr.interval_blocks):
                    logger.info(
                        "    IntervalBlock %d: id=%s, %d interval readings",
                        k,
                        ib.id,
                        len(ib.interval_readings),
                    )
                    if ib.interval_readings:
                        first_reading = ib.interval_readings[0]
                        logger.info(
                            "      First reading: start=%s, value=%d",
                            first_reading.start,
                            first_reading.value,
                        )

        return usage_points
