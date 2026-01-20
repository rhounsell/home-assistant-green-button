"""Module containing parsers for the Energy Services Provider Interface (ESPI) Atom feed defined by the North American Energy Standards Board."""

import datetime
import logging
from collections.abc import Callable
from typing import Final
from typing import TypeVar
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as defusedET
from homeassistant.components import sensor

from .. import model

T = TypeVar("T")

_NAMESPACE_MAP: Final = {
    "atom": "http://www.w3.org/2005/Atom",
    "espi": "http://naesb.org/espi",
}


_UOM_MAP: Final = {
    # Use strings to keep model generic across energy and gas
    "72": "Wh",   # Watt-hours (electricity)
    "42": "m³",   # Cubic meters (gas)
    "80": "currency",
}


_CURRENCY_MAP: Final = {
    "124": "CAD",  # Canadian Dollar
    "840": "USD",  # US Dollar
}


_SERVICE_KIND: Final = {
    # 0 - Electricity.
    "0": sensor.SensorDeviceClass.ENERGY,
    # 1 - Gas.
    "1": sensor.SensorDeviceClass.GAS,
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


def _parse_optional_child_text(
    elem: ET.Element, xpath: str, parser: Callable[[str], T], default: T
) -> T:
    """Parse optional child text at xpath; return default if missing.

    Raises if more than one match is found or if the value cannot be parsed.
    """
    matches = elem.findall(xpath, _NAMESPACE_MAP)
    if len(matches) == 0:
        return default
    if len(matches) > 1:
        raise EspiXmlParseError(
            f"Multiple values at path '{xpath}' of entry:\n{_pretty_print(elem)}"
        )
    text = matches[0].text
    if text is None:
        return default
    try:
        return parser(text)
    except ValueError as ex:
        raise EspiXmlParseError(
            f"Invalid value {text!r} at path '{xpath}' of entry:\n{_pretty_print(elem)}"
        ) from ex
    except KeyError as ex:
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

        logger.debug(
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
                        logger.debug(
                            "Skipping daily summary ReadingType: %s (intervalLength=%d seconds)",
                            rt_href,
                            interval_length,
                        )
                        continue

                    reading_type = rt_entry.to_reading_type()
                    consumed_energy_reading_types.append((rt_entry, reading_type))
                    logger.debug(
                        "Found energy consumed ReadingType: %s (flowDirection=%d, intervalLength=%d)",
                        reading_type.id,
                        flow_direction,
                        interval_length,
                    )
                else:
                    logger.debug(
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
                        logger.debug(
                            "Associated MeterReading %s with consumed energy ReadingType %s",
                            meter_reading.id,
                            reading_type.id,
                        )
                        break
            except (ValueError, EspiXmlParseError) as ex:
                logger.warning("Failed to process MeterReading entry: %s", ex)
                continue

        logger.debug(
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

        logger.debug(
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
            logger.debug(
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

    @property
    def elem(self) -> ET.Element:
        """Public accessor for the XML element."""
        return self._elem

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
            # Check for exact match or prefix match (for feed vs entry hrefs)
            for related_href in related_hrefs:
                if related_entry_href == related_href or related_entry_href.startswith(related_href + "/"):
                    matches.append(parser(related_entry))
                    break
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
                # 'cost' is optional in some feeds; default to 0 if missing
                cost=_parse_optional_child_text(elem, "./espi:cost", int, 0),
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
            commodity=_parse_optional_child_text(
                self._elem, "./atom:content/espi:ReadingType/espi:commodity", int, None
            ),
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
        logger = logging.getLogger(__name__)
        self_href = self.find_self_href()
        sensor_device_class = self.parse_child_text(
            "espi:ServiceCategory/espi:kind",
            _SERVICE_KIND.__getitem__,
        )

        # Find meter readings - handle both direct links and feed links
        related_hrefs = self.find_related_hrefs()
        logger.debug("UsagePoint %s has related hrefs: %s", self_href, related_hrefs)

        meter_readings = []
        usage_summaries: list[model.UsageSummary] = []
        for mr_entry in self._root.find_entries("MeterReading"):
            mr_href = mr_entry.find_self_href()
            # Check if meter reading matches any related href (exact or prefix match for feeds)
            for related_href in related_hrefs:
                if mr_href == related_href or mr_href.startswith(related_href + "/"):
                    logger.debug("Matched MeterReading %s to UsagePoint %s", mr_href, self_href)

                    # Get the related ReadingType to check flowDirection
                    try:
                        reading_type_entries = mr_entry.find_related_entries(
                            "ReadingType", EspiEntry.to_reading_type
                        )
                        if reading_type_entries:
                            reading_type = reading_type_entries[0]
                            # Get flowDirection from the ReadingType entry
                            rt_entry = None
                            for rt in self._root.find_entries("ReadingType"):
                                if rt.find_self_href() == reading_type.id:
                                    rt_entry = rt
                                    break

                            if rt_entry:
                                flow_direction = rt_entry.parse_child_text("espi:flowDirection", int)
                                interval_length = rt_entry.parse_child_text("espi:intervalLength", int)

                                # For electricity: include sub-daily consumption (< 86400)
                                # For gas: include daily consumption (== 86400)
                                if (
                                    sensor_device_class == sensor.SensorDeviceClass.ENERGY
                                    and flow_direction == 1
                                    and interval_length < 86400
                                ) or (
                                    sensor_device_class == sensor.SensorDeviceClass.GAS
                                    and flow_direction == 1
                                    and interval_length == 86400
                                ):
                                    logger.debug(
                                        "Including MeterReading %s (flowDirection=%d, intervalLength=%d)",
                                        mr_href, flow_direction, interval_length
                                    )
                                    meter_readings.append(mr_entry.to_meter_reading())
                                else:
                                    logger.debug(
                                        "Skipping MeterReading %s (flowDirection=%d, intervalLength=%d)",
                                        mr_href, flow_direction, interval_length
                                    )
                            else:
                                # If we can't determine flow direction, include it
                                logger.warning("Could not determine flowDirection for %s, including by default", mr_href)
                                meter_readings.append(mr_entry.to_meter_reading())
                        else:
                            # If no reading type found, include it
                            logger.warning("No ReadingType found for %s, including by default", mr_href)
                            meter_readings.append(mr_entry.to_meter_reading())
                    except (ValueError, EspiXmlParseError) as ex:
                        logger.warning("Failed to check flowDirection for %s: %s, including by default", mr_href, ex)
                        meter_readings.append(mr_entry.to_meter_reading())

                    break

        # Attach UsageSummary entries related to this UsagePoint
        for us_entry in self._root.find_entries("UsageSummary"):
            # Match by related links
            us_related = us_entry.find_related_hrefs()
            if any(self_href == href or self_href.startswith(href + "/") for href in us_related):
                try:
                    # billingPeriod
                    start = us_entry.parse_child_text("espi:billingPeriod/espi:start", _to_utc_datetime)
                    duration = us_entry.parse_child_text("espi:billingPeriod/espi:duration", _to_timedelta)
                    currency = _parse_optional_child_text(
                        us_entry.elem,
                        "./atom:content/espi:UsageSummary/espi:currency",
                        _CURRENCY_MAP.__getitem__,
                        "CAD",
                    )
                    # Prefer an explicit Amount Due in costAdditionalDetailLastPeriod
                    def _find_amount_due(e: ET.Element) -> float | None:
                        for cad in e.findall(
                            "./atom:content/espi:UsageSummary/espi:costAdditionalDetailLastPeriod",
                            _NAMESPACE_MAP,
                        ):
                            note = cad.find("espi:note", _NAMESPACE_MAP)
                            if note is not None and (note.text or "").strip().lower() in {
                                "amount due",
                                "total gas charges ($)",
                            }:
                                amt = cad.find("espi:amount", _NAMESPACE_MAP)
                                meas = cad.find("espi:measurement", _NAMESPACE_MAP)
                                if amt is not None and meas is not None:
                                    p10 = meas.find("espi:powerOfTenMultiplier", _NAMESPACE_MAP)
                                    try:
                                        power = int(p10.text) if p10 is not None and p10.text else -3
                                    except ValueError:
                                        power = -3
                                    val = float(amt.text or 0)
                                    return val * (10 ** power)
                        return None
                    total_cost = _find_amount_due(us_entry.elem)
                    # Extract currentBillingPeriodOverAllConsumption (m³) if available
                    consumption_m3: float | None = None
                    try:
                        meas = us_entry.elem.find(
                            "./atom:content/espi:UsageSummary/espi:currentBillingPeriodOverAllConsumption",
                            _NAMESPACE_MAP,
                        )
                        if meas is not None:
                            p10 = meas.find("espi:powerOfTenMultiplier", _NAMESPACE_MAP)
                            uom = meas.find("espi:uom", _NAMESPACE_MAP)
                            val_el = meas.find("espi:value", _NAMESPACE_MAP)
                            if val_el is not None and val_el.text is not None:
                                try:
                                    power = int(p10.text) if p10 is not None and p10.text else -3
                                except ValueError:
                                    power = -3
                                # Only accept m³ (uom 42)
                                is_m3 = (uom is not None and (uom.text or "").strip() == "42")
                                raw_val = float(val_el.text)
                                val = raw_val * (10 ** power)
                                consumption_m3 = float(val) if is_m3 else None
                    except Exception:
                        consumption_m3 = None
                    if total_cost is None:
                        # Fallback to billLastPeriod with implicit -3 scaling
                        try:
                            raw = us_entry.parse_child_text("espi:billLastPeriod", float)
                            total_cost = raw * (10 ** -3)
                        except Exception:
                            total_cost = 0.0
                    usage_summaries.append(
                        model.UsageSummary(
                            id=us_entry.find_self_href(),
                            start=start,
                            duration=duration,
                            total_cost=float(total_cost or 0.0),
                            currency=currency,
                            consumption_m3=consumption_m3,
                        )
                    )
                except Exception as ex:
                    logger.warning("Failed to parse UsageSummary for %s: %s", self_href, ex)

        logger.debug(
            "UsagePoint %s found %d meter readings and %d usage summaries (after filtering)",
            self_href, len(meter_readings), len(usage_summaries)
        )

        return model.UsagePoint(
            id=self_href,
            sensor_device_class=sensor_device_class,
            meter_readings=meter_readings,
            usage_summaries=usage_summaries,
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
        logger.debug("Found %d UsagePoint entries", len(feed.find_entries("UsagePoint")))
        logger.debug(
            "Found %d MeterReading entries", len(feed.find_entries("MeterReading"))
        )
        logger.debug(
            "Found %d IntervalBlock entries", len(feed.find_entries("IntervalBlock"))
        )
        logger.debug(
            "Found %d ReadingType entries", len(feed.find_entries("ReadingType"))
        )

        usage_points = feed.to_usage_points()

        # Debug: Log the parsed structure
        for i, up in enumerate(usage_points):
            logger.debug(
                "UsagePoint %d: id=%s, device_class=%s",
                i,
                up.id,
                up.sensor_device_class,
            )
            logger.debug("UsagePoint %d: %d meter readings", i, len(up.meter_readings))
            for j, mr in enumerate(up.meter_readings):
                logger.debug(
                    "  MeterReading %d: id=%s, %d interval blocks",
                    j,
                    mr.id,
                    len(mr.interval_blocks),
                )
                for k, ib in enumerate(mr.interval_blocks):
                    logger.debug(
                        "    IntervalBlock %d: id=%s, %d interval readings",
                        k,
                        ib.id,
                        len(ib.interval_readings),
                    )
                    if ib.interval_readings:
                        first_reading = ib.interval_readings[0]
                        logger.debug(
                            "      First reading: start=%s, value=%d",
                            first_reading.start,
                            first_reading.value,
                        )

        return usage_points
