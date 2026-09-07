"""Validation regressions for incomplete and unsupported Green Button input."""

# ruff: noqa: SLF001, TID251

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree as ET

from custom_components.green_button import model, scaling, statistics
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator
from custom_components.green_button.parsers import espi
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
from tests.common import MockConfigEntry


def _interval_reading_element(
    *, cost: str | None, duration: str = "3600", value: str = "1"
) -> ET.Element:
    """Build a minimal ESPI interval reading element."""
    cost_element = "" if cost is None else f"<cost>{cost}</cost>"
    return ET.fromstring(
        "<IntervalReading xmlns='http://naesb.org/espi'>"
        f"{cost_element}"
        "<timePeriod><start>1788220800</start>"
        f"<duration>{duration}</duration></timePeriod><value>{value}</value>"
        "</IntervalReading>"
    )


def test_interval_parser_preserves_missing_cost_and_explicit_zero() -> None:
    """Missing cost remains unavailable while an explicit zero remains a zero."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    entry = espi.EspiEntry(None, ET.Element("entry"), "IntervalBlock")  # type: ignore[arg-type]
    parser = entry.create_interval_reading_parser(reading_type)

    assert parser(_interval_reading_element(cost=None)).cost is None
    assert parser(_interval_reading_element(cost="0")).cost == 0


@pytest.mark.parametrize(
    ("duration", "value", "match"),
    [
        pytest.param("0", "1", "duration must be positive", id="zero-duration"),
        pytest.param("-1", "1", "duration must be positive", id="negative-duration"),
        pytest.param("3600", "-1", "value must not be negative", id="negative-value"),
    ],
)
def test_interval_parser_rejects_invalid_measurements(
    duration: str, value: str, match: str
) -> None:
    """Invalid interval duration and consumption are rejected before import."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    entry = espi.EspiEntry(None, ET.Element("entry"), "IntervalBlock")  # type: ignore[arg-type]

    with pytest.raises(espi.EspiXmlParseError, match=match):
        entry.create_interval_reading_parser(reading_type)(
            _interval_reading_element(cost="1", duration=duration, value=value)
        )


def test_missing_cost_and_unknown_energy_unit_cannot_generate_statistics() -> None:
    """Incomplete monetary data and unsupported units do not become valid totals."""
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    reading = model.IntervalReading(
        reading_type, None, datetime(2026, 9, 1, tzinfo=UTC), timedelta(hours=1), 1
    )
    summary = model.UsageSummary(
        "summary", datetime(2026, 9, 1, tzinfo=UTC), timedelta(days=1), None, "CAD"
    )

    with pytest.raises(ValueError, match="cost is missing"):
        scaling.interval_cost(reading, 0)
    with pytest.raises(ValueError, match="cost is missing"):
        scaling.usage_summary_cost(summary, 0)
    with pytest.raises(ValueError, match="Unsupported energy unit"):
        statistics._convert_to_kwh(1, "unsupported")


async def test_empty_import_is_rejected_before_storage(
    hass: HomeAssistant,
) -> None:
    """An empty or wholly unsupported document cannot be archived as an import."""
    coordinator = GreenButtonCoordinator(
        hass, MockConfigEntry(domain=DOMAIN, entry_id="test-entry")
    )
    report = espi.EspiParseReport([], accepted_readings=0, skipped_readings=3)

    with (
        patch.object(espi, "parse_xml_with_report", return_value=report),
        patch.object(
            coordinator, "_trigger_statistics_update_for_all_readings", new_callable=AsyncMock
        ) as trigger_statistics,
        pytest.raises(UpdateFailed, match="no supported interval readings or summaries"),
    ):
        await coordinator.async_add_xml_data("<feed />")

    trigger_statistics.assert_not_awaited()
