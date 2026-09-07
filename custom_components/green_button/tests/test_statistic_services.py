"""Verify services target external series rather than legacy sensor statistics."""

# ruff: noqa: TID251

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from custom_components.green_button import model, services, statistics
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator
from custom_components.green_button.statistic_ids import statistic_id_from_unique_id
from custom_components.green_button.xml_storage import async_get_xml_storage
import pytest

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from tests.common import MockConfigEntry


@pytest.mark.parametrize(
    ("device_class", "identity", "updater_name"),
    [
        pytest.param(
            SensorDeviceClass.ENERGY,
            "meter_cost",
            "update_cost_statistics",
            id="electricity",
        ),
        pytest.param(
            SensorDeviceClass.GAS,
            "point_gas_cost",
            "update_gas_cost_statistics",
            id="gas",
        ),
    ],
)
async def test_recalculation_targets_external_statistics(
    hass: HomeAssistant,
    device_class: SensorDeviceClass,
    identity: str,
    updater_name: str,
) -> None:
    """Service-created entities use the same ID mapping as display sensors."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={
            "gas_usage_allocation": "monthly_increment",
            "gas_cost_allocation": "monthly_increment",
            "gas_cost_power_of_ten_multiplier": -3,
            "electricity_cost_power_of_ten_multiplier": -3,
        },
    )
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "coordinator": GreenButtonCoordinator(hass, entry)
    }
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml("<feed />", "fixture")
    unique_id = f"{entry.entry_id}_{identity}"
    registered = er.async_get(hass).async_get_or_create(
        "sensor", DOMAIN, unique_id, config_entry=entry
    )
    hass.states.async_set(registered.entity_id, "0")
    start = datetime(2026, 7, 1, tzinfo=UTC)
    rt = model.ReadingType("type", 1, "CAD", -3, "Wh", 3600)
    reading = model.IntervalReading(rt, 1000, start, timedelta(hours=1), 1000)
    mr = model.MeterReading(
        "meter",
        rt,
        [model.IntervalBlock("block", rt, start, timedelta(hours=1), [reading])],
    )
    up = model.UsagePoint(
        "point",
        device_class,
        [mr],
        [model.UsageSummary("summary", start, timedelta(days=30), 1.0, "CAD", 1)],
    )
    await services.async_setup_services(hass)

    with (
        patch.object(services.espi, "parse_xml", return_value=[up]),
        patch.object(statistics, updater_name, new_callable=AsyncMock) as update,
    ):
        await hass.services.async_call(
            DOMAIN, "recalculate_cost_statistics", {}, blocking=True
        )

    update.assert_awaited_once()
    entity = update.await_args.args[1]
    assert entity.long_term_statistics_id == statistic_id_from_unique_id(unique_id)
    assert entity.long_term_statistics_id != registered.entity_id
    assert statistics.create_metadata(entity)["source"] == DOMAIN


async def test_delete_targets_external_statistics(hass: HomeAssistant) -> None:
    """The legacy entity-ID input resolves to the new series for deletion."""
    entry = er.async_get(hass).async_get_or_create("sensor", DOMAIN, "unique")
    await services.async_setup_services(hass)
    with patch.object(statistics, "clear_statistic") as clear:
        await hass.services.async_call(
            DOMAIN,
            "delete_statistics",
            {"statistic_id": entry.entity_id},
            blocking=True,
        )
    clear.assert_awaited_once_with(hass, statistic_id_from_unique_id("unique"))


async def test_delete_rejects_another_integration(hass: HomeAssistant) -> None:
    """Other integrations cannot be mapped to Green Button series."""
    entry = er.async_get(hass).async_get_or_create("sensor", "other", "unique")
    await services.async_setup_services(hass)
    with (
        patch.object(statistics, "clear_statistic") as clear,
        pytest.raises(HomeAssistantError, match="not a Green Button entity"),
    ):
        await hass.services.async_call(
            DOMAIN,
            "delete_statistics",
            {"statistic_id": entry.entity_id},
            blocking=True,
        )
    clear.assert_not_awaited()
