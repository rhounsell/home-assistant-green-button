"""Verify services target external series rather than legacy sensor statistics."""

# ruff: noqa: TID251

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from custom_components.green_button import model, services, statistics
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.coordinator import GreenButtonCoordinator
from custom_components.green_button.parsers import espi
from custom_components.green_button.statistic_ids import statistic_id_from_unique_id
from custom_components.green_button.xml_storage import async_get_xml_storage
import pytest

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import entity_registry as er
from tests.common import MockConfigEntry, MockUser


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
            DOMAIN,
            "recalculate_cost_statistics",
            {"config_entry_id": entry.entry_id},
            blocking=True,
        )

    update.assert_awaited_once()
    entity = update.await_args.args[1]
    assert entity.long_term_statistics_id == statistic_id_from_unique_id(unique_id)
    assert entity.long_term_statistics_id != registered.entity_id
    assert statistics.create_metadata(entity)["source"] == DOMAIN


async def test_recalculation_writes_the_canonical_electricity_stream_once(
    hass: HomeAssistant,
) -> None:
    """The service receives one reconciled meter reading, not every XML stream."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    coordinator = GreenButtonCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"coordinator": coordinator}
    unique_id = f"{entry.entry_id}_meter_cost"
    registered = er.async_get(hass).async_get_or_create(
        "sensor", DOMAIN, unique_id, config_entry=entry
    )
    hass.states.async_set(registered.entity_id, "0")
    start = datetime(2026, 7, 1, tzinfo=UTC)
    reading_type = model.ReadingType("type", 1, "CAD", -3, "Wh", 3600)
    reading = model.IntervalReading(reading_type, 1000, start, timedelta(hours=1), 1000)
    meter_reading = model.MeterReading(
        "meter",
        reading_type,
        [
            model.IntervalBlock(
                "block", reading_type, start, timedelta(hours=1), [reading]
            )
        ],
    )
    usage_point = model.UsagePoint("point", SensorDeviceClass.ENERGY, [meter_reading])
    await services.async_setup_services(hass)

    with (
        patch.object(
            coordinator,
            "async_reconstruct_stored_usage_points",
            new=AsyncMock(return_value=[usage_point]),
        ) as reconstruct,
        patch.object(
            statistics, "update_cost_statistics", new_callable=AsyncMock
        ) as update,
    ):
        await hass.services.async_call(
            DOMAIN,
            "recalculate_cost_statistics",
            {"config_entry_id": entry.entry_id},
            blocking=True,
        )

    reconstruct.assert_awaited_once()
    update.assert_awaited_once()
    assert update.await_args.args[3] is meter_reading
    assert update.await_args.kwargs["merge_with_existing"] is False


async def test_delete_targets_external_statistics(hass: HomeAssistant) -> None:
    """The legacy entity-ID input resolves to the new series for deletion."""
    config_entry = MockConfigEntry(domain=DOMAIN)
    config_entry.add_to_hass(hass)
    entry = er.async_get(hass).async_get_or_create(
        "sensor", DOMAIN, "unique", config_entry=config_entry
    )
    await services.async_setup_services(hass)
    with patch.object(statistics, "clear_statistic") as clear:
        await hass.services.async_call(
            DOMAIN,
            "delete_statistics",
            {
                "config_entry_id": config_entry.entry_id,
                "statistic_id": entry.entity_id,
            },
            blocking=True,
        )
    clear.assert_awaited_once_with(hass, statistic_id_from_unique_id("unique"))


async def test_delete_rejects_another_integration(hass: HomeAssistant) -> None:
    """Other integrations cannot be mapped to Green Button series."""
    config_entry = MockConfigEntry(domain=DOMAIN)
    config_entry.add_to_hass(hass)
    entry = er.async_get(hass).async_get_or_create("sensor", "other", "unique")
    await services.async_setup_services(hass)
    with (
        patch.object(statistics, "clear_statistic") as clear,
        pytest.raises(HomeAssistantError, match="not a Green Button entity"),
    ):
        await hass.services.async_call(
            DOMAIN,
            "delete_statistics",
            {
                "config_entry_id": config_entry.entry_id,
                "statistic_id": entry.entity_id,
            },
            blocking=True,
        )
    clear.assert_not_awaited()


async def test_import_targets_only_the_selected_config_entry(
    hass: HomeAssistant,
) -> None:
    """Importing XML cannot copy one home's data into another entry."""
    first_entry = MockConfigEntry(domain=DOMAIN)
    second_entry = MockConfigEntry(domain=DOMAIN)
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    first_coordinator = GreenButtonCoordinator(hass, first_entry)
    second_coordinator = GreenButtonCoordinator(hass, second_entry)
    hass.data.setdefault(DOMAIN, {})[first_entry.entry_id] = {
        "coordinator": first_coordinator
    }
    hass.data[DOMAIN][second_entry.entry_id] = {"coordinator": second_coordinator}
    report = espi.EspiParseReport([], accepted_readings=0, skipped_readings=0)
    await services.async_setup_services(hass)

    with (
        patch.object(
            first_coordinator,
            "async_add_xml_data",
            new_callable=AsyncMock,
            return_value=report,
        ) as first_import,
        patch.object(
            second_coordinator, "async_add_xml_data", new_callable=AsyncMock
        ) as second_import,
    ):
        await hass.services.async_call(
            DOMAIN,
            "import_espi_xml",
            {"config_entry_id": first_entry.entry_id, "xml": "<feed />"},
            blocking=True,
        )

    first_import.assert_awaited_once_with("<feed />", store_in_config=True)
    second_import.assert_not_awaited()


async def test_clear_stored_xml_targets_only_the_selected_config_entry(
    hass: HomeAssistant,
) -> None:
    """Clearing one entry's archive preserves every other entry's archive."""
    first_entry = MockConfigEntry(domain=DOMAIN)
    second_entry = MockConfigEntry(domain=DOMAIN)
    first_entry.add_to_hass(hass)
    second_entry.add_to_hass(hass)
    first_storage = await async_get_xml_storage(hass, first_entry.entry_id)
    second_storage = await async_get_xml_storage(hass, second_entry.entry_id)
    await first_storage.async_add_xml("<first />", "electricity")
    await second_storage.async_add_xml("<second />", "electricity")
    await services.async_setup_services(hass)

    await hass.services.async_call(
        DOMAIN,
        "clear_stored_xml",
        {"config_entry_id": first_entry.entry_id, "commodity": "electricity"},
        blocking=True,
    )

    assert first_storage.get_stored_xmls() == []
    assert second_storage.get_stored_xmls() == [
        {"label": "electricity", "xmls": ["<second />"]}
    ]


async def test_clear_archive_resets_active_history_before_a_new_import(
    hass: HomeAssistant,
) -> None:
    """Cleared source data cannot be merged back by the next import."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    coordinator = GreenButtonCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"coordinator": coordinator}
    reading_type = model.ReadingType("type", 1, "CAD", 0, "Wh", 3600)
    start = datetime(2026, 7, 1, tzinfo=UTC)
    old_meter = model.MeterReading(
        "old-meter",
        reading_type,
        [
            model.IntervalBlock(
                "old",
                reading_type,
                start,
                timedelta(hours=1),
                [model.IntervalReading(reading_type, 0, start, timedelta(hours=1), 1)],
            )
        ],
    )
    coordinator.usage_points = [
        model.UsagePoint("usage-point", SensorDeviceClass.ENERGY, [old_meter])
    ]
    coordinator.async_set_updated_data({"usage_points": coordinator.usage_points})
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml("<old />", "electricity")
    await services.async_setup_services(hass)

    await hass.services.async_call(
        DOMAIN,
        "clear_stored_xml",
        {"config_entry_id": entry.entry_id, "commodity": "electricity"},
        blocking=True,
    )

    new_meter = model.MeterReading(
        "new-meter",
        reading_type,
        [
            model.IntervalBlock(
                "new",
                reading_type,
                start,
                timedelta(hours=1),
                [model.IntervalReading(reading_type, 0, start, timedelta(hours=1), 2)],
            )
        ],
    )
    report = espi.EspiParseReport(
        [model.UsagePoint("usage-point", SensorDeviceClass.ENERGY, [new_meter])],
        accepted_readings=1,
        skipped_readings=0,
    )
    with (
        patch.object(espi, "parse_xml_with_report", return_value=report),
        patch.object(
            coordinator,
            "_trigger_statistics_update_for_all_readings",
            new_callable=AsyncMock,
        ),
    ):
        await coordinator.async_add_xml_data("<new />")

    assert [meter.id for meter in coordinator.usage_points[0].meter_readings] == [
        "new-meter"
    ]


async def test_delete_rejects_a_green_button_entity_from_another_entry(
    hass: HomeAssistant,
) -> None:
    """A scoped delete cannot remove another Green Button entry's series."""
    selected_entry = MockConfigEntry(domain=DOMAIN)
    other_entry = MockConfigEntry(domain=DOMAIN)
    selected_entry.add_to_hass(hass)
    other_entry.add_to_hass(hass)
    entity = er.async_get(hass).async_get_or_create(
        "sensor", DOMAIN, "other_unique", config_entry=other_entry
    )
    await services.async_setup_services(hass)

    with (
        patch.object(statistics, "clear_statistic") as clear,
        pytest.raises(HomeAssistantError, match="does not belong"),
    ):
        await hass.services.async_call(
            DOMAIN,
            "delete_statistics",
            {
                "config_entry_id": selected_entry.entry_id,
                "statistic_id": entity.entity_id,
            },
            blocking=True,
        )

    clear.assert_not_awaited()


@pytest.mark.parametrize(
    ("service", "data"),
    [
        pytest.param("import_espi_xml", {"xml": "<feed />"}, id="import"),
        pytest.param("clear_stored_xml", {}, id="clear-archive"),
        pytest.param("recalculate_cost_statistics", {}, id="recalculate"),
        pytest.param(
            "delete_statistics",
            {"statistic_id": "sensor.green_button_display"},
            id="delete-statistics",
        ),
    ],
)
async def test_mutating_services_require_an_administrator(
    hass: HomeAssistant,
    hass_read_only_user: MockUser,
    service: str,
    data: dict[str, str],
) -> None:
    """Import, deletion, clearing, and recalculation require an administrator."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    await services.async_setup_services(hass)

    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN,
            service,
            {"config_entry_id": entry.entry_id, **data},
            context=Context(user_id=hass_read_only_user.id),
            blocking=True,
        )
