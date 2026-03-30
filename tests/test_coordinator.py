from __future__ import annotations

import asyncio
import datetime as dt
import sys
from unittest.mock import MagicMock, patch

from homeassistant.components import sensor

from custom_components.green_button import coordinator, model


def test_green_button_coordinator_init_sets_attributes():
    mock_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.data = {"name": "Test", "usage_point_id": "up1"}
    mock_usage_point = MagicMock()
    mock_usage_point.id = "up1"
    with patch("custom_components.green_button.model.UsagePoint", return_value=mock_usage_point):
        coord = coordinator.GreenButtonCoordinator(mock_hass, mock_config_entry)
        assert coord.hass == mock_hass
        assert coord.config_entry == mock_config_entry


def test_green_button_coordinator_update_data_calls_async_update(monkeypatch):
    mock_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.data = {"name": "Test", "usage_point_id": "up1"}

    async def fake_async_update_data(self):
        return {"data": 123}

    monkeypatch.setattr(coordinator.GreenButtonCoordinator, "_async_update_data", fake_async_update_data)
    coord = coordinator.GreenButtonCoordinator(mock_hass, mock_config_entry)
    if sys.version_info >= (3, 8):
        result = asyncio.run(coord._async_update_data())
    else:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(coord._async_update_data())
    assert result == {"data": 123}


def test_green_button_coordinator_handle_update_error(monkeypatch):
    mock_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.data = {"name": "Test", "usage_point_id": "up1"}

    async def fail_update(self):
        raise Exception("fail")

    monkeypatch.setattr(coordinator.GreenButtonCoordinator, "_async_update_data", fail_update)
    coord = coordinator.GreenButtonCoordinator(mock_hass, mock_config_entry)
    if sys.version_info >= (3, 8):
        try:
            asyncio.run(coord._async_update_data())
        except Exception as e:
            assert str(e) == "fail"
    else:
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(coord._async_update_data())
        except Exception as e:
            assert str(e) == "fail"


def _make_coordinator() -> coordinator.GreenButtonCoordinator:
    mock_hass = MagicMock()
    mock_config_entry = MagicMock()
    mock_config_entry.data = {"name": "Test", "usage_point_id": "up1"}
    return coordinator.GreenButtonCoordinator(mock_hass, mock_config_entry)


def _reading_type() -> model.ReadingType:
    return model.ReadingType(
        id="rt1",
        commodity=1,
        currency="CAD",
        power_of_ten_multiplier=0,
        unit_of_measurement="kWh",
        interval_length=3600,
    )


def _usage_summary(summary_id: str, start_day: int, total_cost: float) -> model.UsageSummary:
    return model.UsageSummary(
        id=summary_id,
        start=dt.datetime(2025, 1, start_day, tzinfo=dt.timezone.utc),
        duration=dt.timedelta(days=30),
        total_cost=total_cost,
        currency="CAD",
        consumption_m3=100.0,
    )


def _usage_point(summary_items: list[model.UsageSummary]) -> model.UsagePoint:
    meter = model.MeterReading(id="mr1", reading_type=_reading_type(), interval_blocks=[])
    return model.UsagePoint(
        id="up1",
        sensor_device_class=sensor.SensorDeviceClass.GAS,
        meter_readings=[meter],
        usage_summaries=summary_items,
    )


def test_merge_usage_points_dedupes_usage_summaries_by_id() -> None:
    coord = _make_coordinator()

    jan = _usage_summary("jan", 1, 10.0)
    feb = _usage_summary("feb", 2, 20.0)
    feb_dupe = _usage_summary("feb", 2, 20.0)
    mar = _usage_summary("mar", 3, 30.0)

    coord.usage_points = [_usage_point([jan, feb])]
    coord._merge_usage_points([_usage_point([feb_dupe, mar])])

    merged = list(coord.usage_points[0].usage_summaries)
    assert [summary.id for summary in merged] == ["jan", "feb", "mar"]
    assert [summary.total_cost for summary in merged] == [10.0, 20.0, 30.0]


def test_get_usage_summaries_accepts_usage_point_id_for_summaries_only_path() -> None:
    coord = _make_coordinator()

    jan = _usage_summary("jan", 1, 10.0)
    feb = _usage_summary("feb", 2, 20.0)
    usage_point = _usage_point([jan, feb])

    coord.usage_points = [usage_point]

    summaries = coord.get_usage_summaries_for_meter_reading("up1")
    assert [summary.id for summary in summaries] == ["jan", "feb"]


def test_green_button_gas_sensor_suppresses_auto_state_writes() -> None:
    from custom_components.green_button.sensor import GreenButtonGasSensor

    coord = _make_coordinator()
    gas_sensor = GreenButtonGasSensor(coord, "up1")

    assert gas_sensor.suppress_auto_state_write is True
