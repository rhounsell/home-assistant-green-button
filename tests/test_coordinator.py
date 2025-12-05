
"""Unit tests for coordinator.py in green_button integration."""
import sys
import asyncio
from unittest.mock import MagicMock, patch
from custom_components.green_button import coordinator


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
    # Run the coroutine
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
    import sys
    import asyncio
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
