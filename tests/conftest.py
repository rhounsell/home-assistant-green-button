"""Global fixtures for green_button integration."""
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir."""
    yield


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock dependencies that are not needed for config flow tests."""
    with patch("homeassistant.setup._async_process_dependencies", return_value=None):
        yield