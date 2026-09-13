"""Verify config-flow XML archive labeling."""

import pytest
from homeassistant.components.sensor import SensorDeviceClass

from custom_components.green_button import model
from custom_components.green_button.config_flow import _initial_xml_label


@pytest.mark.parametrize(
    ("device_class", "expected_label"),
    [
        pytest.param(SensorDeviceClass.ENERGY, "electricity", id="electricity"),
        pytest.param(SensorDeviceClass.GAS, "gas", id="gas"),
    ],
)
def test_initial_xml_label_uses_usage_point_device_class(
    device_class: SensorDeviceClass,
    expected_label: str,
) -> None:
    """Initial XML archives remain clearable by their detected commodity."""
    usage_point = model.UsagePoint("point", device_class, [])

    assert _initial_xml_label([usage_point]) == expected_label
