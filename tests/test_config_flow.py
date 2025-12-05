"""Test the Green Button config flow."""
from unittest.mock import mock_open, patch
import pytest

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.green_button.const import DOMAIN


# Sample valid Green Button XML for testing
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" 
      xmlns:espi="http://naesb.org/espi">
    <id>urn:uuid:test-feed</id>
    <title>Green Button Data</title>
    <updated>2024-01-01T00:00:00Z</updated>
    <entry>
        <id>urn:uuid:test-usage-point</id>
        <link rel="self" href="/UsagePoint/1"/>
        <link rel="up" href="/UsagePoint"/>
        <title>Usage Point</title>
        <content>
            <UsagePoint xmlns="http://naesb.org/espi">
                <ServiceCategory><kind>0</kind></ServiceCategory>
            </UsagePoint>
        </content>
    </entry>
</feed>"""


@pytest.mark.asyncio
async def test_form_xml_inline(hass: HomeAssistant) -> None:
    """Test we get the form and can submit inline XML."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    # Note: errors might be an empty dict instead of None
    assert result["errors"] == {} or result["errors"] is None
    assert result["step_id"] == "user"

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test Green Button",
            "input_type": "xml",
            "xml": SAMPLE_XML,
            "gas_cost_allocation": "pro_rate_daily",
            "gas_usage_allocation": "daily_readings",
        },
    )
    await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Test Green Button"
    # Check actual data structure as returned by the config flow
    assert result2["data"]["name"] == "Test Green Button"
    assert result2["data"]["usage_point_id"] == "/UsagePoint/1"
    assert result2["data"]["xml"] == SAMPLE_XML
    assert result2["data"]["gas_cost_allocation"] == "pro_rate_daily"
    assert result2["data"]["gas_usage_allocation"] == "daily_readings"
@pytest.mark.asyncio
async def test_form_xml_file(hass: HomeAssistant) -> None:
    """Test we can submit XML via file path."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] is None

    with patch(
        "custom_components.green_button.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry, patch(
        "builtins.open", mock_open(read_data=SAMPLE_XML)
    ), patch(
        "os.path.exists", return_value=True
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Test Green Button File",
                "input_type": "file",
                "xml_file_path": "/config/greenbutton.xml",
                "gas_cost_allocation": "monthly_increment",
                "gas_usage_allocation": "monthly_increment",
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Test Green Button File"
    assert result2["data"]["name"] == "Test Green Button File"
    assert result2["data"]["input_type"] == "file"
    assert result2["data"]["xml_file_path"] == "/config/greenbutton.xml"
    assert len(mock_setup_entry.mock_calls) == 1


@pytest.mark.asyncio
async def test_form_invalid_xml(hass: HomeAssistant) -> None:
    """Test we handle invalid XML."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test Invalid XML",
            "input_type": "xml",
            "xml": "Not valid XML",
            "gas_cost_allocation": "pro_rate_daily",
            "gas_usage_allocation": "daily_readings",
        },
    )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {"xml": "invalid_xml"}


@pytest.mark.asyncio
async def test_form_file_not_found(hass: HomeAssistant) -> None:
    """Test we handle file not found error."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch("os.path.exists", return_value=False):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Test Missing File",
                "input_type": "file",
                "xml_file_path": "/config/missing.xml",
                "gas_cost_allocation": "pro_rate_daily",
                "gas_usage_allocation": "daily_readings",
            },
        )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {"xml_file_path": "file_not_found"}


@pytest.mark.asyncio
async def test_form_missing_required_field(hass: HomeAssistant) -> None:
    """Test we handle missing required fields."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    # Missing 'name' field
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "input_type": "xml",
            "xml": SAMPLE_XML,
            "gas_cost_allocation": "pro_rate_daily",
            "gas_usage_allocation": "daily_readings",
        },
    )

    assert result2["type"] == FlowResultType.FORM
    assert "base" in result2["errors"] or "name" in result2["errors"]


@pytest.mark.asyncio
async def test_form_empty_xml_inline(hass: HomeAssistant) -> None:
    """Test we handle empty inline XML when xml input type is selected."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test Empty XML",
            "input_type": "xml",
            "xml": "",
            "gas_cost_allocation": "pro_rate_daily",
            "gas_usage_allocation": "daily_readings",
        },
    )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {"xml": "missing_xml"}


@pytest.mark.asyncio
async def test_form_empty_file_path(hass: HomeAssistant) -> None:
    """Test we handle empty file path when file input type is selected."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test Empty Path",
            "input_type": "file",
            "xml_file_path": "",
            "gas_cost_allocation": "pro_rate_daily",
            "gas_usage_allocation": "daily_readings",
        },
    )

    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {"xml_file_path": "missing_file_path"}
