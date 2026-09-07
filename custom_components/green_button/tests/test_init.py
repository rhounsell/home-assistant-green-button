"""Verify Green Button lifecycle operations preserve imported XML.

Run from the core workspace with:
uv run --no-sync pytest -o pythonpath=config config/custom_components/green_button/tests
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock, patch

from custom_components import green_button
from custom_components.green_button.const import DOMAIN
from custom_components.green_button.xml_storage import (
    GreenButtonXmlStorage,
    async_get_xml_storage,
)
import pytest

from homeassistant.config_entries import SOURCE_USER, ConfigEntries, ConfigEntry
from homeassistant.core import HomeAssistant

XML = '<feed xmlns="http://www.w3.org/2005/Atom" />'


def _create_entry() -> ConfigEntry:
    return ConfigEntry(
        domain=DOMAIN,
        title="Green Button",
        version=1,
        minor_version=1,
        data={},
        options={},
        source=SOURCE_USER,
        unique_id=None,
        discovery_keys=MappingProxyType({}),
        subentries_data=(),
    )


@pytest.fixture
async def hass(tmp_path: Path) -> AsyncGenerator[HomeAssistant]:
    """Use real storage in a temporary config directory."""
    instance = HomeAssistant(str(tmp_path))
    instance.config_entries = ConfigEntries(instance, {})
    yield instance
    await instance.async_stop(force=True)


async def test_unload_preserves_xml(hass: HomeAssistant) -> None:
    """A fresh storage object can read the archive after successful unload."""
    entry = _create_entry()
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml(XML, "electricity")
    hass.data[DOMAIN][entry.entry_id] = {"coordinator": object()}

    with (
        patch.object(hass.config_entries, "async_unload_platforms", return_value=True),
        patch.object(green_button, "async_unload_services") as unload_services,
    ):
        assert await green_button.async_unload_entry(hass, entry)

    assert DOMAIN not in hass.data
    unload_services.assert_awaited_once_with(hass)
    reloaded = GreenButtonXmlStorage(hass, entry.entry_id)
    await reloaded.async_load()
    assert reloaded.get_stored_xmls() == [{"label": "electricity", "xmls": [XML]}]


async def test_failed_unload_keeps_runtime_data(hass: HomeAssistant) -> None:
    """An unsuccessful platform unload retains the entry and storage cache."""
    entry = _create_entry()
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml(XML, "electricity")
    coordinator = object()
    hass.data[DOMAIN][entry.entry_id] = {"coordinator": coordinator}

    with (
        patch.object(hass.config_entries, "async_unload_platforms", return_value=False),
        patch.object(green_button, "async_unload_services") as unload_services,
    ):
        assert not await green_button.async_unload_entry(hass, entry)

    assert hass.data[DOMAIN][entry.entry_id]["coordinator"] is coordinator
    assert await async_get_xml_storage(hass, entry.entry_id) is storage
    unload_services.assert_not_awaited()


async def test_unload_keeps_other_entry(hass: HomeAssistant) -> None:
    """Unloading one entry preserves another entry and shared services."""
    entry = _create_entry()
    other = _create_entry()
    await async_get_xml_storage(hass, entry.entry_id)
    other_storage = await async_get_xml_storage(hass, other.entry_id)
    await other_storage.async_add_xml(XML, "gas")
    hass.data[DOMAIN][entry.entry_id] = {"coordinator": object()}
    other_data = {"coordinator": object()}
    hass.data[DOMAIN][other.entry_id] = other_data

    with (
        patch.object(hass.config_entries, "async_unload_platforms", return_value=True),
        patch.object(green_button, "async_unload_services") as unload_services,
    ):
        assert await green_button.async_unload_entry(hass, entry)

    assert entry.entry_id not in hass.data[DOMAIN]
    assert hass.data[DOMAIN][other.entry_id] is other_data
    assert await async_get_xml_storage(hass, other.entry_id) is other_storage
    assert other_storage.get_stored_xmls() == [{"label": "gas", "xmls": [XML]}]
    unload_services.assert_not_awaited()


async def test_options_reload_preserves_xml(hass: HomeAssistant) -> None:
    """The options listener can unload and set up from the saved archive."""
    entry = _create_entry()
    storage = await async_get_xml_storage(hass, entry.entry_id)
    await storage.async_add_xml(XML, "electricity")

    async def reload_entry(entry_id: str) -> bool:
        assert entry_id == entry.entry_id
        assert await green_button.async_unload_entry(hass, entry)
        return await green_button.async_setup_entry(hass, entry)

    with (
        patch.object(hass.config_entries, "async_unload_platforms", return_value=True),
        patch.object(
            hass.config_entries, "async_forward_entry_setups", new=AsyncMock()
        ),
        patch.object(hass.config_entries, "async_reload", side_effect=reload_entry),
    ):
        assert await green_button.async_setup_entry(hass, entry)
        await entry.update_listeners[0](hass, entry)

    reloaded = await async_get_xml_storage(hass, entry.entry_id)
    assert reloaded is not storage
    assert reloaded.get_stored_xmls() == [{"label": "electricity", "xmls": [XML]}]
    assert hass.data[DOMAIN][entry.entry_id]["coordinator"].usage_points


async def test_concurrent_first_storage_imports_share_one_serialized_archive(
    hass: HomeAssistant,
) -> None:
    """Concurrent first access cannot create competing caches or duplicate XML."""
    entry = _create_entry()

    first_storage, second_storage = await asyncio.gather(
        async_get_xml_storage(hass, entry.entry_id),
        async_get_xml_storage(hass, entry.entry_id),
    )
    stored = await asyncio.gather(
        first_storage.async_add_xml(XML, "electricity"),
        second_storage.async_add_xml(XML, "electricity"),
    )

    assert first_storage is second_storage
    assert sorted(stored) == [False, True]
    assert first_storage.get_stored_xmls() == [{"label": "electricity", "xmls": [XML]}]
