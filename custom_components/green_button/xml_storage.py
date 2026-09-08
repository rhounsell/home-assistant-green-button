"""Separate storage for Green Button XML data.

Uses a dedicated Store instance instead of config entry data to handle
large XML files properly. Config entries use delayed writes and are not
designed for multi-MB data storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}_xml"
TEMP_STORAGE_PREFIX = f"{DOMAIN}_xml_temp"
_STORAGE_LOCKS = f"{DOMAIN}_xml_storage_locks"


def _get_storage_key(entry_id: str) -> str:
    """Get the storage key for a config entry's XML data."""
    return f"{STORAGE_KEY_PREFIX}_{entry_id}"


def _get_temp_storage_key(unique_id: str) -> str:
    """Get the temporary storage key based on unique_id (for config flow)."""
    return f"{TEMP_STORAGE_PREFIX}_{unique_id}"


class GreenButtonXmlStorage:
    """Storage handler for Green Button XML data."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the XML storage."""
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store[dict[str, Any]](
            hass,
            STORAGE_VERSION,
            _get_storage_key(entry_id),
            private=True,
            # Use thread-safe serialization for large data
            serialize_in_event_loop=False,
        )
        self._data: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def async_load(self) -> dict[str, Any]:
        """Load stored XML data from disk."""
        async with self._lock:
            return await self._async_load()

    async def _async_load(self) -> dict[str, Any]:
        """Load stored XML data while holding the storage lock."""
        if self._data is None:
            self._data = await self._store.async_load() or {"stored_xmls": []}
        return self._data

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save XML data to disk immediately."""
        async with self._lock:
            await self._async_save(data)

    async def _async_save(self, data: dict[str, Any]) -> None:
        """Save XML data while holding the storage lock."""
        self._data = data
        document_count = sum(
            len(entry.get("xmls", [])) for entry in data.get("stored_xmls", [])
        )
        _LOGGER.info(
            "Saving %d Green Button XML document(s) for entry %s",
            document_count,
            self.entry_id,
        )
        await self._store.async_save(data)

    def async_delay_save(self, data: dict[str, Any], delay: float = 1.0) -> None:
        """Schedule a delayed save of XML data."""
        self._data = data
        self._store.async_delay_save(lambda: data, delay)
        _LOGGER.debug("Scheduled delayed save of XML data for entry %s", self.entry_id)

    async def async_remove(self) -> None:
        """Remove the storage file."""
        async with self._lock:
            await self._store.async_remove()
            self._data = None
        _LOGGER.info("Removed XML storage file for entry %s", self.entry_id)

    def get_stored_xmls(self) -> list[dict[str, Any]]:
        """Get the stored XMLs list from cached data."""
        if self._data is None:
            return []
        return self._data.get("stored_xmls", [])

    async def async_add_xml(self, xml_data: str, label: str) -> bool:
        """Add or merge XML data with a label.

        Return whether new source content was stored. Identical content is
        deduplicated by its SHA-256 hash across every label.
        """
        content_hash = hashlib.sha256(xml_data.encode()).hexdigest()
        async with self._lock:
            data = await self._async_load()
            stored_xmls = data.get("stored_xmls", [])

            # Migrate old format entries (single "xml" key) to new format ("xmls" list)
            for entry in stored_xmls:
                if "xml" in entry and "xmls" not in entry:
                    entry["xmls"] = [entry.pop("xml")]

            for entry in stored_xmls:
                if content_hash in {
                    hashlib.sha256(existing_xml.encode()).hexdigest()
                    for existing_xml in entry.get("xmls", [])
                }:
                    _LOGGER.info(
                        "Ignoring duplicate XML content already stored under label '%s'",
                        entry.get("label"),
                    )
                    return False

            # Check if an entry with this label already exists
            existing_index = next(
                (
                    i
                    for i, entry in enumerate(stored_xmls)
                    if entry.get("label") == label
                ),
                None,
            )

            if existing_index is not None:
                existing_entry = stored_xmls[existing_index]
                existing_xmls = existing_entry.get("xmls", [])
                existing_xmls.append(xml_data)
                stored_xmls[existing_index] = {"label": label, "xmls": existing_xmls}
                _LOGGER.info(
                    "Merged new XML into existing label '%s' (now %d XMLs stored for this label)",
                    label,
                    len(existing_xmls),
                )
            else:
                _LOGGER.info("Adding new stored XML with label '%s'", label)
                stored_xmls.append({"label": label, "xmls": [xml_data]})

            data["stored_xmls"] = stored_xmls

            # Use immediate save for reliability
            await self._async_save(data)
            _LOGGER.info("Stored %d label(s) in XML storage", len(stored_xmls))
            return True

    async def async_clear_label(self, label: str | None = None) -> tuple[int, int]:
        """Clear stored XMLs for a specific label or all labels.

        Returns tuple of (removed_count, remaining_count).
        """
        async with self._lock:
            data = await self._async_load()
            stored_xmls = data.get("stored_xmls", [])

            if not stored_xmls:
                _LOGGER.info("No stored XMLs to clear")
                return (0, 0)

            if label is None:
                removed_count = len(stored_xmls)
                _LOGGER.info("Clearing all %d stored XML label(s)", removed_count)
                data["stored_xmls"] = []
                await self._async_save(data)
                _LOGGER.info(
                    "✅ Cleared all stored XMLs, storage file now contains empty list"
                )
                return (removed_count, 0)

            original_count = len(stored_xmls)
            stored_xmls = [x for x in stored_xmls if x.get("label") != label]
            removed_count = original_count - len(stored_xmls)

            if removed_count > 0:
                _LOGGER.info(
                    "Clearing label '%s': removing %d of %d total label(s)",
                    label,
                    removed_count,
                    original_count,
                )
                data["stored_xmls"] = stored_xmls
                await self._async_save(data)
                _LOGGER.info(
                    "✅ Cleared label '%s', %d label(s) remaining in storage",
                    label,
                    len(stored_xmls),
                )
            else:
                _LOGGER.warning("Label '%s' not found in stored XMLs", label)

            return (removed_count, len(stored_xmls))


async def async_get_xml_storage(
    hass: HomeAssistant, entry_id: str
) -> GreenButtonXmlStorage:
    """Get or create an XML storage instance for a config entry."""
    storage_key = f"{DOMAIN}_xml_storage_{entry_id}"
    domain_data = hass.data.setdefault(DOMAIN, {})
    storage_locks = domain_data.setdefault(_STORAGE_LOCKS, {})
    storage_lock = storage_locks.setdefault(entry_id, asyncio.Lock())

    async with storage_lock:
        if storage_key not in domain_data:
            storage = GreenButtonXmlStorage(hass, entry_id)
            await storage.async_load()
            domain_data[storage_key] = storage

        return domain_data[storage_key]


def async_evict_xml_storage(hass: HomeAssistant, entry_id: str) -> None:
    """Evict one cached storage object and its initialization lock."""
    domain_data = hass.data.get(DOMAIN)
    if domain_data is None:
        return
    domain_data.pop(f"{DOMAIN}_xml_storage_{entry_id}", None)
    storage_locks = domain_data.get(_STORAGE_LOCKS)
    if storage_locks is not None:
        storage_locks.pop(entry_id, None)
        if not storage_locks:
            domain_data.pop(_STORAGE_LOCKS, None)


async def async_migrate_temp_storage(
    hass: HomeAssistant, unique_id: str, entry_id: str
) -> bool:
    """Migrate temporary XML storage (from config flow) to permanent storage.

    Args:
        hass: Home Assistant instance
        unique_id: The unique_id used to create temp storage during config flow
        entry_id: The actual entry_id to migrate to

    Returns:
        True if migration occurred, False if no temp storage found
    """
    # Create temp storage instance to check for data
    temp_store = Store[dict[str, Any]](
        hass,
        STORAGE_VERSION,
        _get_temp_storage_key(unique_id),
        private=True,
    )

    # Try to load temp data
    temp_data = await temp_store.async_load()

    if not temp_data or not temp_data.get("stored_xmls"):
        _LOGGER.debug("No temporary storage found for unique_id %s", unique_id)
        return False

    _LOGGER.info(
        "[STORAGE MIGRATION] Found temporary storage for unique_id %s, migrating to entry %s",
        unique_id,
        entry_id,
    )

    # Get/create permanent storage
    perm_storage = await async_get_xml_storage(hass, entry_id)

    migrated_count = 0
    for xml_entry in temp_data["stored_xmls"]:
        label = xml_entry.get("label", "imported_data")
        xmls = xml_entry.get("xmls", [])
        if not xmls and "xml" in xml_entry:
            xmls = [xml_entry["xml"]]
        for xml_data in xmls:
            if xml_data and await perm_storage.async_add_xml(xml_data, label):
                migrated_count += 1

    # Delete temporary storage
    await temp_store.async_remove()

    _LOGGER.info(
        "[STORAGE MIGRATION] Successfully migrated %d XML document(s) to "
        ".storage/green_button_xml_%s",
        migrated_count,
        entry_id,
    )

    return True
