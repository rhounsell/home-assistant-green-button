"""Separate storage for Green Button XML data.

Uses a dedicated Store instance instead of config entry data to handle
large XML files properly. Config entries use delayed writes and are not
designed for multi-MB data storage.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}_xml"


def _get_storage_key(entry_id: str) -> str:
    """Get the storage key for a config entry's XML data."""
    return f"{STORAGE_KEY_PREFIX}_{entry_id}"


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

    async def async_load(self) -> dict[str, Any]:
        """Load stored XML data from disk."""
        if self._data is None:
            self._data = await self._store.async_load() or {"stored_xmls": []}
        return self._data

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save XML data to disk immediately."""
        self._data = data
        await self._store.async_save(data)
        _LOGGER.info("Saved XML data to storage file for entry %s", self.entry_id)

    def async_delay_save(self, data: dict[str, Any], delay: float = 1.0) -> None:
        """Schedule a delayed save of XML data."""
        self._data = data
        self._store.async_delay_save(lambda: data, delay)
        _LOGGER.debug("Scheduled delayed save of XML data for entry %s", self.entry_id)

    async def async_remove(self) -> None:
        """Remove the storage file."""
        await self._store.async_remove()
        self._data = None
        _LOGGER.info("Removed XML storage file for entry %s", self.entry_id)

    def get_stored_xmls(self) -> list[dict[str, Any]]:
        """Get the stored XMLs list from cached data."""
        if self._data is None:
            return []
        return self._data.get("stored_xmls", [])

    async def async_add_xml(self, xml_data: str, label: str) -> None:
        """Add or merge XML data with a label.
        
        If a label already exists, the new XML is appended to the list.
        If the label is new, a new entry is created.
        """
        data = await self.async_load()
        stored_xmls = data.get("stored_xmls", [])

        # Migrate old format entries (single "xml" key) to new format ("xmls" list)
        for entry in stored_xmls:
            if "xml" in entry and "xmls" not in entry:
                entry["xmls"] = [entry.pop("xml")]

        # Check if an entry with this label already exists
        existing_index = next(
            (i for i, entry in enumerate(stored_xmls) if entry.get("label") == label),
            None
        )

        if existing_index is not None:
            # Merge with existing: append new XML to the list for this label
            existing_entry = stored_xmls[existing_index]
            existing_xmls = existing_entry.get("xmls", [])
            # Also handle old format
            if "xml" in existing_entry and not existing_xmls:
                existing_xmls = [existing_entry["xml"]]
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
        await self.async_save(data)
        _LOGGER.info("Stored %d label(s) in XML storage", len(stored_xmls))

    async def async_clear_label(self, label: str | None = None) -> tuple[int, int]:
        """Clear stored XMLs for a specific label or all labels.
        
        Returns tuple of (removed_count, remaining_count).
        """
        data = await self.async_load()
        stored_xmls = data.get("stored_xmls", [])

        if not stored_xmls:
            return (0, 0)

        if label is None:
            # Clear all
            removed_count = len(stored_xmls)
            data["stored_xmls"] = []
            await self.async_save(data)
            return (removed_count, 0)
        else:
            # Clear specific label
            original_count = len(stored_xmls)
            stored_xmls = [x for x in stored_xmls if x.get("label") != label]
            removed_count = original_count - len(stored_xmls)

            if removed_count > 0:
                data["stored_xmls"] = stored_xmls
                await self.async_save(data)

            return (removed_count, len(stored_xmls))


async def async_get_xml_storage(hass: HomeAssistant, entry_id: str) -> GreenButtonXmlStorage:
    """Get or create an XML storage instance for a config entry."""
    storage_key = f"{DOMAIN}_xml_storage_{entry_id}"

    if storage_key not in hass.data.setdefault(DOMAIN, {}):
        storage = GreenButtonXmlStorage(hass, entry_id)
        await storage.async_load()  # Pre-load the data
        hass.data[DOMAIN][storage_key] = storage

    return hass.data[DOMAIN][storage_key]
