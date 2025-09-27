"""A module defining classes that store the component's runtime state."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Collection
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from typing import Protocol
from typing import runtime_checkable

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_platform

from . import const
from . import model
from . import services
from . import side_channel


class PlatformState(Protocol):
    """A protocol for PlatformState objects definined in this module."""

    @property
    def platform(self) -> entity_platform.EntityPlatform:
        """Return the associated EntityPlatform."""

    @property
    def entities(self) -> Collection[GreenButtonEntity]:
        """Return all associated entities."""


@runtime_checkable
class GreenButtonEntity(Protocol):
    """A protocol for Entities created by the component."""

    @property
    def entity_id(self) -> str:
        """Return the entity's entity_id."""

    @property
    def name(self) -> str:
        """Return the entity's name."""

    @property
    def meter_reading_id(self) -> str:
        """Return the MeterReading ID associated with the entity."""

    @property
    def long_term_statistics_id(self) -> str:
        """Return the statistic ID associated with the entity."""

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the entity's native unit of measurement."""

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update the entity's state and statistics to match the reading."""

    async def reset(self) -> None:
        """Reset the entity and its statistics."""


@typing.final
@dataclasses.dataclass(frozen=True)
class SensorPlatformState:
    """An object holding the state for a specific sensor platform."""

    platform: entity_platform.EntityPlatform
    service: services.EntityService
    entities: MutableSequence[GreenButtonEntity]

    def add_sensors(self, entities: Collection[GreenButtonEntity]) -> None:
        """Add sensors to the state."""
        self.entities.extend(entities)

    @classmethod
    async def create(
        cls,
        hass: HomeAssistant,
        entry: ConfigEntry,
        platform: entity_platform.EntityPlatform,
    ) -> SensorPlatformState:
        """Create a new instance."""
        return cls(
            service=await services.EntityService.async_create_and_register(
                hass, platform
            ),
            platform=platform,
            entities=[],
        )


@typing.final
@dataclasses.dataclass(frozen=True)
class NumberPlatformState:
    """An object holding the state for the number platform."""

    platform: entity_platform.EntityPlatform
    service: services.EntityService
    entities: MutableSequence[GreenButtonEntity]

    def add_sensors(self, entities: Collection[GreenButtonEntity]) -> None:
        """Add sensors to the state."""
        self.entities.extend(entities)

    @classmethod
    async def create(
        cls,
        hass: HomeAssistant,
        entry: ConfigEntry,
        platform: entity_platform.EntityPlatform,
    ) -> NumberPlatformState:
        """Create a new instance."""
        return cls(
            service=await services.EntityService.async_create_and_register(
                hass, platform
            ),
            platform=platform,
            entities=[],
        )


@typing.final
@dataclasses.dataclass(frozen=True)
class EntryState:
    """An object holding the state for a ConfigEntry."""

    platform_states: MutableMapping[Platform, PlatformState]

    async def async_setup_sensor_platform(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        platform: entity_platform.EntityPlatform,
    ) -> SensorPlatformState:
        """Set up a SensorPlatformState."""
        platform_state = await SensorPlatformState.create(hass, entry, platform)
        self.platform_states[Platform.SENSOR] = platform_state
        return platform_state

    async def async_setup_number_platform(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        platform: entity_platform.EntityPlatform,
    ) -> NumberPlatformState:
        """Set up a NumberPlatformState."""
        platform_state = await NumberPlatformState.create(hass, entry, platform)
        self.platform_states[Platform.NUMBER] = platform_state
        return platform_state

    @classmethod
    async def create(cls, hass: HomeAssistant, entry: ConfigEntry) -> EntryState:
        """Create a new instance."""
        return cls(platform_states={})


@typing.final
@dataclasses.dataclass(frozen=True)
class State:
    """An object holding the global state for the component."""

    service: services.Service
    side_channels: side_channel.SideChannels
    entry_states: MutableMapping[str, EntryState]

    def get_entry_state(self, entry_unique_id: str) -> EntryState | None:
        """Get the entry state from the unique ID of a ConfigEntry."""
        return self.entry_states.get(entry_unique_id)

    async def async_setup_entry(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> EntryState:
        """Set up the state for a ConfigEntry."""
        unique_id = entry.unique_id
        assert unique_id is not None

        entry_state = await EntryState.create(hass, entry)
        self.entry_states[unique_id] = entry_state
        return entry_state

    async def async_unload_entry(self, entry: ConfigEntry) -> None:
        """Tear down the state for a ConfigEntry."""
        unique_id = entry.unique_id
        assert unique_id is not None

        self.entry_states.pop(unique_id)

    @classmethod
    async def create(cls, hass: HomeAssistant) -> State:
        """Create a new instance."""
        return cls(
            service=await services.Service.async_create_and_register(hass),
            side_channels=side_channel.SideChannels(),
            entry_states={},
        )


def get(hass: HomeAssistant) -> State:
    """Get the component state."""
    return hass.data[const.DOMAIN]


async def async_ensure_setup(hass: HomeAssistant) -> State:
    """Set up integration runtime state if not already done.

    This function is idempotent.
    """
    if const.DOMAIN not in hass.data:
        hass.data[const.DOMAIN] = await State.create(hass)
    return hass.data[const.DOMAIN]
