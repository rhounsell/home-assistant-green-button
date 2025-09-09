"""A module containing the component's service implementations."""
from __future__ import annotations

import abc
import asyncio
import dataclasses
import datetime
import enum
import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Any
from typing import final
from typing import Protocol

import voluptuous as vol
from homeassistant.components.recorder import statistics
from homeassistant.components.recorder import util as recorder_util
from homeassistant.core import HomeAssistant
from homeassistant.core import ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity as ha_entity
from homeassistant.helpers import entity_platform
from homeassistant.helpers import service
from homeassistant.helpers.typing import ServiceDataType

from . import const
from . import model
from . import state
from .parsers import espi

_LOGGER = logging.getLogger(__name__)


class _ServiceParameter(enum.Enum):
    @classmethod
    def create_schema(cls) -> vol.Schema:
        """Create a schema from all parameters in the enum."""
        return vol.Schema({e.value[0]: e.value[1] for e in cls})

    def extract(self, obj: ServiceDataType) -> Any:
        """Extract the value of this parameter from the service call data."""
        key = self.value[0]
        key = key if isinstance(key, str) else key.schema
        return obj.get(key)


class _ServiceData:
    """A wrapper around service call data to to work with _ServiceParameter."""

    def __init__(self, data: ServiceDataType) -> None:
        """Initialize the instance."""
        self._data = data

    def get(self, key: _ServiceParameter) -> Any:
        """Get the value associated with the key."""
        res = key.extract(self._data)
        if isinstance(res, dict):
            return _ServiceData(res)
        return res

    def __getitem__(self, key: _ServiceParameter) -> Any:
        """Get the value associated with the key."""
        return self.get(key)

    def __str__(self) -> str:
        return str(self._data)


@dataclasses.dataclass(frozen=True)
class ServiceActionSpec:
    """A dataclass containing metadata for a component service action."""

    name: str
    schema: vol.Schema
    factory: Callable[[HomeAssistant, _ServiceData], ServiceAction]


class ServiceAction(Protocol):
    """The protocol for a component service action."""

    async def __call__(self) -> None:
        """Execute the action."""


class _DeleteStatisticsParameter(_ServiceParameter):
    STATISTICS_ID = (
        vol.Required("statistic_id"),
        cv.matches_regex(f"{re.escape(const.DOMAIN)}:[a-zA-Z_]+"),
    )


class _DeleteStatisticsAction:
    def __init__(self, hass: HomeAssistant, data: _ServiceData) -> None:
        self._hass = hass
        self._data = data

    @classmethod
    def create_spec(cls) -> ServiceActionSpec:
        """Create the spec for this action."""
        return ServiceActionSpec(
            name="delete_statistics",
            schema=_DeleteStatisticsParameter.create_schema(),
            factory=cls,
        )

    async def _import(self, usage_point: model.UsagePoint) -> None:
        _LOGGER.info("Processing UsagePoint %r", usage_point.id)
        entry_state = state.get(self._hass).get_entry_state(usage_point.id)
        if entry_state is None:
            return

        for platform_state in entry_state.platform_states.values():
            for entity in platform_state.entities:
                meter_reading = usage_point.get_meter_reading_by_id(
                    entity.meter_reading_id
                )
                if meter_reading is None:
                    continue
                await entity.update_sensor_and_statistics(meter_reading)

    async def __call__(self) -> None:
        stats_id = self._data[_DeleteStatisticsParameter.STATISTICS_ID]
        recorder_util.get_instance(self._hass).async_clear_statistics([stats_id])


class _ImportEspiXmlParameter(_ServiceParameter):
    XML = (vol.Required("xml"), str)


class _ImportEspiXmlAction:
    def __init__(self, hass: HomeAssistant, data: _ServiceData) -> None:
        self._hass = hass
        self._data = data

    @classmethod
    def create_spec(cls) -> ServiceActionSpec:
        """Create the spec for this action."""
        return ServiceActionSpec(
            name="import_espi_xml",
            schema=_ImportEspiXmlParameter.create_schema(),
            factory=cls,
        )

    async def _import(self, usage_point: model.UsagePoint) -> None:
        _LOGGER.info("Processing UsagePoint %r", usage_point.id)
        entry_state = state.get(self._hass).get_entry_state(usage_point.id)
        if entry_state is None:
            return

        for platform_state in entry_state.platform_states.values():
            for entity in platform_state.entities:
                meter_reading = usage_point.get_meter_reading_by_id(
                    entity.meter_reading_id
                )
                if meter_reading is None:
                    continue
                await entity.update_sensor_and_statistics(meter_reading)

    async def __call__(self) -> None:
        xml = self._data[_ImportEspiXmlParameter.XML]
        usage_points = espi.parse_xml(xml)

        ids = [usage_point.id for usage_point in usage_points]
        _LOGGER.info("Found %d UsagePoints with ids: %s", len(ids), ids)

        await asyncio.gather(
            *(self._import(usage_point) for usage_point in usage_points)
        )


class Service:
    """An object that manages all component service actions."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the service."""
        self._hass = hass

    async def _async_register_service(self, spec: ServiceActionSpec) -> None:
        async def handler(call: ServiceCall) -> None:
            return await spec.factory(self._hass, _ServiceData(call.data))()

        self._hass.services.async_register(
            const.DOMAIN,
            spec.name,
            service.verify_domain_control(self._hass, const.DOMAIN)(handler),
            schema=spec.schema,
        )

    async def async_register(self) -> None:
        """Register handlers for the service."""
        await self._async_register_service(_DeleteStatisticsAction.create_spec())
        await self._async_register_service(_ImportEspiXmlAction.create_spec())

    @classmethod
    async def async_create_and_register(cls, hass: HomeAssistant) -> "Service":
        """Create a new instance and registers it."""
        new_service = cls(hass)
        await new_service.async_register()
        return new_service


@dataclasses.dataclass(frozen=False)
class _EntityServiceAction(abc.ABC):
    """The base class for all entity service actions."""

    hass: HomeAssistant
    entity: state.GreenButtonEntity
    data: _ServiceData

    @abc.abstractmethod
    async def __call__(self) -> None:
        """Execute the action."""


@final
@dataclasses.dataclass(frozen=True)
class _EntityServiceActionSpec:
    """A dataclass containing metadata for a component entity service action."""

    name: str
    schema: vol.Schema
    factory: Callable[
        [HomeAssistant, state.GreenButtonEntity, _ServiceData], _EntityServiceAction
    ]


class _LogStatisticsParameter(_ServiceParameter):
    START = (vol.Required("start"), cv.datetime)
    END = (vol.Required("end"), cv.datetime)


class _LogStatisticsAction(_EntityServiceAction):
    @classmethod
    def create_spec(cls) -> _EntityServiceActionSpec:
        """Create the spec for this action."""
        return _EntityServiceActionSpec(
            name="log_statistics",
            schema=_LogStatisticsParameter.create_schema(),
            factory=cls,
        )

    async def __call__(self) -> None:
        start: datetime.datetime = self.data.get(_LogStatisticsParameter.START)
        end: datetime.datetime = self.data.get(_LogStatisticsParameter.END)

        local_tz = datetime.datetime.now().astimezone().tzinfo
        start = start.replace(tzinfo=local_tz)
        end = end.replace(tzinfo=local_tz)

        def round_down_5m(datetime_val: datetime.datetime) -> datetime.datetime:
            return datetime_val.replace(
                minute=datetime_val.minute - (datetime_val.minute % 5),
                second=0,
                microsecond=0,
            )

        def action():
            data_hour = statistics.statistics_during_period(
                hass=self.hass,
                start_time=start,
                end_time=end,
                statistic_ids=[self.entity.entity_id],
                period="hour",
            )
            data_5_min = statistics.statistics_during_period(
                hass=self.hass,
                start_time=start,
                end_time=end,
                statistic_ids=[self.entity.entity_id],
                period="5minute",
            )
            data_5_min_mod = statistics.statistics_during_period(
                hass=self.hass,
                start_time=round_down_5m(start - datetime.timedelta.resolution),
                end_time=round_down_5m(end - datetime.timedelta.resolution),
                statistic_ids=[self.entity.entity_id],
                period="5minute",
            )
            data_single = statistics.statistic_during_period(
                hass=self.hass,
                start_time=start,
                end_time=end,
                statistic_id=self.entity.entity_id,
                types=None,
                units=None,
            )

            data_before = statistics.statistic_during_period(
                hass=self.hass,
                start_time=None,
                end_time=start + datetime.timedelta.resolution,
                statistic_id=self.entity.entity_id,
                types=None,
                units=None,
            )

            _LOGGER.info("Hourly data: %s", json.dumps(data_hour))
            _LOGGER.info("5m data: %s", json.dumps(data_5_min))
            _LOGGER.info(
                "%s to %s",
                round_down_5m(start - datetime.timedelta.resolution),
                round_down_5m(end),
            )
            _LOGGER.info("5m data (modified): %s", json.dumps(data_5_min_mod))
            _LOGGER.info("Single data: %s", json.dumps(data_single))
            _LOGGER.info("Before data: %s", json.dumps(data_before))

        recorder_util.get_instance(self.hass).async_add_executor_job(action)


@final
@dataclasses.dataclass(frozen=False)
class _ResetEntityAction(_EntityServiceAction):
    @classmethod
    def create_spec(cls) -> _EntityServiceActionSpec:
        """Create the spec for this action."""
        return _EntityServiceActionSpec(
            name="reset", schema=vol.Schema({}), factory=cls
        )

    async def __call__(self) -> None:
        await self.entity.reset()


def entity_service(
    func: Callable[
        [EntityService, state.GreenButtonEntity, ServiceCall], Awaitable[None]
    ]
) -> Callable[
    [EntityService, ha_entity.Entity, ServiceCall], Coroutine[Any, Any, None]
]:
    """Decorate entity service actions and filters out any entities that are not GreenButtonEntity."""

    async def handler(
        self: EntityService, entity: ha_entity.Entity, call: ServiceCall
    ) -> None:
        if not isinstance(entity, state.GreenButtonEntity):
            return None
        return await func(self, entity, call)

    return handler


class EntityService:
    """An object that manages all entity service actions for a specific EntityPlatform."""

    _VALUE_PARAMETER = "value"
    _XML_PARAMETER = "xml"

    def __init__(
        self, hass: HomeAssistant, platform: entity_platform.EntityPlatform
    ) -> None:
        """Initialize the service."""
        self._hass = hass
        self._platform = platform

    def _async_register_admin_entity_service(
        self, spec: _EntityServiceActionSpec
    ) -> None:
        hass = self._hass
        platform = self._platform
        platform_name = platform.platform_name

        if hass.services.has_service(platform_name, spec.name):
            return

        async def entity_handler(entity: ha_entity.Entity, call: ServiceCall) -> None:
            """Handle the entity service call."""
            if not isinstance(entity, state.GreenButtonEntity):
                return None
            return await spec.factory(self._hass, entity, _ServiceData(call.data))()

        async def handle_service(call: ServiceCall) -> None:
            """Handle the service calls."""
            await service.entity_service_call(
                hass, [platform], entity_handler, call, None
            )

        schema = cv.make_entity_service_schema(spec.schema.schema)
        service.async_register_admin_service(
            hass, platform_name, spec.name, handle_service, schema
        )

    async def async_register(self) -> None:
        """Register all actions."""
        self._async_register_admin_entity_service(_LogStatisticsAction.create_spec())
        self._async_register_admin_entity_service(_ResetEntityAction.create_spec())

    @classmethod
    async def async_create_and_register(
        cls, hass: HomeAssistant, platform: entity_platform.EntityPlatform
    ) -> "EntityService":
        """Create a new instance and registers it."""
        new_service = cls(hass, platform)
        await new_service.async_register()
        return new_service
