"""Sensor platform for the Green Button integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry

from . import model, statistics
from ._sensor_common import GreenButtonStatisticsSensor, _has_interval_readings
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator
from .electricity_sensor import (
    GreenButtonCostSensor,
    GreenButtonSensor,
    _electricity_cost_total,
    _electricity_usage_total,
)
from .gas_sensor import GreenButtonGasCostSensor, GreenButtonGasSensor

__all__ = [
    "GreenButtonCostSensor",
    "GreenButtonGasCostSensor",
    "GreenButtonGasSensor",
    "GreenButtonSensor",
    "GreenButtonStatisticsSensor",
    "_electricity_cost_total",
    "_electricity_usage_total",
]

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up one stable entity pair for every unambiguous provider stream."""
    coordinator: GreenButtonCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    active_unique_ids: set[str] = set()

    def _eligible_meter_readings(
        usage_point: model.UsagePoint,
    ) -> list[model.MeterReading]:
        return sorted(
            (
                meter_reading
                for meter_reading in usage_point.meter_readings
                if _has_interval_readings(meter_reading)
            ),
            key=lambda meter_reading: meter_reading.id,
        )

    def _migrate_legacy_entities(
        entity_registry: Any, candidates: list[GreenButtonStatisticsSensor]
    ) -> None:
        """Move an unambiguous suffix-based entity and its external series."""
        by_legacy_id: dict[str, list[GreenButtonStatisticsSensor]] = {}
        for candidate in candidates:
            if candidate.unique_id != candidate.legacy_unique_id:
                by_legacy_id.setdefault(candidate.legacy_unique_id, []).append(
                    candidate
                )

        for legacy_unique_id, matching_candidates in by_legacy_id.items():
            if len(matching_candidates) != 1:
                _LOGGER.warning(
                    "Not migrating ambiguous legacy Green Button stream ID %s",
                    legacy_unique_id,
                )
                continue

            candidate = matching_candidates[0]
            if entity_registry.async_get_entity_id(
                "sensor", DOMAIN, candidate.unique_id
            ):
                continue
            if not (
                legacy_entity_id := entity_registry.async_get_entity_id(
                    "sensor", DOMAIN, legacy_unique_id
                )
            ):
                continue

            entity_registry.async_update_entity(
                legacy_entity_id, new_unique_id=candidate.unique_id
            )
            statistics.rename_external_statistic(
                hass,
                candidate.legacy_unique_id,
                candidate.unique_id,
            )
            _LOGGER.info(
                "Migrated Green Button stream %s to full provider identity",
                legacy_entity_id,
            )

    def _async_create_entities() -> None:
        """Create entities for all streams currently available from the provider."""
        if not coordinator.data or not coordinator.data.get("usage_points"):
            return

        candidates: list[GreenButtonStatisticsSensor] = []
        for usage_point in sorted(coordinator.usage_points, key=lambda point: point.id):
            meter_readings = _eligible_meter_readings(usage_point)
            if usage_point.sensor_device_class == SensorDeviceClass.GAS:
                allocation_mode = (
                    entry.options.get("gas_usage_allocation")
                    or entry.data.get("gas_usage_allocation")
                    or "daily_readings"
                )
                if (
                    allocation_mode == "monthly_increment"
                    and usage_point.usage_summaries
                ):
                    summary_stream_id = (
                        meter_readings[0].id
                        if len(meter_readings) == 1
                        else usage_point.id
                    )
                    candidates.extend(
                        (
                            GreenButtonGasSensor(
                                coordinator, summary_stream_id, usage_point.id
                            ),
                            GreenButtonGasCostSensor(
                                coordinator, summary_stream_id, usage_point.id
                            ),
                        )
                    )
                    continue

                for meter_reading in meter_readings:
                    candidates.append(
                        GreenButtonGasSensor(
                            coordinator, meter_reading.id, usage_point.id
                        )
                    )

                if len(meter_readings) == 1:
                    candidates.append(
                        GreenButtonGasCostSensor(
                            coordinator, meter_readings[0].id, usage_point.id
                        )
                    )
                elif meter_readings and usage_point.usage_summaries:
                    _LOGGER.warning(
                        "Skipping gas cost entities for UsagePoint %s: its billing "
                        "summary cannot be attributed to one of %d meter streams",
                        usage_point.id,
                        len(meter_readings),
                    )
                continue

            for meter_reading in meter_readings:
                candidates.extend(
                    (
                        GreenButtonSensor(
                            coordinator, meter_reading.id, usage_point.id
                        ),
                        GreenButtonCostSensor(
                            coordinator, meter_reading.id, usage_point.id
                        ),
                    )
                )

        if not candidates:
            return

        entity_registry = async_get_entity_registry(hass)
        _migrate_legacy_entities(entity_registry, candidates)
        new_entities = [
            candidate
            for candidate in candidates
            if candidate.unique_id not in active_unique_ids
        ]
        if not new_entities:
            return

        async_add_entities(new_entities)
        active_unique_ids.update(entity.unique_id for entity in new_entities)

    _async_create_entities()
    entry.async_on_unload(coordinator.async_add_listener(_async_create_entities))
