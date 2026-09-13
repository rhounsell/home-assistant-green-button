"""Gas sensor entities for the Green Button integration."""

# Keep provider-specific entity flows explicit during this structural split.
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.helpers.device_registry import DeviceInfo

from . import model, scaling, statistics
from ._sensor_common import (
    GreenButtonStatisticsSensor,
    _cost_multiplier,
    _has_interval_readings,
    _legacy_unique_id,
    _schedule_hass_task_from_any_thread,
)
from .const import DOMAIN
from .coordinator import GreenButtonCoordinator
from .statistic_ids import stream_unique_id

_LOGGER = logging.getLogger(__name__)


class GreenButtonGasSensor(GreenButtonStatisticsSensor):
    """Display imported gas consumption in m³."""

    _attr_device_class = SensorDeviceClass.GAS
    _attr_native_unit_of_measurement = "m³"
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
        usage_point_id: str | None = None,
    ) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._usage_point_id = usage_point_id
        self._cached_native_value: float = 0.0  # Initialize to 0 for Energy Dashboard
        self._legacy_unique_id = _legacy_unique_id(
            coordinator.config_entry.entry_id, meter_reading_id, "_gas"
        )
        self._attr_unique_id = (
            stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "_gas",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Usage"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas entities under a dedicated device."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")
            },
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float:
        """Return the cached total for display."""
        return self._cached_native_value

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "m³"

    async def async_added_to_hass(self) -> None:
        """Initialize the display sensor and schedule imported statistics."""
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Gas Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        found_meter_reading = True
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )
                        break
                if found_meter_reading:
                    break

            # If not found as meter reading, check if it's a UsagePoint ID (UsageSummary-only case)
            if not found_meter_reading:
                for usage_point in self.coordinator.data["usage_points"]:
                    if (
                        (
                            self._usage_point_id is None
                            or usage_point.id == self._usage_point_id
                        )
                        and usage_point.id == self._meter_reading_id
                        and usage_point.usage_summaries
                    ):
                        meter_readings = [
                            meter_reading
                            for meter_reading in usage_point.meter_readings
                            if _has_interval_readings(meter_reading)
                        ]
                        if len(meter_readings) == 1:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics(meter_readings[0]),
                            )
                        else:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics_from_summaries(
                                    usage_point
                                ),
                            )
                        break

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update cached values and schedule historical statistics."""
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )
        self._attr_native_value = statistics.gas_usage_total(
            meter_reading, summaries, usage_allocation_mode
        )
        # Run statistics update in background to not block startup
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_statistics_async(
                meter_reading, summaries, usage_allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        usage_allocation_mode: str,
    ) -> None:
        """Update gas statistics in background without blocking."""
        try:
            await statistics.update_gas_statistics(
                self.hass,
                self,
                meter_reading,
                usage_summaries=summaries,
                allocation_mode=usage_allocation_mode,
            )

            self._cached_native_value = statistics.gas_usage_total(
                meter_reading, summaries, usage_allocation_mode
            )

            # Write the state once after statistics import to update the sensor display
            self.async_write_ha_state()

            _LOGGER.info(
                "%s: Gas statistics update completed, state set to %.2f m³.",
                self.entity_id,
                self._cached_native_value,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update failed.",
                self.entity_id,
            )

    async def update_sensor_and_statistics_from_summaries(
        self, usage_point: model.UsagePoint
    ) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no daily MeterReadings)."""
        usage_allocation_mode = (
            self.coordinator.config_entry.options.get("gas_usage_allocation")
            or self.coordinator.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )
        self._attr_native_value = statistics.gas_usage_total(
            None, list(usage_point.usage_summaries), usage_allocation_mode
        )

        if usage_allocation_mode == "monthly_increment" and usage_point.usage_summaries:
            _LOGGER.info(
                "Gas Sensor %s: Generating statistics from UsageSummaries (no daily readings)",
                self.entity_id,
            )
            # Call update_gas_statistics in background - no meter reading available
            statistics.async_schedule_statistics_update(
                self.hass,
                self.coordinator.config_entry,
                lambda: self._update_gas_statistics_from_summaries_async(
                    usage_point, usage_allocation_mode
                ),
            )
            _LOGGER.debug(
                "%s: Gas statistics update (from summaries) scheduled in background.",
                self.entity_id,
            )
        else:
            _LOGGER.warning(
                "Gas Sensor %s: Cannot generate statistics - monthly_increment mode required for UsageSummary-only data",
                self.entity_id,
            )

    async def _update_gas_statistics_from_summaries_async(
        self, usage_point: model.UsagePoint, usage_allocation_mode: str
    ) -> None:
        """Update gas statistics from summaries in background without blocking."""
        try:
            await statistics.update_gas_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                usage_summaries=list(usage_point.usage_summaries),
                allocation_mode=usage_allocation_mode,
            )

            self._cached_native_value = statistics.gas_usage_total(
                None, list(usage_point.usage_summaries), usage_allocation_mode
            )

            # Write the state once after statistics import to update the sensor display
            self.async_write_ha_state()

            _LOGGER.info(
                "%s: Gas statistics update (from summaries) completed, state set to %.2f m³.",
                self.entity_id,
                self._cached_native_value,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas statistics update (from summaries) failed.",
                self.entity_id,
            )


class GreenButtonGasCostSensor(GreenButtonStatisticsSensor):
    """Gas cost sensor (monetary total) using UsageSummary pro-rated per day."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GreenButtonCoordinator,
        meter_reading_id: str,
        usage_point_id: str | None = None,
    ) -> None:
        super().__init__(coordinator)
        self._meter_reading_id = meter_reading_id
        self._usage_point_id = usage_point_id
        self._legacy_unique_id = _legacy_unique_id(
            coordinator.config_entry.entry_id, meter_reading_id, "_gas_cost"
        )
        self._attr_unique_id = (
            stream_unique_id(
                coordinator.config_entry.entry_id,
                usage_point_id,
                meter_reading_id,
                "_gas_cost",
            )
            if usage_point_id is not None
            else self._legacy_unique_id
        )
        # Simple name - Home Assistant will combine with device name since _attr_has_entity_name=True
        self._attr_name = "Cost"
        self._attr_native_unit_of_measurement = "CAD"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device metadata for grouping gas cost under the gas device."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, f"{self.coordinator.config_entry.entry_id}_gas_device")
            },
            name=f"{self.coordinator.config_entry.title} Natural Gas",
            manufacturer="Green Button",
            model="Natural Gas",
        )

    @property
    def native_value(self) -> float | None:
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        if not summaries:
            return 0.0
        if any(summary.total_cost is None for summary in summaries):
            return None
        self._attr_native_unit_of_measurement = summaries[0].currency
        return float(
            sum(
                scaling.usage_summary_cost(us, _cost_multiplier(self.coordinator, True))
                for us in summaries
            )
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Describe the billing-cost allocation used by this display series."""
        attributes = super().extra_state_attributes
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        if allocation_mode == "pro_rate_daily":
            attributes["cost_allocation"] = "estimated_daily_proration"
        else:
            attributes["cost_allocation"] = "billing_period_increment"
        return attributes

    @property
    def name(self) -> str:
        """Return the entity name (delegates to parent SensorEntity for automatic composition)."""
        return super().name  # type: ignore[misc]

    @property
    def native_unit_of_measurement(self) -> str:
        return self._attr_native_unit_of_measurement or "CAD"

    async def async_added_to_hass(self) -> None:
        """Initialize the display sensor and schedule imported statistics."""
        await super().async_added_to_hass()

        _LOGGER.debug(
            "Gas Cost Sensor %s: Entity added to Home Assistant (preparing imported statistics)",
            self.entity_id,
        )

        # Kick off a statistics update if data already exists (e.g., after import)
        if self.coordinator.data and self.coordinator.data.get("usage_points"):
            self._handle_coordinator_update()

    def _handle_coordinator_update(self) -> None:

        if self.coordinator.data and "usage_points" in self.coordinator.data:
            # Try to find as meter reading first
            found_meter_reading = False
            for usage_point in self.coordinator.data["usage_points"]:
                for meter_reading in usage_point.meter_readings:
                    if (
                        self._usage_point_id is None
                        or usage_point.id == self._usage_point_id
                    ) and meter_reading.id == self._meter_reading_id:
                        found_meter_reading = True
                        _schedule_hass_task_from_any_thread(
                            self.hass, self.update_sensor_and_statistics(meter_reading)
                        )
                        break
                if found_meter_reading:
                    break

            # If not found as meter reading, check if it's a UsagePoint ID (UsageSummary-only case)
            if not found_meter_reading:
                for usage_point in self.coordinator.data["usage_points"]:
                    if (
                        (
                            self._usage_point_id is None
                            or usage_point.id == self._usage_point_id
                        )
                        and usage_point.id == self._meter_reading_id
                        and usage_point.usage_summaries
                    ):
                        meter_readings = [
                            meter_reading
                            for meter_reading in usage_point.meter_readings
                            if _has_interval_readings(meter_reading)
                        ]
                        if len(meter_readings) == 1:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics(meter_readings[0]),
                            )
                        else:
                            _schedule_hass_task_from_any_thread(
                                self.hass,
                                self.update_sensor_and_statistics_from_summaries(
                                    usage_point
                                ),
                            )
                        break

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update the gas cost sensor and schedule historical statistics."""
        # Update state
        self._attr_native_value = self.native_value

        # Update long-term statistics with pro-rated daily cost
        # Run in background to not block startup
        summaries = self.coordinator.get_usage_summaries_for_meter_reading(
            self._meter_reading_id, self._usage_point_id
        )
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_cost_statistics_async(
                meter_reading, summaries, allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_async(
        self,
        meter_reading: model.MeterReading,
        summaries: list[model.UsageSummary],
        allocation_mode: str,
    ) -> None:
        """Update gas cost statistics in background without blocking."""
        try:
            gas_multiplier = _cost_multiplier(self.coordinator, True)
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                meter_reading,
                summaries,
                allocation_mode=allocation_mode,
                gas_cost_multiplier=gas_multiplier,
            )
            _LOGGER.info(
                "%s: Gas cost statistics update completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas cost statistics update failed.",
                self.entity_id,
            )

    async def update_sensor_and_statistics_from_summaries(
        self, usage_point: model.UsagePoint
    ) -> None:
        """Update sensor and statistics when only UsageSummaries are available (no MeterReadings)."""
        # Update entity state (sum of all UsageSummary total_cost values)
        if any(us.total_cost is None for us in usage_point.usage_summaries):
            self._attr_native_value = None
            return
        total = sum(
            float(
                scaling.usage_summary_cost(us, _cost_multiplier(self.coordinator, True))
            )
            for us in usage_point.usage_summaries
        )
        self._attr_native_value = total if total > 0 else 0.0

        # Import gas cost statistics
        allocation_mode = (
            self.coordinator.config_entry.options.get("gas_cost_allocation")
            or self.coordinator.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )

        # Force monthly_increment mode for UsageSummary-only data since pro_rate_daily requires daily readings
        if allocation_mode == "pro_rate_daily":
            _LOGGER.info(
                "Gas Cost Sensor %s: Forcing monthly_increment mode (UsageSummary-only data, no daily readings for pro-rating)",
                self.entity_id,
            )
            allocation_mode = "monthly_increment"

        _LOGGER.info(
            "Gas Cost Sensor %s: Generating statistics from UsageSummaries, mode=%s",
            self.entity_id,
            allocation_mode,
        )

        # Call update_gas_cost_statistics in background - no meter reading available
        statistics.async_schedule_statistics_update(
            self.hass,
            self.coordinator.config_entry,
            lambda: self._update_gas_cost_statistics_from_summaries_async(
                usage_point, allocation_mode
            ),
        )
        _LOGGER.debug(
            "%s: Gas cost statistics update (from summaries) scheduled in background.",
            self.entity_id,
        )

    async def _update_gas_cost_statistics_from_summaries_async(
        self, usage_point: model.UsagePoint, allocation_mode: str
    ) -> None:
        """Update gas cost statistics from summaries in background without blocking."""
        try:
            gas_multiplier = _cost_multiplier(self.coordinator, True)
            await statistics.update_gas_cost_statistics(
                self.hass,
                self,
                None,  # No meter reading available
                list(usage_point.usage_summaries),
                allocation_mode=allocation_mode,
                gas_cost_multiplier=gas_multiplier,
            )
            _LOGGER.info(
                "%s: Gas cost statistics update (from summaries) completed.",
                self.entity_id,
            )
        except Exception:
            _LOGGER.exception(
                "%s: Gas cost statistics update (from summaries) failed.",
                self.entity_id,
            )
