"""Platform for creating a sensor containing the last known update."""
# TODO delete this file
from __future__ import annotations

import asyncio
import datetime
import decimal
import enum
from functools import cached_property
import logging
from typing import Final

import slugify
from homeassistant.components import number
from homeassistant.components.recorder import util as recorder_util
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
# from homeassistant.helpers import entity
from homeassistant.const import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import configs
from . import const
from . import model
from . import state
from . import statistics

_LOGGER = logging.getLogger(__name__)


_MOST_RECENT_ENERGY_READING_NUMBER_DESCRIPTION = number.NumberEntityDescription(
    key="green_button_last_energy_reading",
    entity_category=EntityCategory.DIAGNOSTIC,
    native_step=1,
    icon="mdi:lightning-bolt",
)

_MOST_RECENT_COST_READING_NUMBER_DESCRIPTION = number.NumberEntityDescription(
    key="green_button_most_recent_cost_reading",
    entity_category=EntityCategory.DIAGNOSTIC,
    native_step=float(10**-5),  # Hundred thousandth
    icon="mdi:cash",
)


class _ReadingVariant(enum.Enum):
    """Represents either a meter reading or its cost.

    Used to abstract away specifics reading those two values from a MeterReading.
    """

    READING = "reading"
    COST = "cost"

    def get_entity_description(self) -> number.NumberEntityDescription:
        """Return the EntityDescription for the variant."""
        if self == _ReadingVariant.COST:
            return _MOST_RECENT_COST_READING_NUMBER_DESCRIPTION
        if self == _ReadingVariant.READING:
            return _MOST_RECENT_ENERGY_READING_NUMBER_DESCRIPTION
        raise ValueError(f"Invalid number variant: {repr(self)}")

    def get_unit_of_measurement(self, config: configs.MeterReadingConfig) -> str:
        """Return the unit of measurement for the variant."""
        if self == _ReadingVariant.COST:
            return config.currency
        if self == _ReadingVariant.READING:
            return config.unit_of_measurement
        raise ValueError(f"Invalid number variant: {repr(self)}")

    def get_native_value(
        self, interval_reading: model.IntervalReading
    ) -> decimal.Decimal:
        """Return the native value for the variant in the IntervalReading."""
        if self == _ReadingVariant.COST:
            return decimal.Decimal(interval_reading.cost) / 100000
        if self == _ReadingVariant.READING:
            scale = 10**interval_reading.reading_type.power_of_ten_multiplier
            return decimal.Decimal(interval_reading.value) * scale
        raise ValueError(f"Invalid number variant: {repr(self)}")


class _GreenButtonNumber(number.RestoreNumber):
    _METER_READING_ID_ATTR: Final = "meter_reading_id"
    _LONG_TERM_STATISTICS_ID_ATTR: Final = "long_term_statistics_id"
    _LAST_RESET_ATTR: Final = "last_reset"
    _LAST_REPORTED_ATTR: Final = "last_reported"

    def __init__(
        self,
        name: str,
        config: configs.MeterReadingConfig,
        variant: _ReadingVariant,
    ) -> None:
        super().__init__()
        self._variant = variant
        self._initial_meter_reading = config.initial_meter_reading

        self._attr_unique_id = f"{config.id}_{variant.value}"
        self._attr_name = name
        self.entity_description = variant.get_entity_description()
        self._attr_mode = number.NumberMode.BOX
        self._attr_should_poll = False
        self._attr_assumed_state = True
        self._attr_native_unit_of_measurement = variant.get_unit_of_measurement(config)
        long_term_stats_id = slugify.slugify(name, separator="_")
        self._attr_extra_state_attributes = {
            self._METER_READING_ID_ATTR: config.id,
            self._LONG_TERM_STATISTICS_ID_ATTR: f"{const.DOMAIN}:{long_term_stats_id}",
            self._LAST_REPORTED_ATTR: None,
            self._LAST_RESET_ATTR: None,
        }
        self._attr_native_value = 0.0

    @cached_property
    def name(self) -> str:
        name = super().name
        # assert name is not None
        if not isinstance(name, str):
            raise ValueError("Entity name is undefined")
        return name

    @cached_property
    def native_unit_of_measurement(self) -> str:
        native_unit_of_measurement = super().native_unit_of_measurement
        # assert native_unit_of_measurement is not None
        if not isinstance(native_unit_of_measurement, str):
            raise ValueError("native_unit_of_measurement is undefined")
        return native_unit_of_measurement

    @property
    def meter_reading_id(self) -> str:
        """Return the MeterReading ID associated with the entity."""
        return self._attr_extra_state_attributes[self._METER_READING_ID_ATTR]

    @property
    def long_term_statistics_id(self) -> str:
        """Return the statistic ID associated with the entity."""
        return self._attr_extra_long_term_statistics_id

    @property
    def _attr_extra_long_term_statistics_id(self) -> str:
        return self._attr_extra_state_attributes[self._LONG_TERM_STATISTICS_ID_ATTR]

    @_attr_extra_long_term_statistics_id.setter
    def _attr_extra_long_term_statistics_id(self, value: str) -> None:
        self._attr_extra_state_attributes[self._LONG_TERM_STATISTICS_ID_ATTR] = value

    @property
    def _attr_extra_last_reported(self) -> datetime.datetime | None:
        return self._attr_extra_state_attributes.get(self._LAST_REPORTED_ATTR)

    @_attr_extra_last_reported.setter
    def _attr_extra_last_reported(self, value: datetime.datetime | None) -> None:
        self._attr_extra_state_attributes[self._LAST_REPORTED_ATTR] = value

    @property
    def _attr_extra_last_reset(self) -> datetime.datetime | None:
        return self._attr_extra_state_attributes.get(self._LAST_RESET_ATTR)

    @_attr_extra_last_reset.setter
    def _attr_extra_last_reset(self, value: datetime.datetime | None) -> None:
        self._attr_extra_state_attributes[self._LAST_RESET_ATTR] = value

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._initial_meter_reading is not None:
            await self.update_sensor_and_statistics(self._initial_meter_reading)
            return

        last_sensor_data = await self.async_get_last_number_data()
        if last_sensor_data is not None:
            self._attr_native_value = last_sensor_data.native_value

        last_state = await self.async_get_last_state()
        if last_state is not None:
            attributes = last_state.attributes
            if attributes.get(self._LAST_REPORTED_ATTR) is not None:
                self._attr_extra_last_reported = datetime.datetime.fromisoformat(
                    attributes[self._LAST_REPORTED_ATTR]
                )
            if attributes.get(self._LAST_RESET_ATTR) is not None:
                self._attr_extra_last_reset = datetime.datetime.fromisoformat(
                    attributes[self._LAST_RESET_ATTR]
                )
            if attributes.get(self._LONG_TERM_STATISTICS_ID_ATTR) is not None:
                self._attr_extra_long_term_statistics_id = attributes[
                    self._LONG_TERM_STATISTICS_ID_ATTR
                ]

    async def async_will_remove_from_hass(self) -> None:
        stats_to_clear = [self.entity_id, self.long_term_statistics_id]
        _LOGGER.info("[%s] Clearing statistics: %s", self.entity_id, stats_to_clear)
        recorder_util.get_instance(self.hass).async_clear_statistics(stats_to_clear)

    def set_native_value(self, value: float) -> None:
        self._attr_native_value = value
        self.async_write_ha_state()

    def _update_sensor(self, newest_reading: model.IntervalReading) -> None:
        last_reported = self._attr_extra_last_reported
        if last_reported is not None and last_reported >= newest_reading.end:
            return
        native_value = float(self._variant.get_native_value(newest_reading))
        _LOGGER.info(
            "[%s] Updating state for entity to %s", self.entity_id, native_value
        )
        if (
            self._attr_extra_last_reset is not None
            and self._attr_extra_last_reset >= newest_reading.start
        ):
            _LOGGER.warning(
                "Sensor %s has a last reported date (%s) that is older than "
                "the newest known reading (%s) but somehow has a reset date "
                "(%s) that is newer than that same reading (%s). This should "
                "not be possible",
                self.entity_id,
                last_reported,
                newest_reading.end,
                self._attr_extra_last_reset,
                newest_reading.start,
            )
        self._attr_extra_last_reset = newest_reading.start
        self._attr_native_value = native_value
        self._attr_extra_last_reported = newest_reading.end
        self.async_write_ha_state()

    async def update_sensor_and_statistics(
        self, meter_reading: model.MeterReading
    ) -> None:
        """Update the entity's state and statistics to match the reading."""
        newest_interval_reading = meter_reading.get_newest_interval_reading()
        if newest_interval_reading is not None:
            self._update_sensor(newest_interval_reading)
        await statistics.update_statistics(
            hass=self.hass,
            entity=self,
            data_extractor=self._variant,
            meter_reading=meter_reading,
        )

    async def reset(self) -> None:
        """Reset the entity and its statistics."""
        await asyncio.gather(
            self.async_set_native_value(0),
            statistics.clear_statistic(self.hass, self.long_term_statistics_id),
        )

    # added because it's declared as abstract in the base class
    async def async_set_value(self, value: float) -> None:
        """Set the value of the number entity."""
        self._attr_native_value = value # pylint: disable=protected-access # Access to protected member is intentional
        # Add any additional logic here if needed

async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Configure sensors for the ConfigEntry."""
    platform = entity_platform.async_get_current_platform()

    unique_id = entry.unique_id
    assert unique_id is not None

    entry_state = state.get(hass).get_entry_state(unique_id)
    assert entry_state is not None

    platform_state = await entry_state.async_setup_number_platform(
        hass, entry, platform
    )

    config = configs.ComponentConfig.from_entry(hass, entry)
    entities = []
    for i, meter_reading_config in enumerate(config.meter_reading_configs):
        _LOGGER.info(
            "Setting up sensors for meter reading: %r", meter_reading_config.id
        )
        name_prefix = f"{config.name} {i+1}"
        if meter_reading_config.id.startswith(config.unique_id):
            meter_reading_suffix = meter_reading_config.id[len(config.unique_id) :]
            if meter_reading_suffix.startswith("/"):
                meter_reading_suffix = meter_reading_suffix[1:]
            if meter_reading_suffix:
                name_prefix = f"{config.name} {meter_reading_suffix}"

        entities.append(
            _GreenButtonNumber(
                f"{name_prefix} Reading",
                meter_reading_config,
                _ReadingVariant.READING,
            )
        )
        entities.append(
            _GreenButtonNumber(
                f"{name_prefix} Cost",
                meter_reading_config,
                _ReadingVariant.COST,
            )
        )

    platform_state.add_sensors(entities)
    async_add_entities(entities)
