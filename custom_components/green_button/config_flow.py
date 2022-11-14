"""Config flow for Green Button integration."""
from __future__ import annotations

import logging
from homeassistant import config_entries
from homeassistant import data_entry_flow
from typing import Any

from . import configs
from . import const
from . import state

_LOGGER = logging.getLogger(__name__)


class ConfigFlow(config_entries.ConfigFlow, domain=const.DOMAIN):
    """Handle a config flow for Green Button."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        """Handle the initial step."""
        await state.async_ensure_setup(self.hass)

        step_id = "user"
        schema = configs.ComponentConfig.make_config_entry_step_schema(user_input)
        if user_input is None:
            return self.async_show_form(
                step_id=step_id,
                data_schema=schema,
            )

        try:
            config = configs.ComponentConfig.from_mapping(user_input)
        except configs.InvalidUserInputError as ex:
            _LOGGER.info("Invalid user input", exc_info=True)
            return self.async_show_form(
                step_id=step_id,
                data_schema=schema,
                errors=ex.errors,
            )

        if await self.async_set_unique_id(config.unique_id) is not None:
            _LOGGER.info(
                "A ConfigEntry with the unique ID %r is already configured",
                config.unique_id,
            )
            return self.async_abort(reason="already_configured")

        _LOGGER.info("Created config with unique ID %r", config.unique_id)
        config.set_side_channels(self.hass)
        return self.async_create_entry(
            title=config.name,
            data=config.to_mapping(),
        )
