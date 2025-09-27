"""Config flow for Green Button integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant import config_entries

from . import configs
from . import const

_LOGGER = logging.getLogger(__name__)


class ConfigFlow(config_entries.ConfigFlow, domain=const.DOMAIN):
    """Handle a config flow for Green Button."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step."""
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
        # Store the XML data directly in config entry instead of using side channels
        config_data = dict(config.to_mapping())
        # Add the original XML data for the coordinator to parse
        config_data["xml"] = user_input.get("xml", "")

        return self.async_create_entry(
            title=config.name,
            data=config_data,
        )
