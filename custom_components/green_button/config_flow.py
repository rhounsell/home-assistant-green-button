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
        import voluptuous as vol
        import os
        step_id = "user"
        # Add xml_file_path to the schema
        schema = configs.ComponentConfig.make_config_entry_step_schema(user_input).extend({
            vol.Required("xml_file_path"): str,
        })
        errors = {}
        xml_content = ""
        if user_input is not None:
            xml_path = user_input.get("xml_file_path", "")
            if not xml_path or not os.path.isfile(xml_path):
                errors["xml_file_path"] = "file_not_found"
            else:
                try:
                    with open(xml_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()
                except Exception:
                    errors["xml_file_path"] = "file_read_error"

        if user_input is None or errors:
            return self.async_show_form(
                step_id=step_id,
                data_schema=schema,
                errors=errors,
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
        config_data = dict(config.to_mapping())
        config_data["xml"] = xml_content

        return self.async_create_entry(
            title=config.name,
            data=config_data,
        )
