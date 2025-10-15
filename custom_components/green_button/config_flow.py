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
        from homeassistant.helpers import selector
        import os
        step_id = "user"
        
        # Build a custom schema where XML field is optional
        if user_input is None:
            user_input_default = {
                "name": "Home",
            }
        else:
            user_input_default = user_input
            
        schema = vol.Schema(
            {
                vol.Required(
                    "name",
                    default=user_input_default.get("name"),
                ): str,
                vol.Optional(
                    "xml",
                    default=user_input_default.get("xml", ""),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        multiline=True,
                    )
                ),
                vol.Optional("xml_file_path", default=""): str,
            }
        )
        
        errors = {}
        
        if user_input is not None:
            xml_path = user_input.get("xml_file_path", "").strip()
            xml_content = user_input.get("xml", "").strip()
            
            # Check that at least one is provided
            if not xml_path and not xml_content:
                errors["base"] = "no_xml_provided"
            # If both are provided, use the file path
            elif xml_path and xml_content:
                errors["base"] = "both_xml_provided"
            # If file path is provided, read it
            elif xml_path:
                if not os.path.isfile(xml_path):
                    errors["xml_file_path"] = "file_not_found"
                else:
                    try:
                        with open(xml_path, "r", encoding="utf-8") as f:
                            xml_content = f.read()
                            user_input["xml"] = xml_content
                    except Exception:
                        errors["xml_file_path"] = "file_read_error"
            # If XML content is provided directly, use it (already in user_input["xml"])

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
        # Store the XML content from user_input (which now has the file content if path was provided)
        config_data["xml"] = user_input.get("xml", "")

        return self.async_create_entry(
            title=config.name,
            data=config_data,
        )
