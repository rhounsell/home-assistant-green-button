"""Config flow for Green Button integration."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from homeassistant import config_entries
import voluptuous as vol
from homeassistant.helpers import selector

from . import configs
from . import const

_LOGGER = logging.getLogger(__name__)


class ConfigFlow(config_entries.ConfigFlow, domain=const.DOMAIN):
    """Handle a config flow for Green Button."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)


    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step."""
        step_id = "user"
        
        # Build a custom schema where XML field is optional
        if user_input is None:
            user_input_default = {
                "name": "Home",
                "input_type": "file",
                "gas_cost_allocation": "pro_rate_daily",
                "gas_usage_allocation": "daily_readings",
            }
        else:
            user_input_default = user_input
            
        schema = vol.Schema(
            {
                vol.Required(
                    "name",
                    default=user_input_default.get("name"),
                ): str,
                vol.Required(
                    "input_type",
                    default=user_input_default.get("input_type", "file"),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["file", "xml"],
                        mode="list",
                    )
                ),
                vol.Optional(
                    "xml",
                    default=user_input_default.get("xml", ""),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        multiline=True,
                    )
                ),
                vol.Optional("xml_file_path", default=""): str,
                vol.Optional(
                    "gas_cost_allocation",
                    default=user_input_default.get("gas_cost_allocation", "pro_rate_daily"),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "pro_rate_daily", "label": "Pro-rate gas costs daily"},
                            {"value": "monthly_increment", "label": "Single monthly cost increment"},
                        ],
                        mode="list",
                    )
                ),
                vol.Optional(
                    "gas_usage_allocation",
                    default=user_input_default.get("gas_usage_allocation", "daily_readings"),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "daily_readings", "label": "Use daily readings (m³)"},
                            {"value": "monthly_increment", "label": "Single billing-period usage increment (m³)"},
                        ],
                        mode="list",
                    )
                ),
            }
        )
        
        errors = {}
        
        if user_input is not None:
            input_type = user_input.get("input_type")
            xml_path = user_input.get("xml_file_path", "").strip()
            xml_content = user_input.get("xml", "").strip()
            
            # Validate selection
            if input_type not in ("file", "xml"):
                errors["input_type"] = "input_type_required"
                errors.setdefault("base", "input_type_required")
            elif input_type == "file":
                # Require a file path; ignore xml content if provided
                if not xml_path:
                    errors["xml_file_path"] = "file_required"
                    errors.setdefault("base", "file_required")
                else:
                    # Resolve relative path against HA config directory
                    xml_path_obj = Path(xml_path)
                    if not xml_path_obj.is_absolute():
                        xml_path_obj = Path(self.hass.config.config_dir) / xml_path

                    if not xml_path_obj.is_file():
                        errors["xml_file_path"] = "file_not_found"
                        errors.setdefault("base", "file_not_found")
                    else:
                        try:
                            def _read(path: Path) -> str:
                                return path.read_text(encoding="utf-8")

                            xml_content = await self.hass.async_add_executor_job(
                                _read, xml_path_obj
                            )
                            user_input["xml"] = xml_content
                        except (OSError, IOError, UnicodeDecodeError):
                            errors["xml_file_path"] = "file_read_error"
                            errors.setdefault("base", "file_read_error")
            elif input_type == "xml":
                # Require inline XML content; ignore file path
                if not xml_content:
                    errors["xml"] = "xml_required"
                    errors.setdefault("base", "xml_required")
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
        # Store gas cost allocation toggle
        config_data["gas_cost_allocation"] = user_input.get(
            "gas_cost_allocation", "pro_rate_daily"
        )
        # Store gas usage allocation toggle
        config_data["gas_usage_allocation"] = user_input.get(
            "gas_usage_allocation", "daily_readings"
        )

        return self.async_create_entry(
            title=config.name,
            data=config_data,
        )


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle Green Button options flow."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        # Defaults from options first, then data, then fallback
        current_mode = (
            self.config_entry.options.get("gas_cost_allocation")
            or self.config_entry.data.get("gas_cost_allocation")
            or "pro_rate_daily"
        )
        current_usage_mode = (
            self.config_entry.options.get("gas_usage_allocation")
            or self.config_entry.data.get("gas_usage_allocation")
            or "daily_readings"
        )

        schema = vol.Schema(
            {
                vol.Optional(
                    "gas_cost_allocation",
                    default=current_mode,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "pro_rate_daily", "label": "Pro-rate gas costs daily"},
                            {"value": "monthly_increment", "label": "Single monthly cost increment"},
                        ],
                        mode="list",
                    )
                ),
                vol.Optional(
                    "gas_usage_allocation",
                    default=current_usage_mode,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "daily_readings", "label": "Use daily readings (m³)"},
                            {"value": "monthly_increment", "label": "Single billing-period usage increment (m³)"},
                        ],
                        mode="list",
                    )
                ),
            }
        )

        if user_input is not None:
            # Only store provided options
            return self.async_create_entry(title="", data={
                "gas_cost_allocation": user_input.get("gas_cost_allocation", current_mode),
                "gas_usage_allocation": user_input.get("gas_usage_allocation", current_usage_mode),
            })

        return self.async_show_form(step_id="init", data_schema=schema)
