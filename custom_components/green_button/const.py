"""Constants for the Green Button integration."""

from typing import Final

DOMAIN: Final = "green_button"
# INTEGRATION_SERVICE_DATA_KEY = "integration_service"


# class _CustomEntitiesParams:
#     NAME: Final = "name"
#     USAGE_POINT_HREF: Final = "usage_point_href"
#     ENERGY_UNIT: Final = "energy_unit"
#     COST_CURRENCY: Final = "cost_currency"


# @final
# class ConfigParams(_CustomEntitiesParams):
#     @classmethod
#     def make_schema(cls, hass: HomeAssistant) -> vol.Schema:
#         currency = hass.config.currency
#         return vol.Schema(
#             {
#                 vol.Required(cls.NAME): str,
#                 vol.Required(cls.USAGE_POINT_HREF): str,
#                 vol.Required(
#                     cls.ENERGY_UNIT, default=UnitOfEnergy.KILO_WATT_HOUR
#                 ): vol.In([unit.value for unit in UnitOfEnergy]),
#                 vol.Required(cls.COST_CURRENCY, default=currency): cv.currency,
#             }
#         )


# @final
# class OptionsParams(_CustomEntitiesParams):
#     @classmethod
#     def make_schema(cls) -> vol.Schema:
#         return vol.Schema(
#             {
#                 vol.Required(
#                     cls.USAGE_POINT_HREF,
#                 ): str,
#                 vol.Required(cls.ENERGY_UNIT): vol.In(
#                     [unit.value for unit in UnitOfEnergy]
#                 ),
#                 vol.Required(cls.COST_CURRENCY): cv.currency,
#             }
#         )
