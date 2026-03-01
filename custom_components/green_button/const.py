"""Constants for the Green Button integration."""
from typing import Final

DOMAIN: Final = "green_button"
# INTEGRATION_SERVICE_DATA_KEY = "integration_service"

# Default power of ten multiplier for electricity cost values
# Cost values in Green Button XML are typically in hundredths of cents,
# so -5 converts them to dollars (e.g., 12345 * 10^-5 = $0.12345)
DEFAULT_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER: Final = -5

# Default power of ten multiplier for gas cost values
DEFAULT_GAS_COST_POWER_OF_TEN_MULTIPLIER: Final = -5

# Config/options keys for the user-specified cost power of ten multipliers
CONF_ELECTRICITY_COST_POWER_OF_TEN_MULTIPLIER: Final = "electricity_cost_power_of_ten_multiplier"
CONF_GAS_COST_POWER_OF_TEN_MULTIPLIER: Final = "gas_cost_power_of_ten_multiplier"
