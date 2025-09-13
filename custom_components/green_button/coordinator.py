import logging
from typing import Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from . import model
from .const import DOMAIN
from .parsers import espi

_LOGGER: Final = logging.getLogger(__name__)

class GreenButtonCoordinator(DataUpdateCoordinator):
    """Coordinator to manage Green Button data updates (event-driven, no polling)."""

    def __init__(self, hass: HomeAssistant, xml_path: str | None = None):
        super().__init__(hass, _LOGGER, name=DOMAIN) # No update_interval means no polling
        self.xml_path = xml_path
        self.interval_blocks: list[model.IntervalBlock] | None = None

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch and parse the latest Green Button data."""
        try:
            interval_blocks = None
            if self.xml_path:
                interval_blocks = await self.hass.async_add_executor_job(
                    espi.parse_interval_blocks_xml, self.xml_path
                )
            self.interval_blocks = interval_blocks
            return {"interval_blocks": interval_blocks}
        except Exception as err:
            raise UpdateFailed(f"Error updating Green Button data: {err}") from err

    # TODO need to understand how to extract the latest cumulative energy usage in kWh
    def get_latest_cumulative_energy_kwh(self) -> float | None:
        """Return the latest cumulative energy usage in kWh from usage_points."""
        if not self.interval_blocks:
            return None

        latest_value = None
        latest_time = None

        for interval_block in self.interval_blocks:
            # Find the latest interval_reading
                for interval_reading in interval_block.interval_readings:
                    # Check if this reading is the latest
                    if (
                        latest_time is None
                        or interval_reading.end > latest_time
                    ):
                        latest_time = interval_reading.end
                        # Assume value is in Wh, convert to kWh if needed
                        value = getattr(interval_reading, "value", None)
                        if value is not None:
                            latest_value = float(value) / 1000.0
    
        return latest_value
            