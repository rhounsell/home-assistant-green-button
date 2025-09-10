import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from . import model
from .parsers import espi

_LOGGER = logging.getLogger(__name__)

class GreenButtonCoordinator(DataUpdateCoordinator):
    """Coordinator to manage Green Button data updates (event-driven, no polling)."""

    def __init__(self, hass: HomeAssistant, xml_path: str | None = None):
        super().__init__(
            hass,
            _LOGGER,
            name="Green Button Data",
            # No update_interval means no polling
        )
        self.xml_path = xml_path
        self.usage_points: list[model.UsagePoint] | None = None

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch and parse the latest Green Button data."""
        try:
            usage_points = None
            if self.xml_path:
                usage_points = await self.hass.async_add_executor_job(
                    espi.parse_xml, self.xml_path
                )
            self.usage_points = usage_points
            return {"usage_points": usage_points}
        except Exception as err:
            raise UpdateFailed(f"Error updating Green Button data: {err}") from err
        