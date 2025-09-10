from datetime import timedelta
import logging

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.core import HomeAssistant

from . import model, parsers

_LOGGER = logging.getLogger(__name__)

class GreenButtonCoordinator(DataUpdateCoordinator):
    """Coordinator to manage Green Button data updates."""

    def __init__(self, hass: HomeAssistant, xml_path: str | None = None):
        super().__init__(
            hass,
            _LOGGER,
            name="Green Button Data",
            update_interval=timedelta(hours=1),  # Adjust as needed
        )
        self.xml_path = xml_path
        self.data: model.MeterReading | None = None

    async def _async_update_data(self):
        """Fetch and parse the latest Green Button data."""
        try:
            if self.xml_path:
                # Parse the ESPI XML file
                self.data = await self.hass.async_add_executor_job(
                    parsers.espi.parse_xml, self.xml_path
                )
            return self.data
        except Exception as err:
            raise UpdateFailed(f"Error updating Green Button data: {err}") from err