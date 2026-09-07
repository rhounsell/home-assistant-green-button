"""Green Button data coordinator."""

from __future__ import annotations

import dataclasses
import datetime
import logging
from bisect import bisect_left
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from .xml_storage import async_get_xml_storage

from . import model
from .const import DOMAIN
from .parsers import espi

_LOGGER = logging.getLogger(__name__)


class GreenButtonCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage Green Button data updates (manual updates only, no polling)."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the Green Button coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            config_entry=config_entry,
            # No update_interval - manual updates only
        )
        self.config_entry = config_entry
        self.usage_points: list[model.UsagePoint] = []

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch and parse the latest Green Button data."""
        try:
            usage_points = None
            # Get XML data from config entry instead of file path
            xml_data = self.config_entry.data.get("xml")
            if xml_data:
                usage_points = await self.hass.async_add_executor_job(
                    espi.parse_xml, xml_data
                )
        except Exception as err:
            raise UpdateFailed(f"Error updating Green Button data: {err}") from err
        else:
            self.usage_points = usage_points or []
            return {"usage_points": usage_points or []}

    async def async_add_xml_data(self, xml_data: str, store_in_config: bool = True) -> None:
        """Add new Green Button XML data and update entities.
        
        Args:
            xml_data: The XML data to parse and add
            store_in_config: If True, store the XML in a separate storage file.
                           If False, just merge the data without persisting (for service imports).
        
        The label is auto-detected from the XML content based on commodity type:
        - Electricity (ServiceCategory kind=0) -> 'electricity'
        - Gas (ServiceCategory kind=1) -> 'gas'
        - Unknown -> 'imported_data'
        """
        try:
            # Parse XML first to detect commodity type for auto-labeling
            usage_points = await self.hass.async_add_executor_job(
                espi.parse_xml, xml_data
            )
            new_usage_points = usage_points or []

            # Auto-detect label from commodity type
            label = self._detect_label_from_usage_points(new_usage_points)
            _LOGGER.info("Auto-detected label '%s' from XML content", label)

            # Store XML in separate storage file if requested (for persistence across restarts)
            # NOTE: We use a separate Store instance instead of config entry data because
            # config entries use delayed writes and are not designed for multi-MB data storage.
            if store_in_config:
                _LOGGER.info("Storing XML data to dedicated storage file for entry %s with label '%s'", 
                            self.config_entry.entry_id, label)
                # Use dedicated XML storage (immediate save for reliability)
                xml_storage = await async_get_xml_storage(self.hass, self.config_entry.entry_id)
                await xml_storage.async_add_xml(xml_data, label)
                _LOGGER.info("Successfully stored XML data to .storage/green_button_xml_%s", 
                            self.config_entry.entry_id)

            # Log what we're processing (usage_points already parsed above for label detection)
            total_readings = sum(len(up.meter_readings) for up in new_usage_points)
            _LOGGER.info(
                "Processing %d usage points with %d total meter readings",
                len(new_usage_points),
                total_readings,
            )
            # Log interval block date ranges for each new usage point and meter reading
            if _LOGGER.isEnabledFor(logging.DEBUG):
                for up in new_usage_points:
                    for mr in up.meter_readings:
                        for ib in mr.interval_blocks:
                            if ib.interval_readings:
                                start = ib.interval_readings[0].start
                                end = ib.interval_readings[-1].end
                                _LOGGER.debug(
                                    "[IMPORT] UsagePoint %s MeterReading %s IntervalBlock: %s - %s (%d readings)",
                                    up.id,
                                    mr.id,
                                    start,
                                    end,
                                    len(ib.interval_readings),
                                )
                            else:
                                _LOGGER.debug(
                                    "[IMPORT] UsagePoint %s MeterReading %s IntervalBlock: No readings",
                                    up.id,
                                    mr.id,
                                )

            # Merge new data with existing data (combine multiple imports)
            self._merge_usage_points(new_usage_points)

            # Debug: Log detailed data structure
            for i, up in enumerate(self.usage_points):
                _LOGGER.info(
                    "UsagePoint %d: %d meter readings", i, len(up.meter_readings)
                )
                for j, mr in enumerate(up.meter_readings):
                    _LOGGER.info(
                        "  MeterReading %d: %d intervals", j, len(mr.interval_blocks)
                    )
                    for ib in mr.interval_blocks:
                        if ib.interval_readings:
                            start = ib.interval_readings[0].start
                            end = ib.interval_readings[-1].end
                            _LOGGER.debug(
                                "  IntervalBlock: %s - %s (%d readings)",
                                start,
                                end,
                                len(ib.interval_readings),
                            )
                        else:
                            _LOGGER.debug(
                                "  IntervalBlock: No readings",
                            )

            # Update the data and notify all entities
            self.async_set_updated_data({"usage_points": self.usage_points})

            _LOGGER.info("Successfully updated coordinator with new data")

            # Trigger statistics generation for all meter readings after import
            await self._trigger_statistics_update_for_all_readings()

        except Exception as err:
            _LOGGER.error("Error adding Green Button XML data: %s", err)
            raise UpdateFailed(f"Error adding Green Button XML data: {err}") from err

    def _detect_label_from_usage_points(self, usage_points: list[model.UsagePoint]) -> str:
        """Detect label from usage points based on commodity type.
        
        Returns:
            'electricity' if any usage point is ENERGY type
            'gas' if any usage point is GAS type
            'imported_data' if no usage points or unknown type
        """

        for up in usage_points:
            if up.sensor_device_class == SensorDeviceClass.ENERGY:
                return "electricity"
            elif up.sensor_device_class == SensorDeviceClass.GAS:
                return "gas"

        return "imported_data"

    async def _trigger_statistics_update_for_all_readings(self) -> None:
        """Trigger statistics update for all meter readings in coordinator data.
        
        This ensures that after import, statistics are generated for every meter reading,
        including newly merged ones from imports. The coordinator update listeners
        (entity sensors) will be notified and will generate statistics automatically.
        """
        _LOGGER.info("Starting statistics update for all meter readings")

        if not self.data or not self.data.get("usage_points"):
            _LOGGER.info("No coordinator data available for statistics update")
            return

        if _LOGGER.isEnabledFor(logging.DEBUG):
            # Log all meter readings that need statistics generated
            total_meter_readings = 0
            for usage_point in self.usage_points:
                for meter_reading in usage_point.meter_readings:
                    total_meter_readings += 1
                    interval_count = sum(len(blk.interval_readings) for blk in meter_reading.interval_blocks)
                    if interval_count > 0:
                        _LOGGER.debug(
                            "Will generate statistics for meter reading %s: %d total readings across %d interval blocks",
                            meter_reading.id.split("/")[-1] if "/" in meter_reading.id else meter_reading.id,
                            interval_count,
                            len(meter_reading.interval_blocks),
                        )
                        for ib in meter_reading.interval_blocks:
                            if ib.interval_readings:
                                first = ib.interval_readings[0].start
                                last = ib.interval_readings[-1].end
                                _LOGGER.debug(
                                    "  IntervalBlock: %s - %s (%d readings)",
                                    first.isoformat(),
                                    last.isoformat(),
                                    len(ib.interval_readings),
                                )
                            else:
                                _LOGGER.debug("  IntervalBlock: No readings")

            _LOGGER.info("Statistics update scheduled for %d meter readings", total_meter_readings)

    def has_existing_entities(self) -> bool:
        """Check if entities already exist for the current data."""
        return bool(self.usage_points)

    async def async_load_stored_data(self) -> None:
        """Load XML data from storage file (used during startup).
        
        Uses a separate Store instance instead of config entry data because
        config entries use delayed writes and are not designed for multi-MB data.
        Falls back to config entry data for backwards compatibility.
        """

        # NEW: Check for and migrate temporary storage from config flow
        # (Config flow writes to temp storage to avoid putting large XML in config entry data)
        from .xml_storage import async_migrate_temp_storage
        
        unique_id = self.config_entry.unique_id
        if unique_id:
            try:
                migrated = await async_migrate_temp_storage(
                    self.hass, unique_id, self.config_entry.entry_id
                )
                if migrated:
                    _LOGGER.info("[CONFIG FLOW IMPORT] Successfully migrated XML from temporary storage to permanent storage")
                    # After migration, the data is already in permanent storage,
                    # so we continue below to load and process it
            except Exception as e:
                _LOGGER.warning("[CONFIG FLOW IMPORT] Failed to migrate temporary storage: %s", e)
        
        # LEGACY FALLBACK: Check for initial_xml from config flow (old method, for backwards compatibility)
        # New installations write directly to dedicated temp storage during config flow
        initial_xml = self.config_entry.data.get("initial_xml")
        if initial_xml:
            _LOGGER.info("[CONFIG FLOW IMPORT - LEGACY] Processing initial XML from config flow setup for entry %s",
                        self.config_entry.entry_id)
            _LOGGER.info("[CONFIG FLOW IMPORT - LEGACY] XML size: %d bytes", len(initial_xml))
            _LOGGER.warning("[CONFIG FLOW IMPORT - LEGACY] Using legacy migration path - XML should have been written to storage during config flow")
            # Process through normal flow which auto-detects label and stores properly
            await self.async_add_xml_data(initial_xml, store_in_config=True)

            # Remove initial_xml from config entry data (it's now in proper storage)
            data_updates = dict(self.config_entry.data)
            data_updates.pop("initial_xml", None)
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=data_updates
            ) 
            # Verify removal
            if "initial_xml" not in self.config_entry.data:
                _LOGGER.info("[CONFIG FLOW IMPORT - LEGACY] Successfully migrated initial_xml to .storage/green_button_xml_%s and removed from config entry",
                            self.config_entry.entry_id)
            else:
                _LOGGER.warning("[CONFIG FLOW IMPORT - LEGACY] Failed to remove initial_xml from config entry data!")
            return  # Data already processed

        if self.has_existing_entities():
            _LOGGER.debug("Entities already exist, skipping XML re-parsing on restart")
            return

        try:
            self.usage_points = await self.async_reconstruct_stored_usage_points()
            if not self.usage_points:
                _LOGGER.debug("No stored XML data found in storage or config entry")
                return

            self.async_set_updated_data({"usage_points": self.usage_points})
            self.last_update_success = True
            _LOGGER.info(
                "[RESTART] Successfully loaded %d canonical usage point(s). last_update_success set to True.",
                len(self.usage_points),
            )
        except (ValueError, OSError) as err:
            self.last_update_success = False
            _LOGGER.warning("[RESTART] Failed to load stored XML data: %s. last_update_success set to False.", err)

    async def async_reconstruct_stored_usage_points(self) -> list[model.UsagePoint]:
        """Parse all archived XML and apply the normal import reconciliation."""
        xml_storage = await async_get_xml_storage(self.hass, self.config_entry.entry_id)
        stored_xmls = xml_storage.get_stored_xmls()
        if not stored_xmls:
            stored_xmls = self.config_entry.data.get("stored_xmls", [])
        if not stored_xmls and (xml_data := self.config_entry.data.get("xml")):
            stored_xmls = [{"label": "imported_data", "xml": xml_data}]

        reconstructed = GreenButtonCoordinator(self.hass, self.config_entry)
        for xml_entry in stored_xmls:
            xml_list = xml_entry.get("xmls", [])
            if not xml_list and "xml" in xml_entry:
                xml_list = [xml_entry["xml"]]
            for xml_data in xml_list:
                if not xml_data:
                    continue
                usage_points = await self.hass.async_add_executor_job(
                    espi.parse_xml, xml_data
                )
                if usage_points:
                    reconstructed._merge_usage_points(usage_points)

        return reconstructed.usage_points

    def _merge_usage_points(self, new_usage_points: list[model.UsagePoint]) -> None:
        """Merge new usage points with existing ones, combining interval blocks."""
        if not self.usage_points:
            # No existing data, just use new data
            self.usage_points = [
                self._normalize_usage_point(usage_point)
                for usage_point in new_usage_points
            ]
            _LOGGER.info("[MERGE] No existing usage points, using new data only.")
            for up in new_usage_points:
                for mr in up.meter_readings:
                    for ib in mr.interval_blocks:
                        if ib.interval_readings:
                            start = ib.interval_readings[0].start
                            end = ib.interval_readings[-1].end
                            _LOGGER.debug(
                                "[MERGE] UsagePoint %s MeterReading %s IntervalBlock: %s - %s (%d readings)",
                                up.id,
                                mr.id,
                                start,
                                end,
                                len(ib.interval_readings),
                            )
                        else:
                            _LOGGER.debug(
                                "[MERGE] UsagePoint %s MeterReading %s IntervalBlock: No readings",
                                up.id,
                                mr.id,
                            )
            return

        # Create a mapping of existing usage points by ID
        existing_up_map = {up.id: up for up in self.usage_points}

        for new_up in new_usage_points:
            if new_up.id in existing_up_map:
                # Merge meter readings for existing usage point
                existing_up = existing_up_map[new_up.id]
                merged_meter_readings = self._merge_meter_readings(
                    existing_up, list(new_up.meter_readings)
                )
                # Merge usage summaries (unique by id)
                merged_summaries = self._merge_usage_summaries(
                    existing_up.usage_summaries, new_up.usage_summaries
                )
                merged_up = dataclasses.replace(
                    existing_up,
                    meter_readings=merged_meter_readings,
                    usage_summaries=merged_summaries,
                )
                self.usage_points = [
                    merged_up if up.id == existing_up.id else up
                    for up in self.usage_points
                ]
                _LOGGER.info(
                    "[MERGE] Merged usage point %s: %d meter readings, %d usage summaries",
                    new_up.id,
                    len(merged_meter_readings),
                    len(merged_summaries),
                )
            else:
                # Add new usage point
                self.usage_points.append(self._normalize_usage_point(new_up))
                _LOGGER.info(
                    "[MERGE] Added new usage point %s: %d meter readings, %d usage summaries",
                    new_up.id,
                    len(new_up.meter_readings),
                    len(new_up.usage_summaries),
                )

    def _merge_meter_readings(
        self,
        existing_up: model.UsagePoint,
        new_meter_readings: list[model.MeterReading],
    ) -> list[model.MeterReading]:
        """Merge new meter readings by their individual intervals."""
        existing_mr_map = {mr.id: mr for mr in existing_up.meter_readings}
        new_mr_map = {mr.id: mr for mr in new_meter_readings}
        merged_meter_readings: list[model.MeterReading] = []

        for existing_mr in existing_up.meter_readings:
            matching_new_mr = new_mr_map.get(existing_mr.id)
            if matching_new_mr is None:
                merged_meter_readings.append(
                    self._normalize_meter_reading(existing_mr)
                )
            else:
                merged_mr, replaced, rejected = self._reconcile_meter_reading(
                    existing_mr, matching_new_mr
                )
                merged_meter_readings.append(merged_mr)
                if rejected:
                    _LOGGER.warning(
                        "[MERGE] Reconciled meter reading %s: %d interval replacements, %d ambiguous overlaps rejected",
                        existing_mr.id,
                        replaced,
                        rejected,
                    )
                else:
                    _LOGGER.info(
                        "[MERGE] Reconciled meter reading %s: %d interval replacements",
                        existing_mr.id,
                        replaced,
                    )

        for new_mr in new_meter_readings:
            if new_mr.id not in existing_mr_map:
                merged_meter_readings.append(self._normalize_meter_reading(new_mr))
                _LOGGER.info(
                    "[MERGE] Added meter reading: %s to usage point %s",
                    new_mr.id,
                    existing_up.id,
                )

        return merged_meter_readings

    def _normalize_usage_point(
        self, usage_point: model.UsagePoint
    ) -> model.UsagePoint:
        """Normalize interval data in a newly discovered usage point."""
        return dataclasses.replace(
            usage_point,
            meter_readings=[
                self._normalize_meter_reading(meter_reading)
                for meter_reading in usage_point.meter_readings
            ],
            usage_summaries=self._merge_usage_summaries(
                [], usage_point.usage_summaries
            ),
        )

    @staticmethod
    def _merge_usage_summaries(
        existing_summaries: list[model.UsageSummary],
        new_summaries: list[model.UsageSummary],
    ) -> list[model.UsageSummary]:
        """Deduplicate summaries and retain existing coverage on ambiguity."""
        latest_by_id = {summary.id: summary for summary in new_summaries}
        latest_by_period = {
            (summary.start, summary.duration): summary
            for summary in latest_by_id.values()
        }
        retained = [
            summary
            for summary in existing_summaries
            if summary.id not in latest_by_id
            and (summary.start, summary.duration) not in latest_by_period
        ]
        accepted = sorted(retained, key=lambda summary: summary.start)
        for summary in sorted(latest_by_period.values(), key=lambda summary: summary.start):
            if any(
                summary.start < existing.start + existing.duration
                and existing.start < summary.start + summary.duration
                for existing in accepted
            ):
                _LOGGER.warning(
                    "[MERGE] Rejecting ambiguous overlapping usage summary %s",
                    summary.id,
                )
                continue
            accepted.append(summary)
        return sorted(accepted, key=lambda summary: summary.start)

    def _normalize_meter_reading(
        self, meter_reading: model.MeterReading
    ) -> model.MeterReading:
        """Normalize duplicate or overlapping intervals in one source."""
        normalized, _, rejected = self._reconcile_meter_reading(None, meter_reading)
        if rejected:
            _LOGGER.warning(
                "[MERGE] Meter reading %s contains %d ambiguous overlapping intervals; retaining the first intervals",
                meter_reading.id,
                rejected,
            )
        return normalized

    def _reconcile_meter_reading(
        self,
        existing_mr: model.MeterReading | None,
        new_mr: model.MeterReading,
    ) -> tuple[model.MeterReading, int, int]:
        """Apply latest-import precedence to identical intervals."""
        existing_records = (
            self._interval_records(existing_mr) if existing_mr is not None else []
        )
        new_records = self._interval_records(new_mr)
        existing_by_key = {
            self._interval_key(reading): (reading, block_id)
            for reading, block_id in existing_records
        }
        new_by_key = {
            self._interval_key(reading): (reading, block_id)
            for reading, block_id in new_records
        }
        replacement_keys = existing_by_key.keys() & new_by_key.keys()
        retained_existing = [
            record
            for key, record in existing_by_key.items()
            if key not in replacement_keys
        ]
        accepted: list[tuple[model.IntervalReading, str]] = []
        starts = []
        rejected = 0

        for record in sorted(retained_existing, key=lambda item: item[0].start):
            if self._insert_non_overlapping(record, accepted, starts):
                continue
            rejected += 1

        for record in sorted(new_by_key.values(), key=lambda item: item[0].start):
            if self._insert_non_overlapping(record, accepted, starts):
                continue
            rejected += 1

        return (
            dataclasses.replace(
                new_mr,
                interval_blocks=self._records_to_interval_blocks(accepted),
            ),
            len(replacement_keys),
            rejected,
        )

    @staticmethod
    def _interval_records(
        meter_reading: model.MeterReading,
    ) -> list[tuple[model.IntervalReading, str]]:
        """Return interval readings with their source block IDs."""
        return [
            (reading, block.id)
            for block in meter_reading.interval_blocks
            for reading in block.interval_readings
        ]

    @staticmethod
    def _interval_key(
        reading: model.IntervalReading,
    ) -> tuple[datetime.datetime, datetime.timedelta]:
        """Return the identity used for an interval correction."""
        return reading.start, reading.duration

    @staticmethod
    def _insert_non_overlapping(
        record: tuple[model.IntervalReading, str],
        accepted: list[tuple[model.IntervalReading, str]],
        starts: list[datetime.datetime],
    ) -> bool:
        """Insert a reading unless it overlaps a different accepted interval."""
        reading, _ = record
        index = bisect_left(starts, reading.start)
        for candidate_index in (index - 1, index):
            if candidate_index < 0 or candidate_index >= len(accepted):
                continue
            candidate, _ = accepted[candidate_index]
            if reading.start < candidate.end and candidate.start < reading.end:
                return False
        accepted.insert(index, record)
        starts.insert(index, reading.start)
        return True

    @staticmethod
    def _records_to_interval_blocks(
        records: list[tuple[model.IntervalReading, str]],
    ) -> list[model.IntervalBlock]:
        """Build non-overlapping blocks from canonical interval readings."""
        if not records:
            return []

        blocks: list[model.IntervalBlock] = []
        block_records: list[model.IntervalReading] = []
        block_id = records[0][1]

        def append_block() -> None:
            first = block_records[0]
            last = block_records[-1]
            blocks.append(
                model.IntervalBlock(
                    block_id,
                    first.reading_type,
                    first.start,
                    last.end - first.start,
                    block_records.copy(),
                )
            )

        for reading, source_block_id in records:
            if (
                block_records
                and (
                    source_block_id != block_id
                    or reading.reading_type != block_records[0].reading_type
                    or reading.start != block_records[-1].end
                )
            ):
                append_block()
                block_records = []
                block_id = source_block_id
            block_records.append(reading)

        append_block()
        return blocks

    def get_meter_readings(self) -> list[model.MeterReading]:
        """Get all meter readings from usage points."""
        meter_readings = []
        for usage_point in self.usage_points:
            meter_readings.extend(usage_point.meter_readings)
        return meter_readings

    def get_usage_summaries_for_meter_reading(self, meter_reading_id: str) -> list[model.UsageSummary]:
        """Get usage summaries for the usage point that owns the meter reading."""
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                if meter_reading.id == meter_reading_id:
                    return list(getattr(usage_point, "usage_summaries", []) or [])
        return []

    def get_meter_reading_by_id(
        self, meter_reading_id: str
    ) -> model.MeterReading | None:
        """Get a specific meter reading by ID."""
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                if meter_reading.id == meter_reading_id:
                    return meter_reading
        return None

    def get_latest_cumulative_energy_kwh(self) -> float | None:
        """Return the latest cumulative energy usage in kWh from usage_points."""
        if not self.usage_points:
            return None

        latest_value = None
        latest_time = None

        # Navigate through the hierarchy: UsagePoint -> MeterReading -> IntervalBlock -> IntervalReading
        for usage_point in self.usage_points:
            for meter_reading in usage_point.meter_readings:
                for interval_block in meter_reading.interval_blocks:
                    for interval_reading in interval_block.interval_readings:
                        # Check if this reading is the latest
                        if latest_time is None or interval_reading.end > latest_time:
                            latest_time = interval_reading.end
                            # Convert value based on power of ten multiplier and unit
                            power_multiplier = (
                                interval_reading.reading_type.power_of_ten_multiplier
                            )
                            value = interval_reading.value * (10**power_multiplier)
                            # Convert to kWh if needed (assuming base unit is Wh)
                            latest_value = float(value) / 1000.0

        return latest_value
