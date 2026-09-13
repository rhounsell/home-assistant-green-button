"""Recorder persistence and task orchestration for Green Button statistics."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import hashlib
import logging
import math
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, final

from homeassistant.components.recorder import (
    db_schema as recorder_db_schema,
)
from homeassistant.components.recorder import (
    statistics,
    tasks,
)
from homeassistant.components.recorder.models import StatisticData
from homeassistant.components.recorder.models.statistics import StatisticMetaData
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import recorder as recorder_helper
from homeassistant.helpers.start import async_at_started

from .const import DOMAIN
from .statistic_ids import statistic_id_from_unique_id

if TYPE_CHECKING:
    from homeassistant.components.recorder.core import Recorder

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

_DATA_STATISTICS_LOCKS = f"{DOMAIN}_statistics_locks"
_DATA_STATISTICS_TASKS = f"{DOMAIN}_statistics_tasks"
_DATA_STATISTICS_FINGERPRINTS = f"{DOMAIN}_statistics_fingerprints"


def _queue_task(
    hass: HomeAssistant, task_ctor: Callable[[asyncio.Future[T]], tasks.RecorderTask]
) -> asyncio.Future[T]:
    """Queue a recorder task and return the future completed by that task."""
    future = asyncio.get_event_loop().create_future()
    recorder_helper.get_instance(hass).queue_task(task_ctor(future))
    #   RDH recorder_util.get_instance(hass).queue_task(task_ctor(future))
    return future


def _complete_future(future: asyncio.Future[T], value: T) -> None:
    """Set a result unless the waiter has already completed or cancelled."""

    def _set_result() -> None:
        if not future.done():
            future.set_result(value)

    future.get_loop().call_soon_threadsafe(_set_result)


def _complete_future_exception(future: asyncio.Future[T], err: BaseException) -> None:
    """Set an exception unless the waiter has already completed or cancelled."""

    def _set_exception() -> None:
        if not future.done():
            future.set_exception(err)

    future.get_loop().call_soon_threadsafe(_set_exception)


def _statistics_lock(hass: HomeAssistant, statistic_id: str) -> asyncio.Lock:
    """Return the lock serializing one statistics series."""
    locks = cast(
        dict[str, asyncio.Lock],
        hass.data.setdefault(_DATA_STATISTICS_LOCKS, {}),
    )
    return locks.setdefault(statistic_id, asyncio.Lock())


def async_schedule_statistics_update(
    hass: HomeAssistant,
    entry: ConfigEntry,
    update_factory: Callable[[], Coroutine[Any, Any, None]],
) -> asyncio.Task[None]:
    """Schedule a post-start statistics update that can be cancelled on unload."""

    async def _async_run_after_started() -> None:
        started: asyncio.Future[None] = hass.loop.create_future()

        @callback
        def _mark_started(_hass: HomeAssistant) -> None:
            if not started.done():
                started.set_result(None)

        cancel_started_listener = async_at_started(hass, _mark_started)
        try:
            await started
        finally:
            cancel_started_listener()

        await update_factory()

    entry_id = entry.entry_id
    task = entry.async_create_background_task(
        hass,
        _async_run_after_started(),
        f"Green Button statistics update {entry_id}",
    )
    tasks_by_entry = cast(
        dict[str, set[asyncio.Task[None]]],
        hass.data.setdefault(_DATA_STATISTICS_TASKS, {}),
    )
    entry_tasks = tasks_by_entry.setdefault(entry_id, set())
    entry_tasks.add(task)

    def _discard(completed: asyncio.Task[None]) -> None:
        entry_tasks.discard(completed)
        if not entry_tasks:
            tasks_by_entry.pop(entry_id, None)

    task.add_done_callback(_discard)
    return task


async def async_cancel_statistics_tasks(hass: HomeAssistant, entry_id: str) -> None:
    """Cancel and drain statistics updates belonging to an unloaded entry."""
    tasks_by_entry = cast(
        dict[str, set[asyncio.Task[None]]],
        hass.data.get(_DATA_STATISTICS_TASKS, {}),
    )
    entry_tasks = tasks_by_entry.pop(entry_id, set())
    for task in entry_tasks:
        task.cancel()
    if entry_tasks:
        await asyncio.gather(*entry_tasks, return_exceptions=True)


def _state_and_sum(record: StatisticData) -> tuple[float, float] | None:
    """Return state and sum values, or None when either field is absent."""
    state = record.get("state")
    total = record.get("sum")
    if state is None or total is None:
        return None
    return float(state), float(total)


def _validated_statistics(
    metadata: StatisticMetaData,
    records: list[StatisticData],
) -> list[StatisticData]:
    """Validate a complete replacement before scheduling any mutation."""
    statistic_id = metadata["statistic_id"]
    if metadata["source"] != DOMAIN or not statistic_id.startswith(f"{DOMAIN}:"):
        raise ValueError(f"Invalid external statistic metadata for {statistic_id}")
    if not records:
        raise ValueError(f"Refusing to replace {statistic_id} with no records")

    validated: list[StatisticData] = []
    previous_start: datetime.datetime | None = None
    for record in records:
        start = record["start"]
        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or start.minute
            or start.second
            or start.microsecond
        ):
            raise ValueError(f"Invalid statistic timestamp for {statistic_id}: {start}")
        if previous_start is not None and start <= previous_start:
            raise ValueError(f"Statistics for {statistic_id} are not strictly ordered")
        values = _state_and_sum(record)
        if values is None:
            raise ValueError(
                f"Statistics for {statistic_id} must contain state and sum values"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Statistics for {statistic_id} contain a non-finite value"
            )
        validated.append({**record, "start": start.astimezone(datetime.UTC)})
        previous_start = start
    return validated


@final
@dataclasses.dataclass(frozen=False)
class _ReplaceStatisticsTask(tasks.RecorderTask):
    """Replace a series in one recorder transaction."""

    metadata: StatisticMetaData
    records: list[StatisticData]
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        statistic_id = self.metadata["statistic_id"]
        try:
            with recorder_helper.session_scope(
                session=instance.get_session()
            ) as session:
                instance.statistics_meta_manager.delete(session, [statistic_id])
                # Home Assistant has no public session-aware import API; using this
                # helper keeps deletion and reimport in the same transaction.
                # pylint: disable-next=protected-access
                statistics._import_statistics_with_session(
                    instance,
                    session,
                    self.metadata,
                    self.records,
                    recorder_db_schema.Statistics,
                )
        # Recorder backends can raise implementation-specific errors; every failure
        # must be forwarded so the awaiting caller's future always completes.
        # pylint: disable-next=broad-exception-caught
        except Exception as err:  # noqa: BLE001
            _complete_future_exception(self.future, err)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        metadata: StatisticMetaData,
        records: list[StatisticData],
    ) -> asyncio.Future[None]:
        """Queue an atomic replacement and return its completion future."""

        def ctor(future: asyncio.Future[None]) -> _ReplaceStatisticsTask:
            return cls(metadata=metadata, records=records, future=future)

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _UpsertStatisticsTask(tasks.RecorderTask):
    """Import only new or changed records in one recorder transaction."""

    metadata: StatisticMetaData
    records: list[StatisticData]
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        try:
            with recorder_helper.session_scope(
                session=instance.get_session()
            ) as session:
                # Home Assistant has no public session-aware import API; this task
                # needs completion reporting for the transaction it owns.
                # pylint: disable-next=protected-access
                statistics._import_statistics_with_session(
                    instance,
                    session,
                    self.metadata,
                    self.records,
                    recorder_db_schema.Statistics,
                )
        # Recorder backends can raise implementation-specific errors; every failure
        # must be forwarded so the awaiting caller's future always completes.
        # pylint: disable-next=broad-exception-caught
        except Exception as err:  # noqa: BLE001
            _complete_future_exception(self.future, err)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(
        cls,
        hass: HomeAssistant,
        metadata: StatisticMetaData,
        records: list[StatisticData],
    ) -> asyncio.Future[None]:
        """Queue an atomic upsert and return its completion future."""

        def ctor(future: asyncio.Future[None]) -> _UpsertStatisticsTask:
            return cls(metadata=metadata, records=records, future=future)

        return _queue_task(hass, ctor)


def _statistics_fingerprint(records: Sequence[StatisticData]) -> str:
    """Return a stable fingerprint for normalized statistics records."""
    digest = hashlib.sha256()
    for record in records:
        start = record["start"].astimezone(datetime.UTC)
        values = _state_and_sum(record)
        if values is None:
            raise ValueError("Cannot fingerprint statistics without state and sum")
        state, total = values
        digest.update(f"{start.isoformat()}\0{state:.17g}\0{total:.17g}\n".encode())
    return digest.hexdigest()


def _changed_statistics(
    existing: Sequence[StatisticData],
    desired: Sequence[StatisticData],
) -> tuple[list[StatisticData], bool]:
    """Return new/changed records and whether timestamps need removal."""
    existing_by_start = {
        record["start"].astimezone(datetime.UTC): record for record in existing
    }
    desired_starts = {record["start"].astimezone(datetime.UTC) for record in desired}
    changed: list[StatisticData] = []
    for record in desired:
        desired_values = _state_and_sum(record)
        if desired_values is None:
            raise ValueError("Cannot compare statistics without state and sum")
        current = existing_by_start.get(record["start"].astimezone(datetime.UTC))
        if current is None or _state_and_sum(current) != desired_values:
            changed.append(record)
    return changed, bool(existing_by_start.keys() - desired_starts)


async def _async_replace_statistics(
    hass: HomeAssistant,
    metadata: StatisticMetaData,
    records: list[StatisticData],
    existing: list[StatisticData] | None = None,
) -> bool:
    """Synchronize an external series by writing only changed records."""
    validated = _validated_statistics(metadata, records)
    fingerprints = cast(
        dict[str, str],
        hass.data.setdefault(_DATA_STATISTICS_FINGERPRINTS, {}),
    )
    statistic_id = metadata["statistic_id"]
    fingerprint = _statistics_fingerprint(validated)
    if fingerprints.get(statistic_id) == fingerprint:
        _LOGGER.info(
            "No changes detected for Green Button statistic %s; recorder was not updated",
            statistic_id,
        )
        return False

    if existing is None:
        existing = await _get_all_existing_statistics(hass, statistic_id)
    changed, requires_replacement = _changed_statistics(existing, validated)
    if not changed and not requires_replacement:
        fingerprints[statistic_id] = fingerprint
        _LOGGER.info(
            "No changes detected for Green Button statistic %s; recorder was not updated",
            statistic_id,
        )
        return False

    if not requires_replacement:
        _LOGGER.info(
            "Importing %d new or changed records for %s",
            len(changed),
            statistic_id,
        )
        await _UpsertStatisticsTask.queue_task(hass, metadata, changed)
        fingerprints[statistic_id] = fingerprint
        return True

    _LOGGER.info(
        "Replacing %d records for %s from %s through %s",
        len(validated),
        statistic_id,
        validated[0]["start"],
        validated[-1]["start"],
    )
    await _ReplaceStatisticsTask.queue_task(hass, metadata, validated)
    fingerprints[statistic_id] = fingerprint
    return True


@final
@dataclasses.dataclass(frozen=False)
class _ClearStatisticsTask(tasks.RecorderTask):
    """Clear one external series and settle its waiting caller."""

    hass: HomeAssistant
    statistic_id: str
    future: asyncio.Future[None]

    def run(self, instance: Recorder) -> None:
        try:
            statistics.clear_statistics(
                instance=instance, statistic_ids=[self.statistic_id]
            )
        # Recorder backends can raise implementation-specific errors; every failure
        # must be forwarded so the awaiting caller's future always completes.
        # pylint: disable-next=broad-exception-caught
        except Exception as err:  # noqa: BLE001
            _complete_future_exception(self.future, err)
            return
        _complete_future(self.future, None)

    @classmethod
    def queue_task(cls, hass: HomeAssistant, statistic_id: str) -> asyncio.Future[None]:
        """Queue the clear operation and return its completion future."""

        def ctor(future: asyncio.Future[None]) -> _ClearStatisticsTask:
            return cls(hass=hass, statistic_id=statistic_id, future=future)

        return _queue_task(hass, ctor)


@final
@dataclasses.dataclass(frozen=False)
class _RenameExternalStatisticTask(tasks.RecorderTask):
    """Rename an external Green Button series with its entity identity."""

    old_statistic_id: str
    new_statistic_id: str

    def run(self, instance: Recorder) -> None:
        """Rename only metadata owned by this integration."""
        with recorder_helper.session_scope(session=instance.get_session()) as session:
            instance.statistics_meta_manager.update_statistic_id(
                session, DOMAIN, self.old_statistic_id, self.new_statistic_id
            )


def rename_external_statistic(
    hass: HomeAssistant, old_unique_id: str, new_unique_id: str
) -> None:
    """Queue migration of an external series after an entity-ID migration."""
    recorder_helper.get_instance(hass).queue_task(
        _RenameExternalStatisticTask(
            statistic_id_from_unique_id(old_unique_id),
            statistic_id_from_unique_id(new_unique_id),
        )
    )


async def _get_all_existing_statistics(
    hass: HomeAssistant,
    statistic_id: str,
) -> list[StatisticData]:
    """Retrieve all existing hourly statistics for a statistic_id.

    Returns a list of StatisticData dictionaries sorted by start time.
    """
    rec = recorder_helper.get_instance(hass)

    def _get_stats() -> dict[str, list[Any]]:
        return statistics.statistics_during_period(
            hass=hass,
            start_time=datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
            end_time=datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc),
            statistic_ids={statistic_id},
            period="hour",
            types={"sum", "state"},
            units=None,
        )

    raw_stats = await rec.async_add_executor_job(_get_stats)
    stats_list = (raw_stats or {}).get(statistic_id, [])
    result: list[StatisticData] = []
    for stat in stats_list:
        stat_dict = cast(dict[str, Any], stat)
        start_val = stat_dict["start"]
        start_dt = (
            start_val
            if isinstance(start_val, datetime.datetime)
            else datetime.datetime.fromtimestamp(start_val, tz=datetime.timezone.utc)
        )
        result.append(
            {
                "start": start_dt,
                "state": float(stat_dict.get("state", 0.0)),
                "sum": float(stat_dict.get("sum", 0.0)),
            }
        )

    result.sort(key=lambda statistic: statistic["start"])
    return result


async def clear_statistic(hass: HomeAssistant, statistic_id: str) -> None:
    """Clear all statistics with the specified ID."""
    await _ClearStatisticsTask.queue_task(hass=hass, statistic_id=statistic_id)
    fingerprints = cast(
        dict[str, str], hass.data.get(_DATA_STATISTICS_FINGERPRINTS, {})
    )
    fingerprints.pop(statistic_id, None)
