# Green Button Data Handling

## Purpose

Green Button imports are historical source data, not a live meter feed. The integration accepts Green Button ESPI Atom/XML documents, preserves the source documents, reconciles their readings into a canonical in-memory history, and publishes integration-owned long-term statistics for Home Assistant's Energy Dashboard. It does not poll a utility or make outbound API calls.

The display sensors show the total represented by the imported history. The historical series used by the Energy Dashboard is a separate, integration-owned statistic; it is not generated from the display sensor's state history.

## Import and source storage

XML can be supplied when the config entry is created, or later through the **Import Green Button ESPI XML** action. The action accepts exactly one of:

- pasted XML content; or
- an XML file path. Relative paths are resolved from Home Assistant's config directory. Paths outside that directory must be in `allowlist_external_dirs`.

The integration rejects a document that has neither supported interval readings nor usage summaries. It reports and skips unsupported individual readings where possible rather than treating them as zero usage.

Source XML is stored immediately in a private Home Assistant storage document:

```
.storage/green_button_xml_<config-entry-id>
```

Documents are grouped under an auto-detected `electricity`, `gas`, or fallback `imported_data` label. An SHA-256 content hash prevents the same XML document from being stored twice, even if it is submitted under a different label.

During initial setup, XML entered in the config flow is first held in temporary storage (because an entry ID does not yet exist) and is then migrated to the entry's permanent archive. Older config-entry storage formats are read as backwards-compatible fallbacks.

On startup, the integration reparses every archived document and applies the same reconciliation rules used for new imports. The archive is therefore the authoritative source from which active Green Button history is rebuilt after a restart.

## Reconciliation of multiple imports

Imports do not need to be chronological and may have gaps. Data is organized by provider Usage Point and Meter Reading identifiers.

- Meter readings with different IDs remain separate streams.
- Interval identity is its start time and duration. A later import replaces an existing interval with the same identity, allowing provider corrections.
- Intervals that overlap but have different identities are ambiguous. The already accepted coverage is retained and the conflicting interval is logged and skipped.
- Usage summaries are deduplicated by provider ID or identical period. A new summary that ambiguously overlaps a retained period is skipped.
- Canonical interval blocks are rebuilt from the accepted, non-overlapping readings.

This means that reimporting an identical document has no effect, a corrected copy of an identical interval can supersede the previous value, and unrelated or non-overlapping date ranges accumulate normally.

## Entities and display values

Entities are created dynamically for usable provider streams and retain stable identities based on the config entry, Usage Point, and Meter Reading IDs. This avoids collisions when two Usage Points use the same final Meter Reading path segment. Where an older suffix-based entity ID maps unambiguously to one new stream, its entity registration and external statistic are migrated.

For electricity, each eligible meter stream gets a usage sensor and a cost sensor. Gas usage is represented per eligible gas stream in daily-readings mode. A gas cost entity can be associated with a single gas stream; when several streams share a Usage Point, it is skipped because its billing summaries cannot be attributed safely. With **monthly increment** gas usage selected, a Usage Point with summaries is represented by one usage and one cost stream, including the summary-only case.

Display values are totals of the records that can be published to their corresponding statistic:

- Electricity usage is the total of complete hourly allocations, converted to kWh.
- Electricity cost is the total of complete hourly cost allocations. It is unavailable when any included interval lacks cost data; missing cost is not assumed to be zero.
- Gas usage is a total in m³ based on the selected gas-usage allocation mode.
- Gas cost is the sum of available billing-summary costs in the source currency (CAD until a source currency is available).

The entities deliberately have no sensor `state_class`. Their occasional state writes update the display only; they do not ask Home Assistant's automatic sensor-statistics pipeline to infer history from present-day state changes.

## Integration-owned statistics

Each display entity exposes a `statistic_id` attribute that points to its associated external statistic. The IDs have the form:

```
green_button:<SHA-256 hash of the entity unique ID>
```

They are stable when the user renames an entity and separate from the display entity ID. Metadata marks the series as sourced by `green_button`, with a `sum` and no mean, so it can be selected in the Energy Dashboard.

After an import, restart, or relevant entity setup, background tasks generate the statistics. Work for an individual statistic is serialized, and outstanding tasks are cancelled and drained when its config entry unloads.

For a changed series, the integration validates all generated records, then uses one recorder task and transaction to replace that external statistic's metadata and records. A fingerprint avoids writing an unchanged series. This is intentionally different from both recorder-generated sensor statistics and the former `async_import_statistics()` approach.

Every imported record contains an incremental `state` and a running cumulative `sum`. When a new import changes history, existing records before the earliest new timestamp are retained; records from that timestamp forward are rebuilt so their sums remain correct. This supports backfills and out-of-order imports.

## How each kind of history is allocated

### Electricity usage and cost

Intervals are split across UTC hour boundaries in proportion to the time spent in each hour. Only hours with a full hour of coverage are emitted. The final partial hour is held until later source data completes it, preventing a misleading small final bar in the Energy Dashboard.

Energy values are scaled from the source reading and converted from Wh, kWh, or MWh to kWh. Cost values use the source cost multiplier when one is declared; the configured electricity fallback multiplier is used only when the XML omits one. Electricity cost history is generated only with complete interval-cost coverage.

### Gas usage

Gas uses Home Assistant's configured time zone for billing dates.

- **Daily readings** allocates interval usage by physical overlap with each local calendar day and emits one record at local midnight for that day.
- **Monthly increment** emits one consumption increment on each billing period's local end date. It prefers `UsageSummary` consumption and can also use a non-overlapping long billing-period interval. This mode supports UsageSummary-only gas XML.

### Gas cost

Gas cost comes from Usage Summary billing totals. With **pro-rate daily**, a billing total is distributed across local days in proportion to the available daily gas usage. Missing day coverage causes the bill to be allocated over the available measured days; when no positive consumption is available, it is split evenly across the billing period. These are estimates, and the display entity identifies the mode as `estimated_daily_proration`.

With **monthly increment**, the full billing cost is recorded on the billing period's local end date. Summary-only data uses this mode because there are no daily readings with which to pro-rate a bill.

The configured gas fallback multiplier is used only if the source does not declare a cost multiplier. A source-declared multiplier always wins.

## Retention, clearing, and rebuilding

The available diagnostic and maintenance actions are scoped to one config entry:

- **Log Green Button Meter Reading Intervals** logs canonical stream coverage and mapped entities.
- **Log Stored Green Button XML Info** logs archived document labels, sizes, and coverage.
- **Clear Stored Green Button XML Data** removes all archived XML, or only the selected `electricity` or `gas` label, then rebuilds the active in-memory history from what remains. It does not delete recorder statistics.
- **Delete Green Button Statistics** deletes the selected display entity's `green_button:` external series only. The XML archive is retained, so reimporting the source data can rebuild it.
- **Recalculate Green Button Cost Statistics** regenerates electricity and/or gas cost history using the current fallback multiplier settings. Use it after changing those settings; XML-declared multipliers remain unchanged.

Clearing XML and deleting statistics are intentionally separate operations. Clearing the archive stops that source from being part of future reconstruction, while deleting a statistic removes the existing Energy Dashboard history for that one external series.
