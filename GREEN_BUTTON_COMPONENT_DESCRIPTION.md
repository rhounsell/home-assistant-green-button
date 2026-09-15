# Green Button Energy Dashboard Setup

## What the integration does

Green Button imports are historical source data, not a live utility feed. The
integration accepts Green Button ESPI Atom/XML documents, keeps an archive of
the accepted source data, and publishes its own long-term statistics for the
Home Assistant Energy Dashboard. It does not poll a provider or make outbound
API calls.

The entities shown by the integration are display entities: their values are
the totals represented by the imported history. The Energy Dashboard history
is stored in separate, integration-owned external statistics. It is not
created from the display entity's state history, so no recorder exclusions or
other recorder configuration are needed.

## Config entries

Each Green Button setup in Home Assistant is a separate **config entry**. It
owns its XML archive, entities, and imported statistics, so every import,
diagnostic, and maintenance action asks you to select the entry it should
affect. The entry name is chosen during setup; for example, an entry named
**Home** in your production instance appears as **Home** in the action's
**Config entry** selector. Select that entry whenever you want to work with
its imported data.

Multiple entries are useful for genuinely separate datasets, such as different
homes, utility accounts, or a test import that must not mix with production.
Each entry keeps its own options as well as its archive, entities, and
statistics. One entry is normally enough for a single home: multiple
electricity or gas streams within the same export are handled by that entry.

## Quick setup

1. Install the integration and create a Green Button config entry.
2. Import an ESPI XML document during setup, or later with **Import Green
   Button ESPI XML** under Developer Tools -> Actions. Select the config entry
   that should own the data, then provide either pasted XML or one XML file
   path.
3. If the XML file is outside Home Assistant's configuration directory, add
   its parent directory to `homeassistant.allowlist_external_dirs` in
   `configuration.yaml` and restart Home Assistant before importing it.
4. Wait for the background statistics generation to finish, then select the
   imported Green Button usage and cost statistics in the Energy Dashboard.

The imported statistic IDs begin with `green_button:`. An entity also exposes
its associated ID in the `statistic_id` attribute. Select the matching
integration-owned statistic in the Energy Dashboard rather than relying on
the display entity's state history.

## How imported data is handled

The XML archive is the authoritative source. On restart, the integration reads
the archived documents and rebuilds the same canonical history and statistics.
This makes a restart safe and allows additional exports to be imported later.

- Imports may arrive out of chronological order and may have gaps.
- Re-importing identical XML is ignored.
- A later reading with the same start time and duration replaces the earlier
  reading, allowing a provider correction.
- Ambiguous overlapping readings with different intervals are retained from
  the already accepted source and the conflicting reading is logged rather
  than guessed.
- Unsupported readings are skipped where possible. Missing cost data is not
  treated as zero cost.

Electricity history is allocated to complete UTC hours. Gas daily history uses
the Home Assistant local time zone; gas can instead be published as one
increment per billing period. Cost allocation and its power-of-ten multiplier
settings are separate from usage allocation. For the full allocation and
reconciliation rules, see [Green Button Data Handling](DATA_HANDLING_EXPLANATION.md)
and [Monthly Gas Increment Handling](GAS_MONTHLY_INCREMENT_HANDLING.md).

## Verify the import

After importing data:

1. In Developer Tools -> Statistics, find the `green_button:` statistics for
   the imported usage and cost series.
2. Confirm their time range and totals match the XML export and the utility
   bill. A display entity shows the imported total; it is not itself a live
   meter reading.
3. Add the matching imported usage and cost series to the Energy Dashboard.
4. If the data does not look right, use the diagnostic actions below and check
   the Home Assistant logs before deleting data.

## Actions

All actions are scoped to one selected Green Button config entry. Actions that
change source data or statistics require an administrator.

- **Log Green Button Meter Reading Intervals** logs accepted stream coverage
  and the entities mapped to it.
- **Log Stored Green Button XML Info** logs archived XML labels, sizes, and
  date coverage.
- **Import Green Button ESPI XML** imports one XML document from pasted
  content or a permitted file path.
- **Recalculate Green Button Cost Statistics** regenerates electricity and/or
  gas cost history after changing a fallback cost multiplier. A multiplier
  declared in the XML continues to take precedence.
- **Clear Stored Green Button XML Data** removes archived XML for all data or
  one commodity, then rebuilds active source history from the remaining
  archive. It does not delete Energy Dashboard statistics.
- **Delete Green Button Statistics** removes only the selected display
  entity's imported `green_button:` statistic. The XML archive is retained.

## Repair and recovery

Use the diagnostic actions first to confirm the archived source coverage and
the selected config entry. The two destructive actions intentionally do
different things:

- Use **Clear Stored Green Button XML Data** only when the archived source
  itself should no longer participate in future reconstruction. It leaves
  existing recorder and Energy Dashboard statistics in place.
- Use **Delete Green Button Statistics** only when the imported statistic for
  one display entity must be removed. Keep the source XML archive, then
  re-import the source data to rebuild the statistic.

Do not use either action merely to refresh an Energy Dashboard view. First
check that the dashboard is using the expected `green_button:` statistic and
that the archive covers the period you intend to retain.

## Troubleshooting

### The entity is unavailable or no statistic appears

- Confirm that the XML has supported readings or summaries; a document with no
  usable data is rejected.
- Confirm that the correct config entry was selected for the import.
- For file imports, confirm that the path is under the configuration directory
  or is listed in `allowlist_external_dirs`.
- Use **Log Stored Green Button XML Info** and **Log Green Button Meter Reading
  Intervals**, then review the Home Assistant logs.

### The Energy Dashboard has no data or the wrong series

- Verify that the dashboard selection is the `green_button:` imported
  statistic associated with the display entity.
- Check the statistic's date range against the archived XML coverage.
- Wait for background generation after an import, restart, or relevant entity
  setup to complete.

### Costs are incorrectly scaled

- Check whether the source XML declares `powerOfTenMultiplier`; it overrides
  the integration's fallback multiplier.
- If the XML omits it, update the applicable fallback cost multiplier in the
  integration options and run **Recalculate Green Button Cost Statistics**.
