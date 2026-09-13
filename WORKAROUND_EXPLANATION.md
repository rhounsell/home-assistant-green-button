# Monthly Gas Increment Handling

## Purpose

Some Enbridge Green Button exports contain finalized `UsageSummary` statements
for an earlier billing period and a longer `IntervalReading` for the more recent
period. The integration supports this pattern when **Gas usage allocation** is
set to `monthly_increment`, without inventing daily consumption.

For example, an export can contain:

- a finalized summary for July 26 through August 24; and
- a multi-day meter reading for August 25 through September 26.

Both can be represented as billing-period increments when they describe
separate coverage.

## Current behavior

The integration builds one normalized monthly series from archived XML:

1. `UsageSummary` consumption is authoritative for its billing period.
2. A multi-day `IntervalReading` is also included only when it does not overlap
   any `UsageSummary` period.
3. A multi-day reading means either its declared interval length or its actual
   duration is longer than one day. There is no former seven-day minimum.
4. Each increment is written at local midnight on the billing period's local
   end date. Values sharing an end date are combined before recorder import.
5. Daily readings are not presented as complete monthly billing periods.

This prevents double counting: if a later XML document adds a finalized
`UsageSummary` that overlaps a previously used multi-day reading, the summary
remains authoritative and the overlapping reading is not added again.

## When the next export arrives

Suppose the first export contains:

- `UsageSummary`: July 26–August 24, 52 m³; and
- non-overlapping `IntervalReading`: August 25–September 26, 54 m³.

The monthly usage series contains one increment for each period. When a later
export supplies the finalized August 25–September 26 summary, canonical source
reconciliation retains one representation of that period rather than adding a
second bar. The next non-overlapping multi-day reading can represent the newer
period in the same way.

## Statistics and source safety

Imported usage and cost history is written to integration-owned
`green_button:` external statistics. Display sensors do not produce automatic
Home Assistant sensor statistics, so imported history is not counted again as
present-day consumption.

Updates are source-driven and non-destructive where possible: unchanged series
produce no recorder write, and changed timestamps are upserted. A complete
transactional replacement is used only when a source update removes timestamps
that already exist in the external series. The integration does **not** clear
all statistics on every import.

Original XML is stored by commodity and replayed through the same canonical
merge path on startup and cost recalculation. Re-importing identical XML is
ignored.

## Cost allocation is separate

Gas usage and cost allocation are independent options:

- `monthly_increment` cost writes one billing-period cost increment.
- `pro_rate_daily` estimates daily cost from available detailed consumption and
  the applicable `UsageSummary` total. This requires a single attributable
  detailed meter stream; summary-only data uses monthly cost increments.

After changing a fallback cost multiplier, use **Recalculate Green Button Cost
Statistics** for the selected config entry and commodity. XML-declared
`powerOfTenMultiplier` values continue to take precedence over configured
fallbacks.

## Verification and recovery

1. Use **Log Stored Green Button XML Info** to confirm the archived source
   documents and their actual coverage.
2. Verify the Energy Dashboard is using the matching imported usage and cost
   external series for the selected config entry.
3. If an administrator intentionally deletes a display sensor's imported
   statistics, keep the archived XML, then reload the integration to regenerate
   the series from that canonical archive.

**Clear Stored Green Button XML Data** is not a statistic-repair action: it
removes the selected source archive and active in-memory history, but it leaves
existing recorder statistics intact. Delete statistics only as a deliberate,
separate administrator action after confirming the archive covers the history
that must be rebuilt.

## Limits

- This policy is for gas billing-period increments; electricity continues to
  use its own complete-hour allocation path.
- A multi-day reading that overlaps a `UsageSummary` is deliberately withheld
  rather than split or guessed.
- The component cannot establish provider billing semantics beyond the XML
  coverage it receives. If a provider uses different boundaries or values for
  summaries and readings, compare them with the provider bill before treating
  them as interchangeable.
