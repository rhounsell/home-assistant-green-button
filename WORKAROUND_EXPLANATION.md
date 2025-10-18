# Monthly Increment Workaround for Mismatched Billing Periods

## Problem
Enbridge's Green Button API returns XML with:
- **UsageSummary**: Previous FINALIZED billing period (e.g., July 26 - Aug 24)
- **IntervalReading**: Current IN-PROGRESS billing period (e.g., Aug 25 - Sep 26)

When using `monthly_increment` allocation mode, only UsageSummary periods were used, causing:
- Bars to appear at the previous billing period end (Aug 24)
- Current billing period (Aug 25 - Sep 26) not shown

## Solution
The workaround detects long IntervalReadings (≥7 days) that don't overlap with any UsageSummary and treats them as billing periods.

### Logic Flow

1. **Collect UsageSummaries**
   - Add all UsageSummary entries as billing periods
   - Use `consumption_m3` from `currentBillingPeriodOverAllConsumption`

2. **Check IntervalReadings**
   - Find IntervalReadings with duration ≥7 days (likely billing periods)
   - Check if they overlap with any UsageSummary (>50% overlap)
   - If NO overlap → treat as a billing period

3. **Process All Periods**
   - Sort by period end date
   - Place bar at end of period (00:00 of end date)
   - Calculate cumulative sum

### What Happens When Next Month's XML Arrives?

#### Scenario: Current State (October 17)
- **UsageSummary**: July 26 - Aug 24 (52 m³) → Bar at Aug 24
- **IntervalReading**: Aug 25 - Sep 26 (54 m³) → Bar at Sep 26 (from workaround)

#### Next Month (After September 26)
Enbridge will provide:
- **UsageSummary #1**: July 26 - Aug 24 (52 m³) → Bar at Aug 24
- **UsageSummary #2**: Aug 25 - Sep 26 (54 m³) → Bar at Sep 26 (OFFICIAL)
- **IntervalReading**: Sep 27 - Oct XX (XX m³) → Bar at Oct XX (from workaround)

**The transition is seamless:**
1. The workaround-created Sep 26 bar gets replaced by the official UsageSummary bar
2. Both use the same date (Sep 26) and value (54 m³)
3. New workaround bar appears for Oct period

### Key Design Decisions

1. **50% Overlap Threshold**
   - Prevents duplicates when UsageSummary and IntervalReading cover same period
   - Allows for slight date mismatches

2. **Prefer Official Data**
   - UsageSummaries are always used when available
   - IntervalReadings only used when no matching UsageSummary exists

3. **Clear + Reimport**
   - Each update clears all statistics and rebuilds from scratch
   - Ensures old workaround bars are replaced by official data

4. **Minimum 7-Day Period**
   - Filters out daily IntervalReadings
   - Only treats multi-week readings as billing periods

## Code Changes

### File: `statistics.py`

#### Function: `update_gas_statistics()`

**Added:**
- `periods_to_process` list to hold billing periods from both sources
- Logic to scan IntervalReadings for long-duration readings
- Overlap detection to prevent duplicates
- Logging to show source of each billing period

**Modified:**
- Loop variable from `us` (UsageSummary) to tuple `(period_start, period_end, consumption_m3, source)`
- Fallback logic now uses `period_start` instead of `us.start`

## Testing

After implementing this workaround:

1. **Current behavior** (October 17):
   - Should see TWO bars:
     - Aug 24 (or Aug 22 in local time): 52 m³ from UsageSummary
     - Sep 26: 54 m³ from IntervalReading workaround

2. **After next XML import** (post-Sep 26):
   - Aug 24 bar remains (52 m³)
   - Sep 26 bar updated/confirmed (54 m³) from official UsageSummary
   - New bar for Oct period from IntervalReading workaround

3. **To verify**:
   - Delete statistics: `green_button.delete_statistics`
   - Re-import XML
   - Check Energy Dashboard for correct bar placement

## Limitations

- Only works for gas sensors (not electricity, which has different patterns)
- Requires IntervalReading to span ≥7 days
- Assumes Enbridge's pattern: previous UsageSummary + current IntervalReading
- If both UsageSummary and IntervalReading exist for same period with different values, UsageSummary wins

## Future Enhancements

Potential improvements:
1. Make MIN_BILLING_PERIOD_DAYS configurable
2. Add user option to prefer IntervalReading over UsageSummary
3. Detect and warn about value mismatches between sources
4. Support electricity with similar pattern
