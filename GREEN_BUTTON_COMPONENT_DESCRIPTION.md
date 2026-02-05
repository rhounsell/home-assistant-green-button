# Energy Dashboard Setup for Green Button Integration

## No Recorder Configuration Required

The Green Button integration is designed to work directly with the Energy Dashboard **without requiring any recorder exclusions**.

### How It Works

1. **Sensors have `state_class`** - Required for Energy Dashboard compatibility
2. **State only updates after XML import** - Prevents continuous state changes
3. **No automatic statistics compilation** - Because states don't change during normal operation
4. **Manual statistics import** - Historical data is imported via `async_import_statistics()`
5. **Last imported total displayed** - Sensors show the cumulative total from the last import

### Quick Setup

1. Install the Green Button integration
2. Import your XML data using the `green_button.import_xml` service
3. Configure sensors in Energy Dashboard
4. Done! No additional configuration needed

## Verifying It's Working

After setup, check that:
1. ✅ Sensors show in Energy Dashboard configuration (not "unexpected state class")
2. ✅ Sensors show numeric values (last imported cumulative total)
3. ✅ Sensors show as "Available" (green checkmark)
4. ✅ **No warnings** about "Entity not tracked" or "Fix issue"
5. ✅ **Data appears in Energy Dashboard graphs** (the important part!)
6. ✅ No duplicate statistics in Developer Tools → Statistics

## Troubleshooting

### "Unexpected state class" Error

- Reload the integration after updating to the latest version
- Restart Home Assistant if the issue persists

### "Entity unavailable" or Shows Zero

- No data has been imported yet
- Import XML data using the `green_button.import_xml` service
- Check integration logs for import errors
- Verify the coordinator has successfully parsed the XML data

### Duplicate or Corrupted Statistics

If you upgraded from an older version that used recorder exclusions:
1. Remove the old recorder exclusions from `configuration.yaml`
2. Delete corrupted statistics via Developer Tools → Statistics
3. Restart Home Assistant
4. Reload the Green Button integration
5. Re-import your XML data

### Energy Dashboard Shows Wrong Data

- Check that statistics exist using Developer Tools → Statistics
- Verify the date range of i prevents automatic statistics compilation by:

1. **Overriding coordinator update behavior** - Does not call `async_write_ha_state()` on every coordinator data update
2. **Writing state only after import** - State is written once after statistics are imported
3. **Returning cached value** - `native_value` returns the last imported total (prevents "unavailable" warnings)
4. **Manual statistics management** - Uses `async_import_statistics()` for complete control over historical data

### State Classes Used

- **Energy sensors** (consumption, gas): `TOTAL_INCREASING`
- **Cost sensors**: `TOTAL` (allows for refunds/credits)

### Why This Works

- **State changes only on import** - Not during normal operation or coordinator updates
- **No continuous state history** - Prevents Recorder's automatic statistics compilation
- **Manual statistics import** - Imports historical data with correct timestamps
- **Energy Dashboard compatible** - Sensors have valid states and proper `state_class`

This design provides:
- ✅ Full control over statistics
- ✅ No duplicate or corrupted data
- ✅ Energy Dashboard compatibility
- ✅ No recorder configuration needed
- ✅ No warnings or "Fix issue" messages
- Writes state ONCE at startup (makes entity available)
- Does NOT update state during normal operation (prevents auto-generation)
- Manually imports statistics via `async_import_statistics()` (for historical data)

This pattern allows Energy Dashboard compatibility while maintaining full control over statistics.
