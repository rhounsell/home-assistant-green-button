# No Recorder Configuration Required

## Energy Dashboard Compatibility

This integration is designed to work directly with the Energy Dashboard **without requiring any recorder configuration**.

## How It Works

1. **Sensors show last imported total** - Displays the cumulative total from your last XML import
2. **Statistics are manually imported** - Using `async_import_statistics()` for proper historical data handling  
3. **State only updates after import** - No state changes during normal operation prevents automatic statistics compilation
4. **Energy Dashboard uses statistics** - Not live sensor states, so no duplicate statistics are generated

## Expected Behavior

### Sensor States
- **Entity Status**: Available (green checkmark)
- **Entity State**: Shows last imported cumulative total (e.g., "23809.346 kWh")
- **State Updates**: Only after importing new XML data
- **Energy Dashboard**: Works perfectly with manually imported statistics

### No Warnings Expected

The sensors will show valid numeric states after statistics import, so there should be no "Entity unavailable" warnings or "Fix issue" messages in the Energy Dashboard.

## Why No Recorder Exclusion?

Initially, we tried excluding these entities from the recorder to prevent automatic statistics compilation. However, this caused a problem:

- **Energy Dashboard couldn't find the sensors** - Even though statistics existed in the database
- The Energy Dashboard checks if entities are being recorded before listing them

By returning the last imported total from the sensor state but only writing it after statistics import:
- State is written once per import (not on every coordinator update)
- No continuous state history means no automatic statistics compilation
- But the entity is still "known" to the recorder with a valid state
- Energy Dashboard can find the sensors and shows no warnings

## What You'll See

✅ Sensors appear in Energy Dashboard configuration immediately 
✅ Sensors show valid numeric states (last imported total)
✅ Energy Dashboard displays your imported usage data correctly  
✅ Statistics match your imported XML data range  
✅ No duplicate statistics or spikes
✅ No "Entity unavailable" warnings
✅ No "Fix issue" warnings

## Troubleshooting

**Problem**: Sensors not appearing in Energy Dashboard

**Solution**:
1. Verify sensors exist in **Settings → Devices & Services → Green Button**
2. Check that statistics exist using **Developer Tools → Statistics → Get Statistics**
3. Restart Home Assistant
4. Try removing and re-adding the integration

**Problem**: Still seeing spikes in Energy Dashboard

**Solution**: 
1. Make sure you're running the latest version of the integration
2. Delete corrupted statistics using **Developer Tools → Statistics**
3. Restart Home Assistant
4. Re-import your XML data
