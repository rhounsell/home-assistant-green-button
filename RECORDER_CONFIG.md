# No Recorder Configuration Required

## Energy Dashboard Compatibility

This integration is designed to work directly with the Energy Dashboard **without requiring any recorder configuration**.

## How It Works

1. **Sensors show "unknown" state** - This is intentional and prevents duplicate statistics
2. **Statistics are manually imported** - Using `async_import_statistics()` for proper historical data handling  
3. **Energy Dashboard uses statistics** - Not sensor states, so "unknown" state doesn't affect functionality

## Expected Behavior

### Sensor States
- **Entity Status**: Available (green checkmark)
- **Entity State**: "unknown"
- **Energy Dashboard**: Works perfectly with manually imported statistics

### "Fix Issue" Warning (Can Be Ignored)

You may see a warning in **Developer Tools → Statistics**:

> The entity no longer has a state class  
> We have generated statistics for 'Home Electricity Usage' (sensor.home_electricity_usage) in the past, but it no longer has a state class...

**This warning can be safely ignored or dismissed:**

1. Click **"Fix Issue"** 
2. Click **"IGNORE"** (not "Delete")
3. The warning will be dismissed permanently

### Why This Warning Appears

- The sensor has `state_class = TOTAL_INCREASING` (required for Energy Dashboard)
- But returns `None` for its state (prevents automatic statistics compilation)
- HA's validation sees this mismatch and warns you
- **The warning is cosmetic** - your statistics and Energy Dashboard work perfectly

## Why No Recorder Exclusion?

Initially, we tried excluding these entities from the recorder to prevent automatic statistics compilation. However, this caused a worse problem:

- **Energy Dashboard couldn't find the sensors** - Even though statistics existed in the database
- The Energy Dashboard checks if entities are being recorded before listing them

By returning `None` from `native_value`:
- No state history is recorded
- No automatic statistics compilation occurs  
- But the entity is still "known" to the recorder
- Energy Dashboard can find and use the statistics

## What You'll See

✅ Sensors appear in Energy Dashboard configuration  
✅ Energy Dashboard displays your imported usage data correctly  
✅ Statistics match your imported XML data range  
✅ No duplicate statistics or spikes  
⚠️ "Fix issue" warning (can be ignored/dismissed)

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
