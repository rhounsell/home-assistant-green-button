# Required Recorder Configuration

## Important: Prevent Duplicate Statistics

This integration manually imports statistics using `async_import_statistics()` to properly handle historical data. To prevent Home Assistant's automatic statistics compilation from creating duplicate or corrupted records, **you must exclude the Green Button sensors from the recorder**.

## Configuration

Add the following to your Home Assistant `configuration.yaml`:

```yaml
recorder:
  exclude:
    entities:
      - sensor.home_electricity_usage
      - sensor.home_electricity_cost
      - sensor.home_natural_gas_usage
      - sensor.home_natural_gas_cost
```

**Replace the entity IDs** with your actual entity IDs if they differ (e.g., if you named your integration something other than "Home").

### Alternative: Use Entity Globs

If you have multiple Green Button integrations or want a more flexible approach:

```yaml
recorder:
  exclude:
    entity_globs:
      - sensor.*_electricity_usage
      - sensor.*_electricity_cost
      - sensor.*_natural_gas_usage
      - sensor.*_natural_gas_cost
```

## Why This Is Necessary

1. **Green Button sensors have `state_class = TOTAL_INCREASING`** to satisfy Energy Dashboard requirements
2. **Sensor state shows cumulative totals** (e.g., 24,597 kWh) for UI reference
3. **Without recorder exclusion**, Home Assistant would:
   - See state changes (e.g., when new data is imported or HA restarts)
   - Automatically compile statistics from those state changes
   - Create massive "spikes" showing fake consumption equal to the cumulative total
   - Corrupt your Energy Dashboard data

4. **With recorder exclusion**:
   - HA's recorder ignores state changes for these entities
   - No automatic statistics compilation occurs
   - Manually imported statistics (via `async_import_statistics()`) are unaffected
   - Energy Dashboard uses the correct manually imported statistics

## After Configuration

1. **Restart Home Assistant** after adding the recorder exclusion
2. **Verify exclusion is working**: Check Developer Tools → Statistics
   - There should be no "Fix Issue" warnings
   - Statistics should match your imported data range

## What You'll See

- **Entity State**: Shows cumulative total (e.g., "24597.72 kWh")
- **Entity Status**: Available (green checkmark)
- **State History**: Not recorded (due to exclusion)
- **Statistics**: Manually imported data only (no auto-compilation)
- **Energy Dashboard**: Uses manually imported statistics

## Troubleshooting

**Problem**: Still seeing spikes in Energy Dashboard

**Solution**: 
1. Verify the recorder exclusion is in `configuration.yaml`
2. Restart Home Assistant
3. Delete the corrupted statistics using Developer Tools → Statistics → "Fix Issue" → Delete
4. Re-import your XML data

**Problem**: "Fix Issue" warning about missing state_class

**Solution**: You removed the recorder exclusion. Add it back and restart HA.
