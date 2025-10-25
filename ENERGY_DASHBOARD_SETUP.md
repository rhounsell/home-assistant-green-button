# Energy Dashboard Setup for Green Button Integration

## Important: Recorder Configuration Required

The Green Button integration manually manages statistics to avoid corruption. To prevent Home Assistant's Recorder from auto-generating duplicate statistics, you **MUST** add the following to your `configuration.yaml`:

```yaml
recorder:
  exclude:
    entity_globs:
      - sensor.*_electricity_consumption
      - sensor.*_electricity_consumption_cost
      - sensor.*_gas
      - sensor.*_gas_cost
```

### Why This Is Needed

1. **Energy Dashboard requires `state_class`** on energy sensors
2. **Recorder auto-generates statistics** when entities have `state_class`
3. **Green Button manually imports statistics** for historical data and custom logic
4. **Without exclusion**, you'll get duplicate/corrupted statistics

### Alternative: Exclude Specific Entities

If you want more control, exclude only your specific Green Button entities:

```yaml
recorder:
  exclude:
    entities:
      - sensor.home_electricity_consumption
      - sensor.home_electricity_consumption_cost
      # Add any other Green Button sensors here
```

### After Adding Configuration

1. Edit `configuration.yaml` and add the recorder exclusion
2. Restart Home Assistant
3. Delete any corrupted Green Button statistics (if they were created)
4. Reload the Green Button integration
5. Configure sensors in Energy Dashboard

## Expected Warning (Can Be Ignored)

You will see this warning on the Energy Dashboard configuration page:

```
Entity not tracked
Home Assistant Recorder has been configured to exclude these configured entities:
sensor.home_electricity_consumption
sensor.home_electricity_consumption_cost
```

**This is expected and harmless!** Your data WILL still appear in the Energy Dashboard because:
- The exclusion only prevents Recorder from auto-generating statistics from state changes
- Your manual statistics (via `async_import_statistics()`) are still imported
- Energy Dashboard reads from the statistics tables, not from entity states

If the graphs are showing data, everything is working correctly. You can safely ignore this warning.

## Verifying It's Working

After setup, check that:
1. ✅ Sensors show in Energy Dashboard configuration (not "unexpected state class")
2. ✅ Sensors are not "unknown" (have a numeric value)
3. ✅ **Warning about "Entity not tracked" appears** (this is correct!)
4. ✅ **Data appears in Energy Dashboard graphs** (the important part!)
5. ✅ No duplicate statistics in Developer Tools → Statistics

## Troubleshooting

### "Unexpected state class" Error

- You forgot to add `state_class` (should be fixed in latest version)
- **OR** you need to reload the integration after updating

### "Entity unavailable" Error

- Integration hasn't finished loading
- **OR** No data has been imported yet (sensors initialize to 0)

### Duplicate or Corrupted Statistics

- Recorder is auto-generating statistics (add exclusion to `configuration.yaml`)
- Delete corrupted statistics via Developer Tools → Statistics
- Restart HA and reload integration

### Energy Dashboard Shows Zero

- Statistics haven't been imported yet
- Check integration logs for import errors
- Verify XML files are in the correct location

## Technical Details

The Green Button integration:
- Sets `state_class = TOTAL_INCREASING` for energy sensors (consumption, gas)
- Sets `state_class = TOTAL` for monetary sensors (costs can decrease with refunds)
- Writes state ONCE at startup (makes entity available)
- Does NOT update state during normal operation (prevents auto-generation)
- Manually imports statistics via `async_import_statistics()` (for historical data)

This pattern allows Energy Dashboard compatibility while maintaining full control over statistics.
