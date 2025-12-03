# Historical Data Handling in Green Button Integration

## Overview

The Green Button integration handles historical data by **generating its own statistics and bypassing Home Assistant's automatic recorder**. This approach allows the integration to ingest years of historical interval data from Green Button feeds and populate the Energy Dashboard with accurate historical consumption data.

## Architecture

### 1. Disables Automatic Recorder Statistics

The sensor is configured with `TOTAL_INCREASING` state class:

```python
_attr_state_class = SensorStateClass.TOTAL_INCREASING
```

While this signals to Home Assistant that the sensor is cumulative, the integration **prevents the recorder from auto-generating statistics** by:

- **Not calling `async_write_ha_state()`** in the coordinator update cycle
- The `_handle_coordinator_update()` method explicitly **does NOT** call `super()._handle_coordinator_update()`, which would trigger the recorder to auto-generate statistics

This is critical because the recorder's automatic statistics would create **corrupted records** (state/sum swap, massive consumption values) when given cumulative sensor data without proper context.

### 2. Manually Calculates Energy Values from Raw Interval Data

In `update_sensor_and_statistics()`, the sensor:

```python
total_energy = 0
for interval_block in meter_reading.interval_blocks:
    for interval_reading in interval_block.interval_readings:
        if interval_reading.value is not None:
            # Apply power of ten multiplier from the ReadingType
            power_multiplier = interval_reading.reading_type.power_of_ten_multiplier
            value = interval_reading.value * (10**power_multiplier)
            total_energy += value

# Convert from Wh to kWh
self._attr_native_value = total_energy / 1000.0
```

This sums up all interval readings from the Green Button data and stores the cumulative total as the sensor state.

### 3. Imports Historical Statistics Using `async_import_statistics`

Instead of relying on the recorder to generate statistics from individual state changes, the integration:

- Calls `statistics.update_statistics()` with the entire meter reading
- Uses Home Assistant's `async_import_statistics()` API to bulk-import statistics records
- Properly handles overlapping/duplicate data without corrupting records

```python
await statistics.update_statistics(
    self.hass,
    self,
    statistics.DefaultDataExtractor(),
    meter_reading,
)
```

### 4. Generates Statistics with Correct Metadata

The `create_metadata()` function defines how statistics should be recorded:

```python
def create_metadata(entity: GreenButtonEntity) -> statistics.StatisticMetaData:
    return {
        "mean_type": StatisticMeanType.NONE,  # Cumulative total, not averaged
        "has_sum": True,                       # Track the sum value
        "statistic_id": entity.long_term_statistics_id,
        "unit_of_measurement": entity.native_unit_of_measurement,
        "unit_class": None,
    }
```

Key points:
- **`mean_type: NONE`** — indicates this is a cumulative total, not an averaged value
- **`has_sum: True`** — enables the Energy Dashboard to display consumption data
- **`unit_class: None`** — allows flexibility across energy, gas, and monetary sensors

## Data Flow

```
Green Button XML Feed (e.g., from Enbridge, Hydro One)
    ↓
Parse into UsagePoint → MeterReading → IntervalBlocks → IntervalReadings
    ↓
GreenButtonCoordinator (coordinator.py)
    └─ Fetches and parses Green Button data periodically
    ↓
GreenButtonSensor._handle_coordinator_update()
    └─ Called on coordinator update
    ↓
update_sensor_and_statistics()
    ├─ Calculate cumulative total from all interval readings
    ├─ Set sensor state (visible in UI)
    └─ Call statistics.update_statistics()
         ↓
         statistics._generate_statistics_data()
         └─ Splits interval readings into hourly buckets
         └─ Generates hourly records with cumulative sums
         ↓
         async_import_statistics() → Home Assistant Statistics Database
         └─ Bypasses Recorder entirely
         └─ Handles overlapping/duplicate data correctly
```

## Why This Approach is Better

| Aspect | Recorder Auto-Stats | Manual `async_import_statistics` |
|--------|-------------------|----------------------------------|
| **Historical Data** | ❌ Can only generate from current state changes | ✅ Imports all historical intervals from Green Button data |
| **Data Accuracy** | ❌ Creates corrupted records (state/sum swap) | ✅ Correctly maps interval values to statistics |
| **Energy Dashboard** | ❌ Fails or shows incorrect consumption | ✅ Shows accurate historical consumption |
| **Backfill Capability** | ❌ Limited to real-time data | ✅ Can backfill years of historical data |
| **Flexibility** | ❌ Fixed to real-time state-based stats | ✅ Can handle any interval length (15min, hourly, daily) |

## Benefits

1. **Complete Historical Data**: Can ingest years of interval data from Green Button feeds
2. **Accurate Energy Dashboard**: Shows correct historical consumption patterns
3. **Data Integrity**: Avoids corrupted statistics records from improper state/sum tracking
4. **Flexibility**: Handles multiple interval types (electricity, gas, cost)
5. **No Manual Intervention**: Automatically backfills and imports data on first setup

## Implementation Details

### Statistics Generation (`statistics.py`)

The `_generate_statistics_data()` function:
- Collects all interval readings from meter readings
- Sorts them chronologically
- Groups readings into hourly buckets
- Calculates proportional energy for readings that span multiple hours
- Skips partial hours to avoid oversized last bars in Energy Dashboard
- Computes cumulative sums based on existing statistics

### Metadata Management

The `create_metadata()` function creates Home Assistant `StatisticMetaData` with:
- `mean_type: StatisticMeanType.NONE` (not `ARITHMETIC` or `CIRCULAR`)
- `has_sum: True` (enable Energy Dashboard)
- `unit_of_measurement`: kWh, m³, or currency
- `unit_class`: None (flexible for multiple sensor types)

## Notes

- The sensor **does NOT** call `async_write_ha_state()` during coordinator updates to prevent automatic recorder statistics
- Statistics are updated **after** the sensor state is set, but before `async_write_ha_state()` is called
- The integration uses `await` to ensure statistics are imported before returning control
- Duplicate or overlapping statistics are handled correctly by Home Assistant's `async_import_statistics()` API
