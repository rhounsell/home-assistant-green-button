# Historical Data Handling in Green Button Integration

## Overview

The Green Button integration handles historical data by **generating its own statistics and bypassing Home Assistant's automatic recorder**. This approach allows the integration to ingest years of historical interval data from Green Button feeds and populate the Energy Dashboard with accurate historical consumption data.

## Architecture

### 1. Manual Data Import Process

The integration relies entirely on **manual XML imports** rather than automatic API calls:

- **No automatic polling**: The coordinator has `update_interval=None` - it never fetches data automatically
- **Service-based imports**: New data is added via the `import_espi_xml` service call
- **Data merging**: Multiple XML imports are combined intelligently to avoid duplicates
- **Stored data**: XML data is persisted in the config entry for restart recovery

The `_async_update_data()` method only parses already-stored XML data and doesn't perform any external fetching.

### 2. Entity Creation with Duplicate Prevention

Sensors are created dynamically when data becomes available:

- **Entity Registry Checks**: Before creating any sensor, the integration checks if an entity with the same `unique_id` already exists in Home Assistant's entity registry
- **No Duplicate Creation**: If an entity already exists, creation is skipped with a debug log
- **Dynamic Creation**: Sensors are created when the coordinator has data, either during initial setup or after XML imports

### 3. Disables Automatic Recorder Statistics

The sensor is configured with `TOTAL_INCREASING` state class:

```python
_attr_state_class = SensorStateClass.TOTAL_INCREASING
```

While this signals to Home Assistant that the sensor is cumulative, the integration **prevents the recorder from auto-generating statistics** by:

- **Not calling `async_write_ha_state()`** in the coordinator update cycle
- The `_handle_coordinator_update()` method explicitly **does NOT** call `super()._handle_coordinator_update()`, which would trigger the recorder to auto-generate statistics

This is critical because the recorder's automatic statistics would create **corrupted records** (state/sum swap, massive consumption values) when given cumulative sensor data without proper context.

### 4. Manually Calculates Energy Values from Raw Interval Data

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

### 5. Imports Historical Statistics Using `async_import_statistics`

Instead of relying on the recorder to generate statistics from individual state changes, the integration:

- Calls `statistics.update_statistics()` (electricity) or `statistics.update_gas_statistics()` (gas) with the entire meter reading
- Uses Home Assistant's `async_import_statistics()` API to bulk-import statistics records
- Properly handles overlapping/duplicate data without corrupting records

**Gas vs Electricity Statistics:**
- **Electricity**: Uses `update_statistics()` with hourly bucketing
- **Gas**: Uses `update_gas_statistics()` with monthly_increment or daily_readings modes
- **Gas timing**: Statistics generated immediately in `async_added_to_hass()` if data exists
- **Electricity timing**: Statistics generated in `_handle_coordinator_update()` after data changes

## Data Flow

```
Manual XML Import (via import_espi_xml service)
    ↓
Parse into UsagePoint → MeterReading → IntervalBlocks → IntervalReadings
    ↓
GreenButtonCoordinator.async_add_xml_data()
    └─ Merges new data with existing data
    └─ Calls async_set_updated_data() to notify entities
    ↓
_async_create_entities() [Entity Creation]
    └─ Checks entity registry to prevent duplicates
    └─ Creates sensors if they don't exist
    ↓
GreenButtonSensor.async_added_to_hass() [All Sensors]
    └─ Initializes sensor state
    └─ Triggers immediate statistics generation if data exists
    ↓
GreenButtonSensor._handle_coordinator_update() [Electricity]
    └─ Called when coordinator data changes
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

GreenButtonGasSensor._handle_coordinator_update() [Gas]
    └─ Called when coordinator data changes
    ↓
update_sensor_and_statistics()
    └─ Calculate cumulative total from all interval readings
    └─ Call statistics.update_gas_statistics()
    └─ Uses monthly_increment or daily_readings mode
         ↓
         async_import_statistics() → Home Assistant Statistics Database
```

**Key Differences:**
- **Entity Creation**: Uses entity registry checks to prevent duplicates
- **Gas Statistics**: Generated immediately in `async_added_to_hass()` if data exists
- **Electricity Statistics**: Generated in `_handle_coordinator_update()` after data changes
- **No automatic fetching**: All data comes from manual XML imports
- **Coordinator role**: Manages data merging, not fetching

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
5. **Manual Control**: Allows users to import specific data periods as needed

## Implementation Details

### Statistics Generation (`statistics.py`)

**Electricity Statistics (`_generate_statistics_data`):**
- Collects all interval readings from meter readings
- Sorts them chronologically
- Groups readings into hourly buckets
- Calculates proportional energy for readings that span multiple hours
- Skips partial hours to avoid oversized last bars in Energy Dashboard
- Computes cumulative sums based on existing statistics

**Gas Statistics (`update_gas_statistics`):**
- Supports two modes: `daily_readings` (daily totals) and `monthly_increment` (monthly billing periods)
- For `monthly_increment`: Uses UsageSummary data to create monthly increment records
- For `daily_readings`: Aggregates daily gas consumption from interval readings
- Handles both interval-based data and summary-based data from different utility formats

### Metadata Management

The `create_metadata()` function creates Home Assistant `StatisticMetaData` with:
- `mean_type: StatisticMeanType.NONE` (not `ARITHMETIC` or `CIRCULAR`)
- `has_sum: True` (enable Energy Dashboard)
- `unit_of_measurement`: kWh, m³, or currency
- `unit_class`: None (flexible for multiple sensor types)

## Notes

- **Electricity sensors** do NOT call `async_write_ha_state()` during coordinator updates to prevent automatic recorder statistics
- **Gas sensors** call `async_write_ha_state()` in `async_added_to_hass()` to make entities available, but still prevent recorder statistics
- Statistics are updated after the sensor state is set, but before `async_write_ha_state()` for electricity sensors
- Gas statistics are generated immediately in `async_added_to_hass()` if data exists, rather than waiting for coordinator updates
- The integration uses `await` to ensure statistics are imported before returning control
- Duplicate or overlapping statistics are handled correctly by Home Assistant's `async_import_statistics()` API
- No automatic data fetching - all data comes from manual XML imports via the service
