# AliveDrive Telemetry Format Specification

Version 2.5 Used in GM Performance Data Recorder (PDR) systems (2026 Corvette, 2025+ CT5-V Blackwing).

---

## 1. Overview

Telemetry data is embedded inside standard MP4 video files as a custom metadata track.
The MP4 file contains:
- One or more video tracks (the camera footage)
- One **AliveDrive telemetry track** (handler type `"adrv"`)

The telemetry track uses the standard MP4 sample table mechanism (`stbl`, `stco`/`co64`,
`stsz`, `stts`, `stsc`) to store time-indexed telemetry samples. A custom container box
(`"adco"`) in the sample description entry defines how to interpret those samples.

All multi-byte integers are **big-endian** (network byte order), per the MP4/ISO BMFF
specification. Strings are **null-terminated UTF-8**.

---

## 2. MP4 Box Hierarchy

```
ftyp
moov
 ├─ mvhd
 ├─ trak (video)
 │   └─ ...
 └─ trak (telemetry)
     ├─ tkhd                        Track header
     └─ mdia
         ├─ mdhd                    Media header
         ├─ hdlr (handler: "adrv")  Identifies this as an AliveDrive track
         └─ minf
             ├─ dinf
             │   └─ dref
             │       └─ url         Data entry URL (self-contained flag)
             └─ stbl
                 ├─ stsd
                 │   └─ adco        AliveDrive Container (sample entry)
                 │       ├─ advi    Version Information
                 │       ├─ adop    Outing Properties (session metadata)
                 │       ├─ adud    Units Database
                 │       ├─ adcr    Channel Rates
                 │       ├─ adcp    Channel Properties
                 │       └─ adeg    Event Groups
                 ├─ stts             Sample-to-time table
                 ├─ stsc             Sample-to-chunk table
                 ├─ stsz             Sample sizes
                 └─ stco / co64      Chunk offsets
mdat                                 Raw sample data lives here
```

---

## 3. AliveDrive Container Box (`adco`)

**FourCC:** `"adco"`
**Type:** MP4 Sample Entry (in the `stsd` box)

### Binary Layout

```
Offset  Size    Field
0       6       Reserved (zeros) — standard MP4 sample entry
6       2       uint16 data_reference_index (always 1)
8       ...     Child boxes (advi, adop, adud, adcr, adcp, adeg)
```

The child boxes are standard MP4 boxes (4-byte size + 4-byte FourCC + content).

---

## 4. Version Information Box (`advi`)

**FourCC:** `"advi"`

Contains version metadata about the recording source.

### Binary Layout

```
Offset  Size    Field
0       4       Format Version: uint16 major + uint16 minor → Version(major, minor, 0, 0)
4       6       MMP Source Version: uint16 major + uint16 minor + uint16 build → Version(major, minor, 0, build)
10      6       VIP Source Version: same format as above
16      6       App Source Version: same format as above
22      var     Source Tag: null-terminated UTF-8 string (e.g. "com.cosworth.outing.source.pdr2_5")
```

---

## 5. Outing Properties Box (`adop`)

**FourCC:** `"adop"`

Contains key-value metadata about the recording session (vehicle info, GPS bounds,
timestamps, track info, etc.). Properties are stored sequentially until the end of the box.

### Binary Layout (repeating)

```
For each property:
  var     Tag:           null-terminated UTF-8 string (e.g. "com.cosworth.outingproperty.timestamp")
  4       PropertyType:  uint32 FourCC identifying the value type (see table below)
  var     Value:         type-specific payload (see below)
```

### Property Type FourCCs

| FourCC   | Hex          | Uint32 Value | Type Name            | Value Format                                       |
|----------|--------------|--------------|----------------------|----------------------------------------------------|
| `strn`   | `0x7374726E` | 1937011310   | NullTerminatedString | Null-terminated UTF-8 string                       |
| `dtim`   | `0x6474696D` | 1685350765   | DateTime             | 25-byte fixed-length UTF-8: `yyyy-MM-ddTHH:mm:sszzz` |
| `tmzn`   | `0x746D7A6E` | 1953331822   | TimeZone             | Null-terminated UTF-8 string                       |
| `tstm`   | `0x7473746D` | 1953723501   | EpochTicks           | uint64: .NET ticks since Unix epoch (see below)    |
| `guid`   | `0x67756964` | 1735747940   | Guid                 | 16 bytes: big-endian UUID (RFC 4122 byte order)    |
| `siva`   | `0x73697661` | 1936291425   | SIValue              | uint16 quantityUnitId + byte dataType + value      |
| `tcks`   | `0x74636B73` | 1952672627   | Ticks                | int64: .NET ticks (TimeSpan)                       |
| `vrsn`   | `0x7672736E` | 1987212142   | Version              | uint16 major + uint16 minor + uint16 build         |

### EpochTicks Detail

The value is a **uint64** count of .NET ticks (1 tick = 100 nanoseconds) since the Unix
epoch (1970-01-01T00:00:00.001Z — note the 1ms offset in the Cosworth implementation).
Values > `Int64.MaxValue` (0x7FFFFFFFFFFFFFFF) are treated as invalid and skipped.

To convert to a Unix timestamp in seconds:
```
unix_seconds = epoch_ticks / 10_000_000
```

### GUID Detail

GUIDs are stored in big-endian (RFC 4122) byte order. To convert to the standard
little-endian representation used by most platforms:
- Reverse bytes 0-3 (uint32 time_low)
- Reverse bytes 4-5 (uint16 time_mid)
- Reverse bytes 6-7 (uint16 time_hi_and_version)
- Bytes 8-15 remain unchanged

### SIValue Detail

```
  2       uint16 quantityUnitId    (references the Units Database)
  1       byte   dataType          (NumericDataType enum)
  var     value                    (size determined by dataType)
```

### Common Outing Property Tags

| Tag | Type | Description |
|-----|------|-------------|
| `com.cosworth.outingproperty.outing.id` | Guid | Unique outing ID |
| `com.cosworth.outingproperty.timestamp` | DateTime | Recording timestamp with timezone |
| `com.cosworth.outingproperty.source.tag` | String | Source identifier (e.g. `com.cosworth.outing.source.pdr2_5`) |
| `com.cosworth.outingproperty.telemetry.id` | Guid | Telemetry definition ID |
| `com.cosworth.outingproperty.telemetry.version` | Version | Telemetry version |
| `com.cosworth.outingproperty.telemetry.schemaversion` | Version | Schema version |
| `com.cosworth.outingproperty.location.center.latitude` | SIValue | GPS center latitude (radians) |
| `com.cosworth.outingproperty.location.center.longitude` | SIValue | GPS center longitude (radians) |
| `com.cosworth.outingproperty.location.minimum.latitude` | SIValue | GPS bounding box min latitude |
| `com.cosworth.outingproperty.location.minimum.longitude` | SIValue | GPS bounding box min longitude |
| `com.cosworth.outingproperty.location.maximum.latitude` | SIValue | GPS bounding box max latitude |
| `com.cosworth.outingproperty.location.maximum.longitude` | SIValue | GPS bounding box max longitude |
| `com.cosworth.outingproperty.location.starting.latitude` | SIValue | Starting position latitude |
| `com.cosworth.outingproperty.location.starting.longitude` | SIValue | Starting position longitude |
| `com.cosworth.outingproperty.track.id` | Guid | Track identifier |
| `com.cosworth.outingproperty.track.name` | String | Track name |
| `com.cosworth.outingproperty.vehicle.name` | String | Vehicle name |
| `com.cosworth.outingproperty.vehicle.make` | String | Vehicle make |
| `com.cosworth.outingproperty.vehicle.model` | String | Vehicle model |
| `com.cosworth.outingproperty.vehicle.modelyear` | String | Model year |
| `com.cosworth.outingproperty.vehicle.vin` | String | VIN |
| `com.cosworth.outingproperty.vehicle.powertrain.type` | String | ICE / Electric / Hybrid |
| `com.cosworth.outingproperty.vehicle.engine.type` | String | Engine type |
| `com.cosworth.outingproperty.vehicle.engine.revlimit` | SIValue | Rev limiter (RPM) |
| `com.cosworth.outingproperty.total.distance` | SIValue | Total distance traveled |
| `com.cosworth.outingproperty.maximum.speed` | SIValue | Maximum speed |
| `com.cosworth.outingproperty.duration` | Ticks | Recording duration |
| `com.cosworth.outingproperty.fastest.laptime` | Ticks | Fastest lap time |
| `com.cosworth.outingproperty.unitscheme` | String | Unit system (UK / US / Metric) |
| `com.cosworth.outingproperty.outing.type.tag` | String | Circuit / Autocross / Rally / DragRace |
| `com.cosworth.outingproperty.app.tag` | String | Source application |
| `com.cosworth.outingproperty.app.version` | Version | Application version |
| `com.cosworth.outingproperty.country.code` | String | Country code |

---

## 6. Units Database Box (`adud`)

**FourCC:** `"adud"`

Maps numeric IDs to unit tag strings. Used by channel properties and SI values.

### Binary Layout (repeating until end of box)

```
For each entry:
  2       uint16  id
  var     string  tag    (null-terminated UTF-8)
```

Unit tags follow the pattern `com.cosworth.unit.<quantity>.<unit>.si` for SI units,
or `com.cosworth.unit.none.none` for dimensionless quantities.

### Unit Tag Examples

| Tag | Quantity |
|-----|----------|
| `com.cosworth.unit.temperature.kelvin.si` | Temperature (SI) |
| `com.cosworth.unit.velocity.metrespersecond.si` | Velocity (SI) |
| `com.cosworth.unit.pressure.pascal.si` | Pressure (SI) |
| `com.cosworth.unit.acceleration.metrespersecondsquared.si` | Acceleration (SI) |
| `com.cosworth.unit.angle.radian.si` | Angle (SI) |
| `com.cosworth.unit.angularvelocity.radianspersecond.si` | Angular velocity (SI) |
| `com.cosworth.unit.length.metre.si` | Length (SI) |
| `com.cosworth.unit.time.second.si` | Time (SI) |
| `com.cosworth.unit.voltage.volt.si` | Voltage (SI) |
| `com.cosworth.unit.power.watt.si` | Power (SI) |
| `com.cosworth.unit.torque.newtonmetre.si` | Torque (SI) |
| `com.cosworth.unit.none.none` | Dimensionless |

**Important:** All values in the telemetry stream are stored in SI units. Display
conversions (to MPH, Fahrenheit, PSI, etc.) happen in the application layer.

---

## 7. Channel Rates Box (`adcr`)

**FourCC:** `"adcr"`

Defines sample rate groups and which channels belong to each rate. This is the key
structure that tells you how to decode sample data.

### Binary Layout

```
1       byte    numRateTables

For each RateTable (repeated numRateTables times):
  1       byte    rateTableId
  1       byte    numRateDefinitions

  For each RateDefinition (repeated numRateDefinitions times):
    8       uint64  interval        (.NET ticks; 1 tick = 100 nanoseconds)
    2       uint16  numChannels

    For each ChannelDefinition (repeated numChannels times):
      2       uint16  channelId
      1       byte    rawDataType   (NumericDataType enum)
```

### Interval Examples

| Interval (ticks) | Interval (ms) | Frequency |
|-------------------|---------------|-----------|
| 100,000           | 10 ms         | 100 Hz    |
| 200,000           | 20 ms         | 50 Hz     |
| 500,000           | 50 ms         | 20 Hz     |
| 1,000,000         | 100 ms        | 10 Hz     |
| 10,000,000        | 1,000 ms      | 1 Hz      |

Channels that update faster (e.g. accelerometer at 100 Hz) are in a rate definition
with a shorter interval than slower channels (e.g. GPS at 10 Hz).

---

## 8. Channel Properties Box (`adcp`)

**FourCC:** `"adcp"`

Maps channel IDs to their human-readable tags, quantity types, and calibration
parameters. Read sequentially until the end of the box.

### Binary Layout (repeating until end of box)

```
For each channel:
  2       uint16  channelId           (matches IDs in adcr)
  var     string  tag                 (null-terminated UTF-8, e.g. "com.cosworth.channel.throttle.position")
  2       uint16  quantityId          (references adud Units Database)
  --- Calibration Definition ---
  1       byte    calibrationType     (see CalibrationDefinitionType below)
  1       byte    calibratedDataType  (NumericDataType enum — the output type after calibration)
  var     ...     calibration-specific data (see below)
```

### CalibrationDefinitionType

| Value | Type    |
|-------|---------|
| 1     | Numeric |
| 2     | BitField |

### Numeric Calibration (calibrationType = 1)

Used for continuous data channels (throttle position, temperature, speed, etc.).

```
  8       float64   gain
  8       float64   offset
  N       <raw>     minimum     (size determined by rawDataType from adcr)
  N       <raw>     maximum     (size determined by rawDataType from adcr)
```

**Calibration formula:**
```
calibrated_value = raw_value * gain + offset
```

The `minimum` and `maximum` define the valid range of raw values. Values outside this
range should be clamped.

### BitField Calibration (calibrationType = 2)

Used for discrete/enumerated channels (gear, drive mode, stability control state, etc.).

```
  1       byte    numBitFields

  For each BitField (repeated numBitFields times):
    var     string  tag             (null-terminated UTF-8, e.g. "com.cosworth.channel.gear")
    N       <cal>   mask            (size from calibratedDataType — bitmask to extract this field)
    --- Default Entry ---
    var     string  defaultTag      (null-terminated UTF-8)
    N       <cal>   defaultValue    (size from calibratedDataType)
    --- Named Entries ---
    1       byte    numEntries
    For each entry:
      var     string  entryTag      (null-terminated UTF-8)
      N       <cal>   entryValue    (size from calibratedDataType)
```

To decode a BitField channel:
1. Read the raw value from the sample data
2. Apply the mask: `masked = raw_value & mask`
3. Look up `masked` in the entries; if not found, use the default

---

## 9. Event Groups Box (`adeg`)

**FourCC:** `"adeg"`

Maps event group IDs to tag strings. Same dictionary format as `adud`.

### Binary Layout (repeating until end of box)

```
For each entry:
  2       uint16  id
  var     string  tag    (null-terminated UTF-8)
```

### Event Group Tags

| Tag | Meaning |
|-----|---------|
| `com.cosworth.timingevent.started` | Timing event started |
| `com.cosworth.timingevent.finished` | Timing event finished |
| `com.cosworth.timingevent.aborted` | Timing event aborted |

---

## 10. NumericDataType Enum

Used throughout the format to specify the binary encoding of values.

| Value | Name | Size (bytes) | Description |
|-------|------|--------------|-------------|
| 1     | S8   | 1            | Signed 8-bit integer |
| 2     | U8   | 1            | Unsigned 8-bit integer |
| 3     | S16  | 2            | Signed 16-bit integer (big-endian) |
| 4     | U16  | 2            | Unsigned 16-bit integer (big-endian) |
| 5     | S32  | 4            | Signed 32-bit integer (big-endian) |
| 6     | U32  | 4            | Unsigned 32-bit integer (big-endian) |
| 7     | S64  | 8            | Signed 64-bit integer (big-endian) |
| 8     | U64  | 8            | Unsigned 64-bit integer (big-endian) |
| 9     | F32  | 4            | IEEE 754 single-precision float (big-endian) |
| 10    | F64  | 8            | IEEE 754 double-precision float (big-endian) |

---

## 11. Sample Data Format

The actual time-series telemetry data is stored as **MP4 samples** in the `mdat` box,
indexed by the standard sample tables (`stts`, `stsz`, `stco`/`co64`, `stsc`) in the
`stbl` box of the `adrv` track.

### Sample Structure

Each sample represents one time slice and contains the raw values for all channels
that have data at that timestamp. The sample is composed of rate groups:

```
For each RateTable (in order of rateTableId):
  For each RateDefinition within the table where (sample_timestamp % interval == 0):
    For each ChannelDefinition (in order of channelId):
      N bytes   raw_value    (size determined by rawDataType)
```

### Decoding a Sample

1. **Determine the timestamp** of this sample from the `stts` (sample-to-time) table
2. **For each rate definition**, check if this sample includes data for that rate:
   - `has_data = (timestamp_ticks % rate_interval_ticks) == 0`
3. **If yes**, read the raw values for each channel in that rate definition, in
   channel ID order
4. **Apply calibration** to get the final value:
   - Numeric: `calibrated = raw * gain + offset`
   - BitField: `value = raw & mask`, then look up in entry table

### Timing

The `stts` box gives the duration of each sample in the track's timescale. The media
header box (`mdhd`) provides the timescale (ticks per second). Combined, these let you
compute the precise timestamp for each sample.

Alternatively, since the rate definitions specify intervals in .NET ticks, you can
compute timestamps as:
```
sample_timestamp = sample_index * fastest_rate_interval
```

---

## 12. Channel Tag Reference

### GPS & Location

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.gps.utctime` | GPS UTC timestamp | seconds |
| `com.cosworth.channel.gps.latitude` | Latitude | radians |
| `com.cosworth.channel.gps.longitude` | Longitude | radians |
| `com.cosworth.channel.gps.altitude` | Altitude | metres |
| `com.cosworth.channel.gps.speed` | GPS-derived speed | m/s |
| `com.cosworth.channel.gps.heading` | Heading | radians |
| `com.cosworth.channel.gps.accuracy.horizontal` | Horizontal accuracy | metres |
| `com.cosworth.channel.gps.accuracy.vertical` | Vertical accuracy | metres |
| `com.cosworth.channel.gps.fix.quality` | Fix quality | dimensionless |
| `com.cosworth.channel.gps.satellites` | Satellite count | dimensionless |

### Accelerometer & Gyroscope

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.accelerometer.vehicle.x` | Lateral acceleration | m/s^2 |
| `com.cosworth.channel.accelerometer.vehicle.y` | Longitudinal acceleration | m/s^2 |
| `com.cosworth.channel.accelerometer.vehicle.z` | Vertical acceleration | m/s^2 |
| `com.cosworth.channel.accelerometer.device.x` | Device-frame X acceleration | m/s^2 |
| `com.cosworth.channel.accelerometer.device.y` | Device-frame Y acceleration | m/s^2 |
| `com.cosworth.channel.accelerometer.device.z` | Device-frame Z acceleration | m/s^2 |
| `com.cosworth.channel.gyro.vehicle.pitchrate` | Pitch rate | rad/s |
| `com.cosworth.channel.gyro.vehicle.rollrate` | Roll rate | rad/s |
| `com.cosworth.channel.gyro.vehicle.yawrate` | Yaw rate | rad/s |

### Engine & Drivetrain

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.enginespeed` | Engine RPM | rad/s (convert: RPM = rad/s * 60 / 2π) |
| `com.cosworth.channel.throttle.position` | Throttle position | proportion (0-1) |
| `com.cosworth.channel.brake.position` | Brake position | proportion (0-1) |
| `com.cosworth.channel.gear` | Current gear | dimensionless |
| `com.cosworth.channel.steering.angle` | Steering angle | radians |
| `com.cosworth.channel.location.speed` | Vehicle speed | m/s |
| `com.cosworth.channel.location.distance` | Odometer distance | metres |

### Temperatures & Pressures

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.oiltemp` | Oil temperature | kelvin |
| `com.cosworth.channel.oilpressure` | Oil pressure | pascals |
| `com.cosworth.channel.coolanttemp` | Coolant temperature | kelvin |
| `com.cosworth.channel.coolant.pressure` | Coolant pressure | pascals |
| `com.cosworth.channel.transmission.oil.temperature` | Trans oil temp | kelvin |
| `com.cosworth.channel.intake.air.temperature` | Intake air temp | kelvin |
| `com.cosworth.channel.intake.air.pressure` | Intake manifold pressure | pascals |
| `com.cosworth.channel.intake.air.boost.pressure` | Boost pressure | pascals |
| `com.cosworth.channel.outside.air.temperature` | Outside air temp | kelvin |

### Wheels & Tires

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.wheelspeed.frontleft` | Front-left wheel speed | m/s |
| `com.cosworth.channel.wheelspeed.frontright` | Front-right wheel speed | m/s |
| `com.cosworth.channel.wheelspeed.rearleft` | Rear-left wheel speed | m/s |
| `com.cosworth.channel.wheelspeed.rearright` | Rear-right wheel speed | m/s |
| `com.cosworth.channel.tirepressure.frontleft` | FL tire pressure | pascals |
| `com.cosworth.channel.tirepressure.frontright` | FR tire pressure | pascals |
| `com.cosworth.channel.tirepressure.rearleft` | RL tire pressure | pascals |
| `com.cosworth.channel.tirepressure.rearright` | RR tire pressure | pascals |
| `com.cosworth.channel.tiretemperature.frontleft` | FL tire temperature | kelvin |
| `com.cosworth.channel.tiretemperature.frontright` | FR tire temperature | kelvin |
| `com.cosworth.channel.tiretemperature.rearleft` | RL tire temperature | kelvin |
| `com.cosworth.channel.tiretemperature.rearright` | RR tire temperature | kelvin |

### Suspension

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.suspension.displacement.frontleft` | FL suspension travel | metres |
| `com.cosworth.channel.suspension.displacement.frontright` | FR suspension travel | metres |
| `com.cosworth.channel.suspension.displacement.rearleft` | RL suspension travel | metres |
| `com.cosworth.channel.suspension.displacement.rearright` | RR suspension travel | metres |
| `com.cosworth.channel.roll.angle` | Roll angle | radians |
| `com.cosworth.channel.pitch.angle` | Pitch angle | radians |

### Electric / Hybrid Powertrain

| Tag | Description | Typical Unit (SI) |
|-----|-------------|-------------------|
| `com.cosworth.channel.engine.power` | Engine power | watts |
| `com.cosworth.channel.electric.motor.power` | Electric motor power | watts |
| `com.cosworth.channel.engine.torque` | Engine torque | N·m |
| `com.cosworth.channel.electric.motor.torque` | Electric motor torque | N·m |
| `com.cosworth.channel.customer.usable.stateofcharge` | Battery SOC | proportion (0-1) |
| `com.cosworth.channel.battery.voltage` | Battery voltage | volts |
| `com.cosworth.channel.hv.battery.average.temperature` | HV battery avg temp | kelvin |

### Vehicle Control Systems (BitField channels)

| Tag | Description |
|-----|-------------|
| `com.cosworth.channel.drive.performance.mode` | Drive mode |
| `com.cosworth.channel.tractioncontrolactive` | Traction control state |
| `com.cosworth.channel.absactive` | ABS state |
| `com.cosworth.channel.vehicle.stability.enhancement` | Stability control |
| `com.cosworth.channel.electronic.stability.control` | ESC state |
| `com.cosworth.channel.engine.start.stop.state` | Start/stop state |

### Performance Metrics

| Tag | Description |
|-----|-------------|
| `com.cosworth.channel.laptime` | Current lap time |
| `com.cosworth.channel.timedelta` | Time delta vs reference |
| `com.cosworth.channel.corner.radius` | Corner radius |
| `com.cosworth.channel.gps.distance` | GPS-derived distance |

---

## 13. Outing Sources

The `com.cosworth.outingproperty.source.tag` property identifies the data source:

| Tag | Description |
|-----|-------------|
| `com.cosworth.outing.source.pdr1` | GM PDR Generation 1 |
| `com.cosworth.outing.source.pdr2` | GM PDR Generation 2 |
| `com.cosworth.outing.source.pdr2_5` | GM PDR Generation 2.5 |
| `com.cosworth.outing.source.dxr` | DXR |
| `com.cosworth.outing.source.oemdemo` | OEM Demo mode |

---

## 14. SI Unit Conversions

Since all data is stored in SI units, common display conversions:

| From (SI) | To (Display) | Formula |
|-----------|-------------|---------|
| m/s → MPH | `mph = m_s * 2.23694` |
| m/s → KPH | `kph = m_s * 3.6` |
| Kelvin → °C | `celsius = kelvin - 273.15` |
| Kelvin → °F | `fahrenheit = kelvin * 9/5 - 459.67` |
| Pascal → PSI | `psi = pascal * 0.000145038` |
| Pascal → bar | `bar = pascal / 100000` |
| Radians → Degrees | `degrees = radians * 180 / π` |
| rad/s → RPM | `rpm = rad_s * 60 / (2π)` |
| rad/s → °/s | `deg_s = rad_s * 180 / π` |
| N·m → lb·ft | `lbft = nm * 0.737562` |
| Watts → HP | `hp = watts / 745.7` |
| Metres → Feet | `feet = metres * 3.28084` |
| Metres → Miles | `miles = metres / 1609.344` |

---

## 15. Worked Example: Parsing Overview

To extract telemetry from an MP4 file:

### Step 1: Find the AliveDrive Track

Parse the MP4 box hierarchy. Find the `trak` box whose `hdlr` box has handler type
`"adrv"`. The sample description (`stsd`) box within this track's `stbl` contains
the `adco` box.

### Step 2: Parse the Channel Metadata

From the `adco` box, read in order:
1. **`advi`** — verify format version
2. **`adud`** — build a `dict[uint16 → string]` for unit tag lookup
3. **`adcr`** — build the rate table structure (which channels, at what rate, what type)
4. **`adcp`** — build channel properties (tag name, calibration for each channel ID)
5. **`adeg`** — build event group lookup

### Step 3: Read Sample Data

Using the standard MP4 sample tables (`stts`, `stsz`, `stco`, `stsc`):
1. Locate each sample's byte offset and size in `mdat`
2. For each sample, determine its timestamp
3. Decode the sample by iterating rate definitions and reading raw values
4. Apply calibration (gain * raw + offset) to get physical values
5. Map channel IDs to tag strings for meaningful labels

### Step 4: Convert Units

All values are in SI. Convert to desired display units using the table in Section 14.

---

## Appendix A: Legacy Aliases

Many channels have legacy aliases for backward compatibility. The canonical tag is listed
in the Channel Tag Reference above. Legacy aliases include:

| Canonical Tag | Legacy Alias(es) |
|---------------|-----------------|
| `com.cosworth.channel.throttle.position` | `com.cosworth.channel.accelpos`, `com.cosworth.channel.accelerator` |
| `com.cosworth.channel.brake.position` | `com.cosworth.channel.brakepos` |
| `com.cosworth.channel.enginespeed` | `com.cosworth.channel.rpm` |
| `com.cosworth.channel.oiltemp` | (also `com.cosworth.channel.oiltemp` — same) |
| `com.cosworth.channel.accelerometer.vehicle.x` | `com.cosworth.channel.lateralacceleration` |
| `com.cosworth.channel.gps.fix.quality` | `com.cosworth.channel.gpsprecision`, `com.cosworth.channel.gpsfix` |
| `com.cosworth.channel.tractioncontrolactive` | `com.cosworth.channel.tcs` |
| `com.cosworth.channel.absactive` | `com.cosworth.channel.abs` |

---

## Appendix B: Track Types

| Tag | Description |
|-----|-------------|
| `com.cosworth.track.type.circuit` | Circuit / road course |
| `com.cosworth.track.type.autocross` | Autocross |
| `com.cosworth.track.type.rally` | Rally |
| `com.cosworth.track.type.dragrace` | Drag race |

---

## Appendix C: Source Code Reference

This specification was derived from the decompiled `AliveDrive.Tags.dll` (tag definitions)
and `AliveDrive.Mp4.dll` (binary parsing) assemblies of the AliveDrive Apex desktop
application, version 10.1.0.3 (informational version 3.3.1), by Cosworth Electronics Ltd.

Key source files:
- `RecordingFourCCs.cs` — Box FourCC constants
- `NumericDataType.cs` — Raw data type enum
- `MP4PropertyType.cs` — Outing property type FourCCs
- `ChannelRatesBoxReader.cs` — `adcr` binary format
- `ChannelPropertiesReader.cs` — `adcp` binary format
- `CalibrationDefinitionReader.cs` — Calibration parsing
- `NumericCalibrationDefinitionReader.cs` — Gain/offset/min/max
- `DictionaryBoxReaderBase.cs` — `adud` and `adeg` binary format
- `BoxOutingPropertiesReader.cs` — `adop` property iteration
- `AliveDriveVersionInformationBoxReader.cs` — `advi` binary format
- `NumberValueReader.cs` — Raw value reading by NumericDataType
- `RateTableReader.cs` / `RateDefinitionReader.cs` / `ChannelDefinitionReader.cs` — Rate structure
- `AliveDriveContainerBoxReader.cs` — Top-level `adco` parsing
- `ChannelTags.cs` — All 144+ channel tag definitions
