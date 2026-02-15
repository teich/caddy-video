# Native macOS Port Blueprint

This document is the implementation spec for rebuilding `caddy-video` as a native macOS app in a new Xcode repository.

Use this together with `docs/ALIVEDRIVE_FORMAT.md`:
- `ALIVEDRIVE_FORMAT.md` explains the reverse-engineered `adco`/AliveDrive binary format.
- This document explains the app behavior, pipeline, data contracts, and parity targets.

## 1. Product Goal

Given one or more MP4 files that contain an `adco` telemetry stream, produce one highlight MP4 per input with:
- telemetry-driven segment selection
- concatenated highlight timeline
- optional burned-in telemetry text (`RPM`, `Lat G`, `Long G`)

Current final output naming convention:
- `<input_stem>.highlights.mp4`

## 2. Current Reference Pipeline (Behavior to Match)

The existing CLI pipeline (`process_pdr.py`) runs:
1. Extract/decode telemetry (`extract_alive.py`)
2. Plan highlights + render (`make_highlights.py`)

### 2.1 Extraction stage outputs

For input `ADV_0071.mp4`, current artifacts are:
- `ADV_0071.telemetry.bin`
- `ADV_0071.telemetry_packets.csv`
- `ADV_0071.channels.csv`
- `ADV_0071.decoded_updates.csv`
- `ADV_0071.packet_summary.csv`
- `ADV_0071.meta.json`

Only some of these are required for production rendering, but all are useful for debugging and parity validation.

### 2.2 Highlight stage outputs

- `ADV_0071.highlights.csv` (segment plan)
- `ADV_0071.highlights.overlay.ass` (subtitle events)
- `ADV_0071.highlights.ffmpeg.txt` (debug command preview)
- `ADV_0071.highlights.mp4` (final render)

In native app form, keep equivalent internal artifacts (even if not exposed) to make parity testing easy.

## 3. Data Contracts You Must Preserve

## 3.1 Decoded updates table

Equivalent of `decoded_updates.csv` rows:
- `packet_index`
- `packet_pts_time`
- `packet_timestamp_ticks`
- `sample_timestamp_ticks`
- `sample_time_sec`
- `channel_id`
- `channel_tag`
- `quantity_tag`
- `raw_value`
- `calibrated_value`
- `label`

Notes:
- `.NET ticks` are used (`10,000,000 ticks == 1 second`).
- `sample_time_sec = sample_timestamp_ticks / 10_000_000`.
- `label` is used for bitfield channels.

## 3.2 Packet summary table

Equivalent columns:
- `packet_index`
- `packet_size`
- `header_timestamp_ticks`
- `header_payload_len`
- `extra_bytes`
- `decoded_payload_bytes`
- `decoded_ticks`
- `packet_pts_time`
- `packet_duration_time`

This drives telemetry->media time mapping.

## 3.3 Metadata

Equivalent of `meta.json`:
- input path
- stream index
- parsed `advi`
- packet count
- channel count
- rate definition count
- fastest interval ticks

## 4. Telemetry Decode Requirements

Read `ALIVEDRIVE_FORMAT.md` for full binary details. Minimum required flow:

1. Parse MP4 boxes and find `moov/trak/mdia/hdlr` where handler type is `adrv`.
2. In that track, locate `stsd` entry with type `adco`.
3. Parse child boxes inside `adco` sample entry payload:
- `advi` (version info)
- `adud` (unit dictionary)
- `adcr` (rate/channel raw data typing)
- `adcp` (channel properties/calibration)
4. Read data stream packets from the `adco` stream.
5. For each packet, parse header:
- timestamp ticks: 8 bytes BE
- flags: 2 bytes BE
- payload length: 4 bytes BE
6. Decode payload in tick steps using rate definitions:
- For each tick, iterate all rate defs where `sample_ticks % interval == 0`.
- Decode channel raw values using per-channel numeric type.
7. Apply calibration:
- Numeric calibration: clamp to raw min/max, then `value = raw * gain + offset`.
- Bitfield calibration: mask + lookup label, fallback default label.

### Numeric type mapping

The current decoder supports data types `1..10`:
- 1 `Int8`
- 2 `UInt8`
- 3 `Int16`
- 4 `UInt16`
- 5 `Int32`
- 6 `UInt32`
- 7 `Int64`
- 8 `UInt64`
- 9 `Float32`
- 10 `Float64`

All are big-endian in payload decoding.

## 5. Highlight Selection Algorithm (Current Logic)

Build per-second buckets from decoded telemetry.

Required channel tags:
- `com.cosworth.channel.accelerometer.vehicle.x` (`Lat` accel)
- `com.cosworth.channel.accelerometer.vehicle.y` (`Long` accel)
- `com.cosworth.channel.throttle.position`
- `com.cosworth.channel.brake.position`
- `com.cosworth.channel.enginespeed`

Per second, compute:
- `lat_abs_max = max(|lat_mps2 / 9.80665|)`
- `lon_abs_max = max(|lon_mps2 / 9.80665|)`
- `throttle_max` clamped to `[0, 1]`
- `brake_max` clamped to `[0, 1]`
- `rpm_max = max(engine_rad_per_sec * 60 / (2*pi))`

### 5.1 Active start detection

Boolean active if:
- `sqrt(lat_g^2 + lon_g^2) > 0.06` OR
- `throttle_max > 0.08` OR
- `brake_max > 0.03`

Sliding window:
- window size: `8` seconds
- needed active seconds: `4`

First window meeting threshold is `active_start`.

### 5.2 Score per second

Baselines:
- `g_base = p65` of combined G
- `t_base = p50` of throttle
- `rpm_base = p65` of rpm

For each second:
- `g = hypot(lat_abs_max, lon_abs_max)`
- `d_controls = |throttle - prev_throttle| + |brake - prev_brake|`

Score formula:
- `2.8 * max(0, g - g_base)`
- `+ 1.4 * max(0, d_controls - 0.04)`
- `+ 1.2 * brake_max`
- `+ 0.8 * max(0, throttle_max - t_base)`
- `+ 0.8 * max(0, rpm_max - rpm_base) / max(1, rpm_base)`

Cruise penalty:
- if `throttle_max < 0.15` and `brake_max < 0.05` and `d_controls < 0.03`, subtract `0.35`

Degenerate RPM fallback:
- if `rpm_base < 1` and `rpm_peak > 1`, add `0.2 * (rpm_max / rpm_peak)`

### 5.3 Segment pipeline

1. Select top `target_seconds` scored seconds at/after `active_start`.
2. Convert selected seconds to contiguous segments.
3. Expand each by `context_seconds` (default 4s) both sides.
4. Merge segments with gap <= `merge_gap_seconds` (default 4s).
5. Drop segments shorter than `min_segment_seconds` (default 20s).
6. Cap total duration to `max_total_seconds` (default 120s):
- rank segments by average score descending
- include full segments while budget allows
- if remaining budget >= 3s, keep a truncated prefix of next segment
7. If empty, fallback to a single ~60s segment near `active_start`.

## 6. Time Mapping (Telemetry -> Media)

Current logic builds piecewise-linear interpolation from `packet_summary` points:
- telemetry time = `header_timestamp_ticks / 10_000_000`
- media time = `packet_pts_time`
- skip packets with `header_payload_len <= 0`

At least 2 points are required.

Used for:
- converting telemetry-selected segments into media trim ranges
- mapping telemetry values onto highlight timeline for overlays

Important: keep this mapping modular so it can be replaced if sync research discovers a better model.

## 7. Rendering Requirements (Native Equivalent)

Current FFmpeg implementation does:
- trim each segment (`video` and optional `audio`)
- reset PTS per segment
- concat in order
- overlay telemetry text (subtitle burn-in)
- encode H.264 + AAC

Native app equivalent should match output behavior, not necessarily FFmpeg internals.

Suggested Apple stack:
- decode/read: `AVAsset`, `AVAssetReader`
- compose timeline: `AVMutableComposition`
- overlay text: `AVVideoComposition` + Core Animation layer tree or custom compositor
- export: `AVAssetExportSession` or `AVAssetWriter`

Overlay style parity target:
- top-centered text near top margin
- ~30 fps update cadence (29.97 default)
- format: `RPM: #### | Lat G: +0.00 | Long G: -0.00`

## 8. Suggested New Repo Structure (Swift)

Example target layout in new Xcode project:
- `Sources/Core/Telemetry/MP4BoxParser.swift`
- `Sources/Core/Telemetry/AliveDriveDecoder.swift`
- `Sources/Core/Telemetry/Calibration.swift`
- `Sources/Core/Highlight/HighlightScorer.swift`
- `Sources/Core/Highlight/SegmentPlanner.swift`
- `Sources/Core/Timing/TimeMapper.swift`
- `Sources/Core/Render/CompositionBuilder.swift`
- `Sources/Core/Render/OverlayRenderer.swift`
- `Sources/App/UI/*.swift`
- `Tests/CoreTests/*`
- `Fixtures/*.mp4` (small sample clips)
- `Fixtures/Expected/*.csv|json` (golden references)

## 9. MVP Milestones

1. Telemetry decode parity
- Parse `adco` metadata + decode updates.
- Match row counts and key channel values vs Python outputs.

2. Planning parity
- Reproduce segment list from same decoded input and params.
- Allow tiny float tolerance but exact segment boundaries when rounded to milliseconds.

3. Render parity
- Render same segment cuts with audio continuity.
- Add overlay text with same data and formatting.

4. App UX
- Drag/drop MP4s
- Batch processing queue
- Output folder selector
- Progress/errors per file

## 10. Validation Checklist

For one canonical input clip:
- `stream_index` matches
- `channel_count` matches
- `rate_definition_count` matches
- decoded row count matches
- required channel tags present
- `active_start` matches
- segment count and boundaries match
- total highlight duration within <= 0.05s
- overlay value spot checks at multiple timestamps

Also validate known sync behavior from `docs/telemetry_sync_findings.md`:
- current observed lag is around `0.8s`
- no hardcoded offset unless multi-file validation proves a stable correction

## 11. Non-Goals (for initial port)

- Changing highlight scoring policy
- Fixing telemetry/audio sync model
- Supporting non-`adco` telemetry formats
- Real-time playback overlays

Ship functional parity first, then iterate.
