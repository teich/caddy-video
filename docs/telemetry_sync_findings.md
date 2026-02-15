# Telemetry Sync Findings

## Scope

This document records the current telemetry/audio overlay timing investigation.

- Analysis status: observational only
- Offset correction implementation: none
- Runtime behavior changes: none

## Baseline File

- Video: `/Users/oren/src/caddy-video/output/final/ADV_0071.highlights.mp4`
- Video FPS: `30000/1001` (about `29.97` fps)

## Measured Points

- `650 -> 676` (`audio_peak_frame -> text_peak_frame`)
  - Delta: `26` frames
  - Delta time: `0.868s`
- `3179 -> 3202` (`audio_peak_frame -> text_peak_frame`)
  - Delta: `23` frames
  - Delta time: `0.767s`

Derived summary:

- Mean lag: `24.5` frames
- Mean lag time: `0.817s`

## Current Interpretation

- A sub-second lag is confirmed.
- The two measured points vary slightly (`26` vs `23` frames), which may indicate small drift, but this is not conclusive.
- The root cause is not yet proven.

## Explicit Non-Conclusions

- Do not hardcode a fixed offset yet.
- Do not change telemetry/audio mapping logic yet.

## Multi-File Validation Protocol

For each additional file:

1. Collect at least 3 manual pairs across the timeline (early, middle, late):
   - `audio_peak_frame,text_peak_frame`
2. Record video fps/timebase.
3. Convert frame deltas to seconds.
4. Compute:
   - per-file mean lag
   - intra-file lag trend (early vs late)
5. Compare lag patterns across files.

## Decision Gate for Code Changes

Proceed to implementation only if cross-file results show a stable pattern that supports a non-hardcoded mapping fix.
