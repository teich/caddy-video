# caddy-video

Telemetry-driven highlight generation for GM/Cosworth AliveDrive (`adco`) videos.

This repo currently supports a one-command workflow:
1. Take one or more raw MP4s from your PDR/SD card.
2. Extract and decode telemetry automatically.
3. Produce one final highlights MP4 per input, with telemetry HUD overlay (`RPM`, `G-meter`, `steering`, `throttle`, `brake`).

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH
- `ffmpeg` built with `libass` (required for ASS HUD overlay burn-in)

Check subtitle support:

```bash
ffmpeg -filters | rg subtitles
```

You should see a `subtitles` filter line.

## Repository layout

- `docs/ALIVEDRIVE_FORMAT.md`: reverse-engineered AliveDrive format notes.
- `docs/MACOS_APP_PORTING_BLUEPRINT.md`: implementation blueprint for rebuilding this pipeline as a native macOS/Xcode app.
- `docs/telemetry_sync_findings.md`: current telemetry/audio timing investigation notes.
- `extract_alive.py`: telemetry extractor/decoder.
- `make_highlights.py`: plan + render highlights (current main workflow).
- `render_text_overlay.py`: full-length telemetry HUD render helper (debug/inspection).
- `input/`: local source videos (ignored by git).
- `output/`: generated data/videos (ignored by git).

## Quick start (recommended)

Use an input file such as `input/ADV_0071.mp4`.

Process one file:

```bash
python3 process_pdr.py input/ADV_0071.mp4
```

Process multiple files:

```bash
python3 process_pdr.py input/*.mp4
```

Final outputs are written to:
- `output/final/<input_stem>.highlights.mp4`

Useful options:

```bash
python3 process_pdr.py input/*.mp4 \
  --out-dir output/final \
  --target-seconds 90 \
  --max-total-seconds 120 \
  --keep-work
```

`--keep-work` preserves per-file intermediates for debugging.

## Advanced tools

If you want direct control/debugging, these scripts are still available:
- `extract_alive.py`: decode telemetry only
- `make_highlights.py`: plan/render highlights from an existing decoded CSV
- `render_text_overlay.py`: render full-session telemetry HUD (debug helper)

## Tuning highlight behavior

Useful flags on `make_highlights.py`:

- `--target-seconds`: how many top-scoring seconds to seed selection from.
- `--max-total-seconds`: cap final runtime.
- `--min-segment-seconds`: minimum kept segment length.
- `--context-seconds`: padding added around selected moments.
- `--merge-gap-seconds`: merge nearby segments.
- `--no-overlay`: render highlights without telemetry HUD.
- `--overlay-fps`: subtitle sampling rate for HUD animation.

Example tighter cut:

```bash
python3 make_highlights.py \
  --video input/ADV_0071.mp4 \
  --decoded output/ADV_0071.decoded_updates.csv \
  --out-dir output \
  --target-seconds 70 \
  --max-total-seconds 90 \
  --run
```

## Notes

- Planning uses telemetry only; there is no hard-coded time floor.
- Generated files are intentionally git-ignored.
- `render_text_overlay.py` is optional and mainly useful for debugging full-session telemetry HUD rendering.
- Timing investigation status: see `docs/telemetry_sync_findings.md`.
