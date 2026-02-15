# caddy-video

Telemetry-driven highlight generation for GM/Cosworth AliveDrive (`adco`) videos.

This repo currently does three things:
1. Extract and decode embedded telemetry from an MP4.
2. Plan highlight segments from telemetry signal intensity.
3. Render one final highlights MP4 with telemetry text overlay (`RPM`, `Lat G`, `Long G`).

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH
- `ffmpeg` built with `libass` (required for text overlay burn-in)

Check subtitle support:

```bash
ffmpeg -filters | rg subtitles
```

You should see a `subtitles` filter line.

## Repository layout

- `ALIVEDRIVE_FORMAT.md`: reverse-engineered AliveDrive format notes.
- `extract_alive.py`: telemetry extractor/decoder.
- `make_highlights.py`: plan + render highlights (current main workflow).
- `render_text_overlay.py`: full-length overlay render helper (debug/inspection).
- `input/`: local source videos (ignored by git).
- `output/`: generated data/videos (ignored by git).

## Quick start

Use an input file such as `input/ADV_0071.mp4`.

### 1. Decode telemetry

```bash
python3 extract_alive.py input/ADV_0071.mp4 --out-dir output
```

Generates (among others):
- `output/ADV_0071.decoded_updates.csv`
- `output/ADV_0071.channels.csv`

### 2. Build a highlight plan (no render)

```bash
python3 make_highlights.py \
  --video input/ADV_0071.mp4 \
  --decoded output/ADV_0071.decoded_updates.csv \
  --out-dir output
```

This writes:
- `output/ADV_0071.highlights.csv` (planned segments)
- `output/ADV_0071.highlights.ffmpeg.txt` (render command preview)
- `output/ADV_0071.highlights.overlay.ass` (overlay subtitles for planned cut)

### 3. Render final highlight video (single pass)

```bash
python3 make_highlights.py \
  --video input/ADV_0071.mp4 \
  --decoded output/ADV_0071.decoded_updates.csv \
  --out-dir output \
  --run
```

Final output:
- `output/ADV_0071.highlights.mp4`

## Tuning highlight behavior

Useful flags on `make_highlights.py`:

- `--target-seconds`: how many top-scoring seconds to seed selection from.
- `--max-total-seconds`: cap final runtime.
- `--min-segment-seconds`: minimum kept segment length.
- `--context-seconds`: padding added around selected moments.
- `--merge-gap-seconds`: merge nearby segments.
- `--no-overlay`: render highlights without text overlay.
- `--overlay-fps`: subtitle sampling rate for overlay text.

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
- `render_text_overlay.py` is optional and mainly useful for debugging full-session overlays.
