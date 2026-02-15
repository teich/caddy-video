#!/usr/bin/env python3
import argparse
import bisect
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LAT_TAG = "com.cosworth.channel.accelerometer.vehicle.x"
LON_TAG = "com.cosworth.channel.accelerometer.vehicle.y"
THROTTLE_TAG = "com.cosworth.channel.throttle.position"
BRAKE_TAG = "com.cosworth.channel.brake.position"
RPM_TAG = "com.cosworth.channel.enginespeed"
NEEDED_TAGS = {LAT_TAG, LON_TAG, THROTTLE_TAG, BRAKE_TAG, RPM_TAG}


@dataclass
class Bucket:
    lat_abs_max: float = 0.0
    lon_abs_max: float = 0.0
    throttle_max: float = 0.0
    brake_max: float = 0.0
    rpm_max: float = 0.0


@dataclass
class Segment:
    start: int
    end: int  # exclusive

    @property
    def duration(self) -> int:
        return self.end - self.start


def run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout


def probe_duration(video: Path) -> float:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    ).strip()
    return float(out)


def has_audio_stream(video: Path) -> bool:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ]
    )
    return bool(out.strip())


def build_buckets(decoded_csv: Path, total_secs: int):
    buckets = [Bucket() for _ in range(total_secs)]
    rpm_peak = 0.0
    with decoded_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tag = row.get("channel_tag", "")
            if tag not in NEEDED_TAGS:
                continue
            sec = int(float(row["sample_time_sec"]))
            if sec < 0 or sec >= total_secs:
                continue
            v = float(row["calibrated_value"])
            b = buckets[sec]
            if tag == LAT_TAG:
                b.lat_abs_max = max(b.lat_abs_max, abs(v / 9.80665))
            elif tag == LON_TAG:
                b.lon_abs_max = max(b.lon_abs_max, abs(v / 9.80665))
            elif tag == THROTTLE_TAG:
                b.throttle_max = max(b.throttle_max, max(0.0, min(1.0, v)))
            elif tag == BRAKE_TAG:
                b.brake_max = max(b.brake_max, max(0.0, min(1.0, v)))
            elif tag == RPM_TAG:
                rpm = max(0.0, v * 60.0 / (2.0 * math.pi))
                b.rpm_max = max(b.rpm_max, rpm)
                rpm_peak = max(rpm_peak, rpm)
    return buckets, rpm_peak


def detect_active_start(buckets: List[Bucket]) -> int:
    active = []
    for b in buckets:
        g = math.hypot(b.lat_abs_max, b.lon_abs_max)
        active.append(g > 0.06 or b.throttle_max > 0.08 or b.brake_max > 0.03)

    window = 8
    needed = 4
    run_sum = 0
    for i, v in enumerate(active):
        run_sum += 1 if v else 0
        if i >= window:
            run_sum -= 1 if active[i - window] else 0
        if i >= window - 1 and run_sum >= needed:
            return i - window + 1
    return 0


def compute_scores(buckets: List[Bucket], rpm_peak: float):
    scores = [0.0] * len(buckets)
    g_vals = [math.hypot(b.lat_abs_max, b.lon_abs_max) for b in buckets]
    t_vals = [b.throttle_max for b in buckets]
    rpm_vals = [b.rpm_max for b in buckets]
    g_base = sorted(g_vals)[int(0.65 * len(g_vals))]
    t_base = sorted(t_vals)[int(0.50 * len(t_vals))]
    rpm_base = sorted(rpm_vals)[int(0.65 * len(rpm_vals))]

    prev_t = 0.0
    prev_b = 0.0
    for i, b in enumerate(buckets):
        g = g_vals[i]
        d_controls = abs(b.throttle_max - prev_t) + abs(b.brake_max - prev_b)
        prev_t = b.throttle_max
        prev_b = b.brake_max

        score = (
            2.8 * max(0.0, g - g_base)
            + 1.4 * max(0.0, d_controls - 0.04)
            + 1.2 * b.brake_max
            + 0.8 * max(0.0, b.throttle_max - t_base)
            + 0.8 * max(0.0, b.rpm_max - rpm_base) / max(1.0, rpm_base)
        )
        # Penalize steady-state cruising to bias toward transitions and events.
        if b.throttle_max < 0.15 and b.brake_max < 0.05 and d_controls < 0.03:
            score -= 0.35
        # Use rpm_peak only as safety fallback if baseline is degenerate.
        if rpm_base < 1.0 and rpm_peak > 1.0:
            score += 0.2 * (b.rpm_max / rpm_peak)
        scores[i] = score
    return scores


def pick_seconds(scores: List[float], start_sec: int, target_seconds: int):
    selected = [False] * len(scores)
    if start_sec >= len(scores):
        return selected
    indexed = [(s, i) for i, s in enumerate(scores[start_sec:], start=start_sec)]
    indexed.sort(reverse=True)
    for _, i in indexed[: max(1, target_seconds)]:
        selected[i] = True
    return selected


def to_segments(selected: List[bool]) -> List[Segment]:
    out = []
    i = 0
    while i < len(selected):
        if not selected[i]:
            i += 1
            continue
        j = i + 1
        while j < len(selected) and selected[j]:
            j += 1
        out.append(Segment(i, j))
        i = j
    return out


def merge_segments(segs: List[Segment], gap: int) -> List[Segment]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)
    out = [segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s.start <= last.end + gap:
            out[-1] = Segment(last.start, max(last.end, s.end))
        else:
            out.append(s)
    return out


def expand_segments(segs: List[Segment], total_secs: int, context: int) -> List[Segment]:
    expanded = [Segment(max(0, s.start - context), min(total_secs, s.end + context)) for s in segs]
    return merge_segments(expanded, gap=0)


def drop_short_segments(segs: List[Segment], min_len: int) -> List[Segment]:
    return [s for s in segs if s.duration >= min_len]


def cap_total_duration_by_score(segs: List[Segment], scores: List[float], max_seconds: int) -> List[Segment]:
    if max_seconds <= 0:
        return segs
    ranked = []
    for s in segs:
        avg = sum(scores[s.start:s.end]) / max(1, s.duration)
        ranked.append((avg, s))
    ranked.sort(key=lambda x: x[0], reverse=True)

    out = []
    used = 0
    for _, s in ranked:
        if used >= max_seconds:
            break
        if used + s.duration <= max_seconds:
            out.append(s)
            used += s.duration
        else:
            rem = max_seconds - used
            if rem >= 3:
                out.append(Segment(s.start, s.start + rem))
            break
    return sorted(out, key=lambda s: s.start)


def write_plan_csv(path: Path, segs: List[Segment], scores: List[float]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_index", "start_sec", "end_sec", "duration_sec", "avg_score"])
        for i, s in enumerate(segs):
            avg = sum(scores[s.start:s.end]) / max(1, s.duration)
            w.writerow([i, s.start, s.end, s.duration, f"{avg:.4f}"])


def read_plan_csv(path: Path) -> List[Segment]:
    segs = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            segs.append(Segment(int(row["start_sec"]), int(row["end_sec"])))
    return segs


def sec_to_ass(ts: float) -> str:
    h = int(ts // 3600)
    ts -= h * 3600
    m = int(ts // 60)
    ts -= m * 60
    s = int(ts)
    cs = int(round((ts - s) * 100))
    if cs >= 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def load_overlay_series(decoded_csv: Path):
    series = {RPM_TAG: [], LAT_TAG: [], LON_TAG: []}
    with decoded_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row["channel_tag"]
            if tag not in series:
                continue
            t = float(row["sample_time_sec"])
            v = float(row["calibrated_value"])
            series[tag].append((t, v))

    for tag in series:
        series[tag].sort(key=lambda x: x[0])
        if not series[tag]:
            raise ValueError(f"missing required channel for overlay: {tag}")
    return series


def make_lookup(points):
    times = [t for t, _ in points]
    vals = [v for _, v in points]

    def lookup(t: float) -> float:
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return vals[0]
        return vals[idx]

    return lookup


def highlight_time_to_source_time(segments: List[Segment], t_highlight: float) -> float:
    acc = 0.0
    for s in segments:
        seg_dur = float(s.duration)
        if t_highlight < acc + seg_dur:
            return s.start + (t_highlight - acc)
        acc += seg_dur
    # Clamp to end of final segment.
    return float(segments[-1].end)


def write_overlay_ass(decoded_csv: Path, segments: List[Segment], out_ass: Path, fps: float):
    if not segments:
        raise ValueError("cannot write overlay ass: no segments")

    series = load_overlay_series(decoded_csv)
    rpm_lookup = make_lookup(series[RPM_TAG])
    lat_lookup = make_lookup(series[LAT_TAG])
    lon_lookup = make_lookup(series[LON_TAG])

    total_duration = sum(s.duration for s in segments)
    frame_dt = 1.0 / fps

    lines = []
    t = 0.0
    while t < total_duration:
        t2 = min(t + frame_dt, total_duration)
        src_t = highlight_time_to_source_time(segments, t)
        rpm_rad_s = rpm_lookup(src_t)
        rpm = rpm_rad_s * 60.0 / (2.0 * math.pi)
        lat_g = lat_lookup(src_t) / 9.80665
        lon_g = lon_lookup(src_t) / 9.80665
        text = f"RPM: {rpm:6.0f} | Lat G: {lat_g:+.2f} | Long G: {lon_g:+.2f}"
        lines.append(
            "Dialogue: 0,"
            f"{sec_to_ass(t)},"
            f"{sec_to_ass(t2)},"
            "Telemetry,,0,0,0,,"
            f"{text}"
        )
        t = t2

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Telemetry,Menlo,38,&H00FFFFFF,&H000000FF,&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out_ass.write_text(header + "\n".join(lines) + "\n")


def ffmpeg_escape_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_filter_complex(segs: List[Segment], include_audio: bool, overlay_ass: Optional[Path] = None) -> str:
    parts = []
    for i, s in enumerate(segs):
        parts.append(f"[0:v]trim=start={s.start}:end={s.end},setpts=PTS-STARTPTS[v{i}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={s.start}:end={s.end},asetpts=PTS-STARTPTS[a{i}]")

    if include_audio:
        concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(len(segs))])
        parts.append(f"{concat_inputs}concat=n={len(segs)}:v=1:a=1[vcat][a]")
    else:
        concat_inputs = "".join([f"[v{i}]" for i in range(len(segs))])
        parts.append(f"{concat_inputs}concat=n={len(segs)}:v=1:a=0[vcat]")
    if overlay_ass is not None:
        esc = ffmpeg_escape_path(overlay_ass)
        parts.append(f"[vcat]subtitles=filename='{esc}'[v]")
    else:
        parts.append("[vcat]null[v]")
    return ";".join(parts)


def render_one_pass(video: Path, segs: List[Segment], out_video: Path, overlay_ass: Optional[Path] = None):
    include_audio = has_audio_stream(video)
    fc = build_filter_complex(segs, include_audio, overlay_ass=overlay_ass)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
    ]
    if include_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    cmd += [str(out_video)]
    run(cmd)


def build_plan(
    video: Path,
    decoded: Path,
    target_seconds: int,
    context_seconds: int,
    min_segment_seconds: int,
    merge_gap_seconds: int,
    max_total_seconds: int,
):
    total_secs = max(1, math.ceil(probe_duration(video)))
    buckets, rpm_peak = build_buckets(decoded, total_secs)
    active_start = detect_active_start(buckets)
    start_floor = active_start

    scores = compute_scores(buckets, rpm_peak)
    selected = pick_seconds(scores, start_floor, target_seconds)
    segs = to_segments(selected)
    segs = expand_segments(segs, total_secs, context_seconds)
    segs = merge_segments(segs, merge_gap_seconds)
    segs = drop_short_segments(segs, min_segment_seconds)
    segs = cap_total_duration_by_score(segs, scores, max_total_seconds)

    if not segs:
        s = min(start_floor, max(0, total_secs - 60))
        segs = [Segment(s, min(total_secs, s + 60))]

    return segs, scores, active_start, start_floor


def main():
    ap = argparse.ArgumentParser(description="Telemetry-driven highlight plan/render")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--decoded", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--target-seconds", type=int, default=90)
    ap.add_argument("--context-seconds", type=int, default=4)
    ap.add_argument("--min-segment-seconds", type=int, default=20)
    ap.add_argument("--merge-gap-seconds", type=int, default=4)
    ap.add_argument("--max-total-seconds", type=int, default=120)
    ap.add_argument("--overlay-fps", type=float, default=29.97)
    ap.add_argument("--no-overlay", action="store_true", help="Disable telemetry text overlay")
    ap.add_argument("--run", action="store_true", help="Render output video from plan")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    plan_csv = args.out_dir / f"{stem}.highlights.csv"
    ffmpeg_cmd_txt = args.out_dir / f"{stem}.highlights.ffmpeg.txt"
    overlay_ass = args.out_dir / f"{stem}.highlights.overlay.ass"
    out_video = args.out_dir / f"{stem}.highlights.mp4"

    segs, scores, active_start, start_floor = build_plan(
        video=args.video,
        decoded=args.decoded,
        target_seconds=args.target_seconds,
        context_seconds=args.context_seconds,
        min_segment_seconds=args.min_segment_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        max_total_seconds=args.max_total_seconds,
    )

    write_plan_csv(plan_csv, segs, scores)
    with_overlay = not args.no_overlay
    if with_overlay:
        write_overlay_ass(args.decoded, segs, overlay_ass, args.overlay_fps)

    include_audio = has_audio_stream(args.video)
    fc = build_filter_complex(segs, include_audio, overlay_ass=overlay_ass if with_overlay else None)
    cmd_preview = [
        "ffmpeg",
        "-y",
        "-i",
        str(args.video),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
    ]
    if include_audio:
        cmd_preview += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    cmd_preview += [str(out_video)]
    ffmpeg_cmd_txt.write_text(" ".join(cmd_preview) + "\n")

    print(f"active_start_sec={active_start}")
    print(f"start_floor_sec={start_floor}")
    print(f"segments={len(segs)}")
    print(f"total_highlight_seconds={sum(s.duration for s in segs)}")
    print(f"wrote {plan_csv}")
    if with_overlay:
        print(f"wrote {overlay_ass}")
    print(f"wrote {ffmpeg_cmd_txt}")

    if args.run:
        segs_from_plan = read_plan_csv(plan_csv)
        render_one_pass(args.video, segs_from_plan, out_video, overlay_ass=overlay_ass if with_overlay else None)
        print(f"wrote {out_video}")
    else:
        print("plan-only mode: review plan CSV, then rerun with --run")


if __name__ == "__main__":
    main()
