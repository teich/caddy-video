#!/usr/bin/env python3
import argparse
import bisect
import csv
import subprocess
from pathlib import Path

RPM_TAG = "com.cosworth.channel.enginespeed"
LAT_TAG = "com.cosworth.channel.accelerometer.vehicle.x"
LON_TAG = "com.cosworth.channel.accelerometer.vehicle.y"


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


def load_series(decoded_csv: Path):
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
            raise ValueError(f"missing required channel: {tag}")
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


def write_ass(out_ass: Path, duration_sec: float, fps: float, series):
    rpm_lookup = make_lookup(series[RPM_TAG])
    lat_lookup = make_lookup(series[LAT_TAG])
    lon_lookup = make_lookup(series[LON_TAG])

    frame_dt = 1.0 / fps
    lines = []
    t = 0.0
    while t < duration_sec:
        t2 = min(t + frame_dt, duration_sec)
        rpm_rad_s = rpm_lookup(t)
        rpm = rpm_rad_s * 60.0 / (2.0 * 3.141592653589793)
        lat_g = lat_lookup(t) / 9.80665
        lon_g = lon_lookup(t) / 9.80665
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


def probe_duration(input_video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def burn_ass(input_video: Path, ass_file: Path, output_video: Path):
    escaped_ass = str(ass_file).replace("\\", "\\\\").replace(":", "\\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        f"subtitles=filename='{escaped_ass}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Render simple telemetry text overlay")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--decoded", type=Path, required=True)
    ap.add_argument("--ass-out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    series = load_series(args.decoded)
    duration = probe_duration(args.video)
    args.ass_out.parent.mkdir(parents=True, exist_ok=True)
    args.video_out.parent.mkdir(parents=True, exist_ok=True)

    write_ass(args.ass_out, duration, args.fps, series)
    burn_ass(args.video, args.ass_out, args.video_out)

    print(f"wrote {args.ass_out}")
    print(f"wrote {args.video_out}")


if __name__ == "__main__":
    main()
