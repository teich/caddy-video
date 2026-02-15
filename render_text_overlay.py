#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from make_highlights import RenderSegment, write_overlay_ass


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
    ap = argparse.ArgumentParser(description="Render full-session telemetry HUD overlay")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--decoded", type=Path, required=True)
    ap.add_argument("--ass-out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    duration = probe_duration(args.video)
    args.ass_out.parent.mkdir(parents=True, exist_ok=True)
    args.video_out.parent.mkdir(parents=True, exist_ok=True)

    write_overlay_ass(
        decoded_csv=args.decoded,
        segments=[RenderSegment(start=0.0, end=duration)],
        out_ass=args.ass_out,
        fps=args.fps,
        mapper=None,
    )
    burn_ass(args.video, args.ass_out, args.video_out)

    print(f"wrote {args.ass_out}")
    print(f"wrote {args.video_out}")


if __name__ == "__main__":
    main()
