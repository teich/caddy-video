#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def process_one(
    video: Path,
    out_dir: Path,
    keep_work: bool,
    work_root: Optional[Path],
    target_seconds: int,
    max_total_seconds: int,
    full_debug_overlay: bool,
) -> Path:
    video = video.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
        tmp_base = work_root
    else:
        tmp_base = None

    with tempfile.TemporaryDirectory(prefix=f"{video.stem}_", dir=tmp_base) as td:
        work_dir = Path(td)

        extract_cmd = [
            "python3",
            str(Path(__file__).with_name("extract_alive.py")),
            str(video),
            "--out-dir",
            str(work_dir),
        ]
        run(extract_cmd)

        decoded_csv = work_dir / f"{video.stem}.decoded_updates.csv"
        packet_summary_csv = work_dir / f"{video.stem}.packet_summary.csv"
        if not decoded_csv.exists():
            raise RuntimeError(f"decoded telemetry not found: {decoded_csv}")
        if not packet_summary_csv.exists():
            raise RuntimeError(f"packet summary not found: {packet_summary_csv}")

        highlight_cmd = [
            "python3",
            str(Path(__file__).with_name("make_highlights.py")),
            "--video",
            str(video),
            "--decoded",
            str(decoded_csv),
            "--packet-summary",
            str(packet_summary_csv),
            "--out-dir",
            str(work_dir),
            "--target-seconds",
            str(target_seconds),
            "--max-total-seconds",
            str(max_total_seconds),
            "--run",
        ]
        if full_debug_overlay:
            highlight_cmd.append("--full-debug-overlay")
        run(highlight_cmd)

        rendered = work_dir / f"{video.stem}.highlights.mp4"
        if not rendered.exists():
            raise RuntimeError(f"highlight output not found: {rendered}")

        final_out = out_dir / f"{video.stem}.highlights.mp4"
        shutil.move(str(rendered), str(final_out))

        if keep_work:
            kept = out_dir / f"{video.stem}.work"
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(work_dir, kept)

        return final_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-command PDR processing: input video(s) -> overlayed highlights video(s)"
    )
    ap.add_argument("videos", nargs="+", type=Path, help="Input video file(s)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/final"),
        help="Directory for final highlight videos",
    )
    ap.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Optional root directory for temporary processing work",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate files per video under <out-dir>/<stem>.work",
    )
    ap.add_argument("--target-seconds", type=int, default=90)
    ap.add_argument("--max-total-seconds", type=int, default=120)
    ap.add_argument(
        "--full-debug-overlay",
        action="store_true",
        help="Overlay all decoded telemetry fields (debug mode)",
    )
    args = ap.parse_args()

    failures = []
    for video in args.videos:
        if not video.exists():
            failures.append((video, "input not found"))
            continue

        try:
            out_path = process_one(
                video=video,
                out_dir=args.out_dir,
                keep_work=args.keep_work,
                work_root=args.work_root,
                target_seconds=args.target_seconds,
                max_total_seconds=args.max_total_seconds,
                full_debug_overlay=args.full_debug_overlay,
            )
            print(f"OK {video} -> {out_path}")
        except Exception as exc:
            failures.append((video, str(exc)))

    if failures:
        print("\\nFailures:")
        for video, msg in failures:
            print(f"- {video}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
