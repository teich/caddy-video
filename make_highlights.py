#!/usr/bin/env python3
import argparse
import bisect
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LAT_TAG = "com.cosworth.channel.accelerometer.vehicle.x"
LON_TAG = "com.cosworth.channel.accelerometer.vehicle.y"
THROTTLE_TAG = "com.cosworth.channel.throttle.position"
BRAKE_TAG = "com.cosworth.channel.brake.position"
RPM_TAG = "com.cosworth.channel.enginespeed"
STEERING_TAG = "com.cosworth.channel.steering.angle"

RPM_TAGS = (RPM_TAG, "com.cosworth.channel.rpm")
LAT_TAGS = (LAT_TAG, "com.cosworth.channel.lateralacceleration")
LON_TAGS = (LON_TAG,)
THROTTLE_TAGS = (THROTTLE_TAG, "com.cosworth.channel.accelpos", "com.cosworth.channel.accelerator")
BRAKE_TAGS = (BRAKE_TAG, "com.cosworth.channel.brakepos")
STEERING_TAGS = (
    STEERING_TAG,
    "com.cosworth.channel.steeringangle",
    "com.cosworth.channel.steerangle",
)

CHANNEL_ID_FALLBACKS = {
    "steering": (42,),
}

TAG_ALIASES = {
    alias: RPM_TAG for alias in RPM_TAGS
}
TAG_ALIASES.update({alias: LAT_TAG for alias in LAT_TAGS})
TAG_ALIASES.update({alias: LON_TAG for alias in LON_TAGS})
TAG_ALIASES.update({alias: THROTTLE_TAG for alias in THROTTLE_TAGS})
TAG_ALIASES.update({alias: BRAKE_TAG for alias in BRAKE_TAGS})
TAG_ALIASES.update({alias: STEERING_TAG for alias in STEERING_TAGS})

NEEDED_TAGS = {LAT_TAG, LON_TAG, THROTTLE_TAG, BRAKE_TAG, RPM_TAG}

HUD_PLAY_RES_X = 1920
HUD_PLAY_RES_Y = 1080
HUD_PANEL_TOP = 850
HUD_G_CX = 220
HUD_G_CY = 962
HUD_G_OUTER_R = 108
HUD_G_INNER_R = 94
HUD_G_DOT_R = 11
HUD_G_RANGE = 1.8
HUD_RPM_X0 = 430
HUD_RPM_X1 = 1490
HUD_RPM_Y0 = 948
HUD_RPM_Y1 = 992
HUD_PEDAL_Y0 = 885
HUD_PEDAL_Y1 = 1035
HUD_THROTTLE_X0 = 1540
HUD_THROTTLE_X1 = 1582
HUD_BRAKE_X0 = 1600
HUD_BRAKE_X1 = 1642
HUD_STEER_CX = 1730
HUD_STEER_CY = 962
HUD_STEER_OUTER_R = 96
HUD_STEER_INNER_R = 82
HUD_STEER_MAX_DEG = 540.0


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


@dataclass
class RenderSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class OverlayChannel:
    channel_id: int
    tag: str
    quantity_tag: str
    points: List[Tuple[float, str]]


class TimeMapper:
    def __init__(self, telemetry_times: List[float], media_times: List[float]):
        if len(telemetry_times) != len(media_times):
            raise ValueError("time mapper arrays must have equal length")
        if len(telemetry_times) < 2:
            raise ValueError("time mapper needs at least two points")
        self.telemetry_times = telemetry_times
        self.media_times = media_times

    @staticmethod
    def _interp(x: float, xs: List[float], ys: List[float]) -> float:
        i = bisect.bisect_right(xs, x) - 1
        if i < 0:
            return ys[0]
        if i >= len(xs) - 1:
            return ys[-1]
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        if x1 == x0:
            return y0
        a = (x - x0) / (x1 - x0)
        return y0 + a * (y1 - y0)

    def telemetry_to_media(self, t: float) -> float:
        return self._interp(t, self.telemetry_times, self.media_times)

    def media_to_telemetry(self, t: float) -> float:
        return self._interp(t, self.media_times, self.telemetry_times)


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
            tag = TAG_ALIASES.get(row.get("channel_tag", ""), "")
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


def load_time_mapper(packet_summary_csv: Optional[Path]) -> Optional[TimeMapper]:
    if packet_summary_csv is None or not packet_summary_csv.exists():
        return None
    telemetry_times = []
    media_times = []
    with packet_summary_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["header_payload_len"]) <= 0:
                continue
            telemetry_times.append(float(row["header_timestamp_ticks"]) / 10_000_000.0)
            media_times.append(float(row["packet_pts_time"]))
    if len(telemetry_times) < 2:
        return None
    return TimeMapper(telemetry_times, media_times)


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


def telemetry_to_render_segments(
    telemetry_segs: List[Segment], mapper: Optional[TimeMapper], media_duration: float
) -> List[RenderSegment]:
    out = []
    for s in telemetry_segs:
        if mapper is None:
            start = float(s.start)
            end = float(s.end)
        else:
            start = mapper.telemetry_to_media(float(s.start))
            end = mapper.telemetry_to_media(float(s.end))
        start = max(0.0, min(media_duration, start))
        end = max(start + 0.001, min(media_duration, end))
        out.append(RenderSegment(start=start, end=end))
    return out


def write_plan_csv(path: Path, telemetry_segs: List[Segment], render_segs: List[RenderSegment], scores: List[float]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "segment_index",
                "telemetry_start_sec",
                "telemetry_end_sec",
                "media_start_sec",
                "media_end_sec",
                "duration_sec",
                "avg_score",
            ]
        )
        for i, (s, rs) in enumerate(zip(telemetry_segs, render_segs)):
            avg = sum(scores[s.start:s.end]) / max(1, s.duration)
            w.writerow([i, s.start, s.end, f"{rs.start:.3f}", f"{rs.end:.3f}", f"{rs.duration:.3f}", f"{avg:.4f}"])


def read_plan_csv(path: Path) -> List[RenderSegment]:
    segs = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if "media_start_sec" in row and "media_end_sec" in row:
                segs.append(RenderSegment(float(row["media_start_sec"]), float(row["media_end_sec"])))
            else:
                segs.append(RenderSegment(float(row["start_sec"]), float(row["end_sec"])))
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


OverlaySeries = Dict[str, List[Tuple[float, float]]]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q = clamp(q, 0.0, 1.0)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def ass_escape_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def trim_channel_prefix(tag: str) -> str:
    prefix = "com.cosworth.channel."
    if tag.startswith(prefix):
        return tag[len(prefix):]
    return tag


def format_number_for_overlay(value: str) -> str:
    if value == "":
        return ""
    try:
        num = float(value)
    except ValueError:
        return value
    if num.is_integer():
        return str(int(num))
    abs_num = abs(num)
    if abs_num != 0.0 and (abs_num >= 100000.0 or abs_num < 0.001):
        return f"{num:.4e}"
    return f"{num:.6g}"


def row_channel_id(row: Dict[str, str]) -> int:
    raw = row.get("channel_id", "").strip()
    if not raw:
        return -1
    return int(raw)


def load_debug_overlay_channels(decoded_csv: Path) -> List[OverlayChannel]:
    series: Dict[str, OverlayChannel] = {}
    with decoded_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("channel_tag", "").strip()
            if not tag:
                continue
            t = float(row["sample_time_sec"])
            raw = format_number_for_overlay(row.get("raw_value", ""))
            calibrated = format_number_for_overlay(row.get("calibrated_value", ""))
            label = row.get("label", "").strip()

            value = f"raw={raw} cal={calibrated}"
            if label:
                value += f" label={label}"

            ch = series.get(tag)
            if ch is None:
                ch = OverlayChannel(
                    channel_id=row_channel_id(row),
                    tag=tag,
                    quantity_tag=row.get("quantity_tag", "").strip(),
                    points=[],
                )
                series[tag] = ch
            ch.points.append((t, value))

    if not series:
        raise ValueError("overlay source has no channels")

    channels = sorted(series.values(), key=lambda c: (c.channel_id, c.tag))
    for ch in channels:
        ch.points.sort(key=lambda p: p[0])
    return channels


def load_overlay_series(decoded_csv: Path) -> OverlaySeries:
    channel_map = {
        "rpm": RPM_TAGS,
        "lat": LAT_TAGS,
        "lon": LON_TAGS,
        "throttle": THROTTLE_TAGS,
        "brake": BRAKE_TAGS,
        "steering": STEERING_TAGS,
    }
    required = {"rpm", "lat", "lon"}
    tags = {tag for aliases in channel_map.values() for tag in aliases}
    by_tag: Dict[str, List[Tuple[float, float]]] = {tag: [] for tag in tags}
    by_channel_id: Dict[int, List[Tuple[float, float]]] = {}

    with decoded_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("channel_tag", "")
            t = float(row["sample_time_sec"])
            v = float(row["calibrated_value"])
            if tag in by_tag:
                by_tag[tag].append((t, v))
            cid_txt = row.get("channel_id")
            if cid_txt:
                try:
                    cid = int(cid_txt)
                except ValueError:
                    cid = None
                if cid is not None:
                    by_channel_id.setdefault(cid, []).append((t, v))

    for points in by_tag.values():
        points.sort(key=lambda x: x[0])
    for points in by_channel_id.values():
        points.sort(key=lambda x: x[0])

    series: OverlaySeries = {}
    missing = []
    for key, aliases in channel_map.items():
        selected: List[Tuple[float, float]] = []
        for alias in aliases:
            if by_tag[alias]:
                selected = by_tag[alias]
                break
        if not selected:
            for cid in CHANNEL_ID_FALLBACKS.get(key, ()):
                if by_channel_id.get(cid):
                    selected = by_channel_id[cid]
                    break
        if not selected and key in required:
            missing.append("/".join(aliases))
        series[key] = selected if selected else [(0.0, 0.0)]
    if missing:
        raise ValueError(f"missing required channel(s) for overlay: {', '.join(missing)}")
    return series


def make_lookup(points: Sequence[Tuple[float, float]], default: float = 0.0) -> Callable[[float], float]:
    if not points:
        return lambda _t: default

    times = [t for t, _ in points]
    vals = [v for _, v in points]

    def lookup(t: float) -> float:
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return vals[0]
        if idx >= len(vals):
            return vals[-1]
        return vals[idx]

    return lookup


def make_text_lookup(points: Sequence[Tuple[float, str]], default: str = "") -> Callable[[float], str]:
    if not points:
        return lambda _t: default

    times = [t for t, _ in points]
    vals = [v for _, v in points]

    def lookup(t: float) -> str:
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return vals[0]
        if idx >= len(vals):
            return vals[-1]
        return vals[idx]

    return lookup


def ass_dialogue(layer: int, start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: {layer},{sec_to_ass(start)},{sec_to_ass(end)},{style},,0,0,0,,{text}"


def ass_xy(x: float, y: float) -> str:
    return f"{int(round(x))} {int(round(y))}"


def draw_rect_path(x0: float, y0: float, x1: float, y1: float) -> str:
    return f"m {ass_xy(x0, y0)} l {ass_xy(x1, y0)} {ass_xy(x1, y1)} {ass_xy(x0, y1)}"


def draw_circle_path(cx: float, cy: float, radius: float, segments: int = 48) -> str:
    points = []
    for i in range(max(12, segments)):
        a = (2.0 * math.pi * i) / max(12, segments)
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    first = points[0]
    rest = " ".join(ass_xy(x, y) for x, y in points[1:] + [first])
    return f"m {ass_xy(first[0], first[1])} l {rest}"


def draw_thick_line_path(x0: float, y0: float, x1: float, y1: float, thickness: float) -> str:
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        half = thickness / 2.0
        return draw_rect_path(x0 - half, y0 - half, x0 + half, y0 + half)
    px = -(dy / length) * (thickness / 2.0)
    py = (dx / length) * (thickness / 2.0)
    p1 = (x0 + px, y0 + py)
    p2 = (x1 + px, y1 + py)
    p3 = (x1 - px, y1 - py)
    p4 = (x0 - px, y0 - py)
    return f"m {ass_xy(*p1)} l {ass_xy(*p2)} {ass_xy(*p3)} {ass_xy(*p4)}"


def build_hud_static_lines(total_duration: float, rpm_max_display: float) -> List[str]:
    lines = []
    full_start = 0.0
    full_end = total_duration

    lines.append(
        ass_dialogue(
            0,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H2A1A11&\\1a&H88&}}"
            f"{draw_rect_path(0, HUD_PANEL_TOP, HUD_PLAY_RES_X, HUD_PLAY_RES_Y)}",
        )
    )

    lines.append(
        ass_dialogue(
            1,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H3C2A1C&\\1a&H48&}}"
            f"{draw_circle_path(HUD_G_CX, HUD_G_CY, HUD_G_OUTER_R, 56)}",
        )
    )
    lines.append(
        ass_dialogue(
            1,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H261910&\\1a&H70&}}"
            f"{draw_circle_path(HUD_G_CX, HUD_G_CY, HUD_G_INNER_R, 56)}",
        )
    )
    for scale in (0.35, 0.65, 1.0):
        lines.append(
            ass_dialogue(
                2,
                full_start,
                full_end,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\1a&HFF&\\3c&H9BC8E8&\\3a&H50&\\bord2\\shad0}}"
                f"{draw_circle_path(HUD_G_CX, HUD_G_CY, HUD_G_INNER_R * scale, 48)}",
            )
        )
    lines.append(
        ass_dialogue(
            2,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H8AAECF&\\1a&H70&}}"
            f"{draw_thick_line_path(HUD_G_CX - HUD_G_INNER_R, HUD_G_CY, HUD_G_CX + HUD_G_INNER_R, HUD_G_CY, 2)}",
        )
    )
    lines.append(
        ass_dialogue(
            2,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H8AAECF&\\1a&H70&}}"
            f"{draw_thick_line_path(HUD_G_CX, HUD_G_CY - HUD_G_INNER_R, HUD_G_CX, HUD_G_CY + HUD_G_INNER_R, 2)}",
        )
    )

    lines.append(
        ass_dialogue(
            1,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H2C1D11&\\1a&H50&}}"
            f"{draw_rect_path(HUD_RPM_X0, HUD_RPM_Y0, HUD_RPM_X1, HUD_RPM_Y1)}",
        )
    )
    rpm_w = HUD_RPM_X1 - HUD_RPM_X0
    for start_ratio, end_ratio, color in (
        (0.0, 0.60, "&H49AF50&"),
        (0.60, 0.85, "&H35CCEF&"),
        (0.85, 1.00, "&H4C59FF&"),
    ):
        x0 = HUD_RPM_X0 + rpm_w * start_ratio
        x1 = HUD_RPM_X0 + rpm_w * end_ratio
        lines.append(
            ass_dialogue(
                2,
                full_start,
                full_end,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{color}\\1a&HBB&}}"
                f"{draw_rect_path(x0, HUD_RPM_Y0, x1, HUD_RPM_Y1)}",
            )
        )
    lines.append(
        ass_dialogue(
            2,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\1a&HFF&\\3c&H9BC8E8&\\3a&H22&\\bord3\\shad0}}"
            f"{draw_rect_path(HUD_RPM_X0, HUD_RPM_Y0, HUD_RPM_X1, HUD_RPM_Y1)}",
        )
    )

    for x0, x1 in ((HUD_THROTTLE_X0, HUD_THROTTLE_X1), (HUD_BRAKE_X0, HUD_BRAKE_X1)):
        lines.append(
            ass_dialogue(
                1,
                full_start,
                full_end,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H2C1D11&\\1a&H50&}}"
                f"{draw_rect_path(x0, HUD_PEDAL_Y0, x1, HUD_PEDAL_Y1)}",
            )
        )
        lines.append(
            ass_dialogue(
                2,
                full_start,
                full_end,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\1a&HFF&\\3c&H9BC8E8&\\3a&H22&\\bord2\\shad0}}"
                f"{draw_rect_path(x0, HUD_PEDAL_Y0, x1, HUD_PEDAL_Y1)}",
            )
        )

    lines.append(
        ass_dialogue(
            1,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H3C2A1C&\\1a&H48&}}"
            f"{draw_circle_path(HUD_STEER_CX, HUD_STEER_CY, HUD_STEER_OUTER_R, 56)}",
        )
    )
    lines.append(
        ass_dialogue(
            1,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H261910&\\1a&H70&}}"
            f"{draw_circle_path(HUD_STEER_CX, HUD_STEER_CY, HUD_STEER_INNER_R, 56)}",
        )
    )
    for norm in (-1.0, -0.5, 0.0, 0.5, 1.0):
        ang = math.radians(-90.0 + (norm * 120.0))
        x0 = HUD_STEER_CX + math.cos(ang) * (HUD_STEER_INNER_R - 16)
        y0 = HUD_STEER_CY + math.sin(ang) * (HUD_STEER_INNER_R - 16)
        x1 = HUD_STEER_CX + math.cos(ang) * (HUD_STEER_INNER_R - 4)
        y1 = HUD_STEER_CY + math.sin(ang) * (HUD_STEER_INNER_R - 4)
        lines.append(
            ass_dialogue(
                2,
                full_start,
                full_end,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H9BC8E8&\\1a&H38&}}"
                f"{draw_thick_line_path(x0, y0, x1, y1, 3)}",
            )
        )
    lines.append(
        ass_dialogue(
            3,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&HAED9FF&\\1a&H24&}}"
            f"{draw_circle_path(HUD_STEER_CX, HUD_STEER_CY, 7, 18)}",
        )
    )

    rpm_mid = (HUD_RPM_X0 + HUD_RPM_X1) // 2
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an2\\pos({rpm_mid},{HUD_RPM_Y0 - 14})\\fs24\\bord1\\shad0}}RPM",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an8\\pos({HUD_RPM_X0},{HUD_RPM_Y1 + 26})\\fs18\\bord1\\shad0}}0",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an8\\pos({HUD_RPM_X1},{HUD_RPM_Y1 + 26})\\fs18\\bord1\\shad0}}{int(round(rpm_max_display))}",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an5\\pos({HUD_G_CX},{HUD_G_CY - HUD_G_OUTER_R - 20})\\fs21\\bord1\\shad0}}G-FORCE",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an5\\pos({(HUD_THROTTLE_X0 + HUD_THROTTLE_X1) // 2},{HUD_PEDAL_Y0 - 18})\\fs18\\bord1\\shad0}}THR",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an5\\pos({(HUD_BRAKE_X0 + HUD_BRAKE_X1) // 2},{HUD_PEDAL_Y0 - 18})\\fs18\\bord1\\shad0}}BRK",
        )
    )
    lines.append(
        ass_dialogue(
            4,
            full_start,
            full_end,
            "HudLabel",
            f"{{\\an5\\pos({HUD_STEER_CX},{HUD_STEER_CY - HUD_STEER_OUTER_R - 20})\\fs21\\bord1\\shad0}}STEER",
        )
    )
    return lines


def highlight_time_to_source_time(segments: List[RenderSegment], t_highlight: float) -> float:
    acc = 0.0
    for s in segments:
        seg_dur = float(s.duration)
        if t_highlight < acc + seg_dur:
            return s.start + (t_highlight - acc)
        acc += seg_dur
    # Clamp to end of final segment.
    return float(segments[-1].end)


def write_overlay_ass(
    decoded_csv: Path,
    segments: List[RenderSegment],
    out_ass: Path,
    fps: float,
    mapper: Optional[TimeMapper],
    full_debug_overlay: bool,
):
    if not segments:
        raise ValueError("cannot write overlay ass: no segments")
    if fps <= 0:
        raise ValueError("overlay fps must be > 0")

    total_duration = sum(s.duration for s in segments)
    frame_dt = 1.0 / fps

    if full_debug_overlay:
        channels = load_debug_overlay_channels(decoded_csv)
        if mapper is not None:
            for ch in channels:
                remapped_points = [(mapper.telemetry_to_media(t), v) for t, v in ch.points]
                remapped_points.sort(key=lambda x: x[0])
                ch.points = remapped_points

        lookups = []
        for ch in channels:
            lookups.append(
                {
                    "channel_id": ch.channel_id,
                    "tag": trim_channel_prefix(ch.tag),
                    "quantity": ch.quantity_tag,
                    "lookup": make_text_lookup(ch.points),
                }
            )

        lines = []
        t = 0.0
        while t < total_duration:
            t2 = min(t + frame_dt, total_duration)
            src_t = highlight_time_to_source_time(segments, t)

            row_lines = [f"src={src_t:.3f}s highlight={t:.3f}s channels={len(lookups)}"]
            for item in lookups:
                unit_suffix = f" [{item['quantity']}]" if item["quantity"] else ""
                row_lines.append(
                    f"{item['channel_id']:03d} {item['tag']}{unit_suffix}: {item['lookup'](src_t)}"
                )
            text = r"\N".join(ass_escape_line(line) for line in row_lines)
            lines.append(ass_dialogue(0, t, t2, "Telemetry", text))
            t = t2

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {HUD_PLAY_RES_X}
PlayResY: {HUD_PLAY_RES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Telemetry,Menlo,15,&H00FFFFFF,&H000000FF,&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,1.5,0,7,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        out_ass.write_text(header + "\n".join(lines) + "\n")
        return

    series = load_overlay_series(decoded_csv)
    if mapper is not None:
        remapped: OverlaySeries = {}
        for key, points in series.items():
            remapped_points = [(mapper.telemetry_to_media(t), v) for t, v in points]
            remapped_points.sort(key=lambda x: x[0])
            remapped[key] = remapped_points
        series = remapped
    rpm_lookup = make_lookup(series["rpm"])
    lat_lookup = make_lookup(series["lat"])
    lon_lookup = make_lookup(series["lon"])
    throttle_lookup = make_lookup(series["throttle"])
    brake_lookup = make_lookup(series["brake"])
    steering_lookup = make_lookup(series["steering"])
    rpm_samples = [max(0.0, v * 60.0 / (2.0 * math.pi)) for _, v in series["rpm"]]
    rpm_max_display = percentile(rpm_samples, 0.995)
    rpm_max_display = clamp(rpm_max_display, 5000.0, 12000.0)
    rpm_max_display = math.ceil(rpm_max_display / 250.0) * 250.0

    lines = build_hud_static_lines(total_duration, rpm_max_display)
    smoothed: Optional[Dict[str, float]] = None
    rpm_rise_alpha = 0.65
    rpm_fall_alpha = 0.95
    control_smooth_alpha = 0.35
    rpm_text_y = min(HUD_PLAY_RES_Y - 26, HUD_RPM_Y1 + 34)
    g_text_y = min(HUD_PLAY_RES_Y - 26, HUD_G_CY + HUD_G_OUTER_R - 8)
    steer_text_y = min(HUD_PLAY_RES_Y - 26, HUD_STEER_CY + HUD_STEER_OUTER_R - 8)
    pedal_text_y = min(HUD_PLAY_RES_Y - 26, HUD_PEDAL_Y1 + 24)
    t = 0.0
    while t < total_duration:
        t2 = min(t + frame_dt, total_duration)
        src_t = highlight_time_to_source_time(segments, t)

        raw = {
            "rpm": max(0.0, rpm_lookup(src_t) * 60.0 / (2.0 * math.pi)),
            "lat": lat_lookup(src_t) / 9.80665,
            "lon": lon_lookup(src_t) / 9.80665,
            "throttle": clamp(throttle_lookup(src_t), 0.0, 1.0),
            "brake": clamp(brake_lookup(src_t), 0.0, 1.0),
            "steer_deg": steering_lookup(src_t) * 180.0 / math.pi,
        }
        if smoothed is None:
            smoothed = dict(raw)
        else:
            rpm_alpha = rpm_rise_alpha if raw["rpm"] >= smoothed["rpm"] else rpm_fall_alpha
            smoothed["rpm"] += rpm_alpha * (raw["rpm"] - smoothed["rpm"])
            smoothed["lat"] += control_smooth_alpha * (raw["lat"] - smoothed["lat"])
            smoothed["lon"] += control_smooth_alpha * (raw["lon"] - smoothed["lon"])
            smoothed["throttle"] += control_smooth_alpha * (raw["throttle"] - smoothed["throttle"])
            smoothed["brake"] += control_smooth_alpha * (raw["brake"] - smoothed["brake"])
            smoothed["steer_deg"] += control_smooth_alpha * (raw["steer_deg"] - smoothed["steer_deg"])

        rpm = max(0.0, smoothed["rpm"])
        rpm_ratio = clamp(rpm / max(1.0, rpm_max_display), 0.0, 1.0)
        lat_g = clamp(smoothed["lat"], -4.0, 4.0)
        lon_g = clamp(smoothed["lon"], -4.0, 4.0)
        throttle = clamp(smoothed["throttle"], 0.0, 1.0)
        brake = clamp(smoothed["brake"], 0.0, 1.0)
        steer_deg = clamp(smoothed["steer_deg"], -HUD_STEER_MAX_DEG, HUD_STEER_MAX_DEG)

        rpm_fill_x = HUD_RPM_X0 + int(round((HUD_RPM_X1 - HUD_RPM_X0) * rpm_ratio))
        if rpm_fill_x > HUD_RPM_X0:
            fill_color = "&H49D36A&"
            if rpm_ratio >= 0.85:
                fill_color = "&H4F65FF&"
            elif rpm_ratio >= 0.60:
                fill_color = "&H3DDCF6&"
            lines.append(
                ass_dialogue(
                    5,
                    t,
                    t2,
                    "HudLabel",
                    f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{fill_color}\\1a&H28&}}"
                    f"{draw_rect_path(HUD_RPM_X0, HUD_RPM_Y0, rpm_fill_x, HUD_RPM_Y1)}",
                )
            )
            lines.append(
                ass_dialogue(
                    6,
                    t,
                    t2,
                    "HudLabel",
                    f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&HD9EEFF&\\1a&H20&}}"
                    f"{draw_rect_path(rpm_fill_x - 2, HUD_RPM_Y0 - 8, rpm_fill_x + 2, HUD_RPM_Y1 + 8)}",
                )
            )

        g_x = HUD_G_CX + clamp(lat_g / HUD_G_RANGE, -1.0, 1.0) * HUD_G_INNER_R
        g_y = HUD_G_CY - clamp(lon_g / HUD_G_RANGE, -1.0, 1.0) * HUD_G_INNER_R
        lines.append(
            ass_dialogue(
                6,
                t,
                t2,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H6AA0CF&\\1a&H48&}}"
                f"{draw_circle_path(g_x, g_y, HUD_G_DOT_R + 4, 20)}",
            )
        )
        lines.append(
            ass_dialogue(
                7,
                t,
                t2,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H95CAFF&\\1a&H14&}}"
                f"{draw_circle_path(g_x, g_y, HUD_G_DOT_R, 20)}",
            )
        )

        pedal_h = HUD_PEDAL_Y1 - HUD_PEDAL_Y0
        if throttle > 0.001:
            thr_top = HUD_PEDAL_Y1 - (pedal_h * throttle)
            lines.append(
                ass_dialogue(
                    5,
                    t,
                    t2,
                    "HudLabel",
                    f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H4ACF59&\\1a&H24&}}"
                    f"{draw_rect_path(HUD_THROTTLE_X0, thr_top, HUD_THROTTLE_X1, HUD_PEDAL_Y1)}",
                )
            )
        if brake > 0.001:
            brk_top = HUD_PEDAL_Y1 - (pedal_h * brake)
            lines.append(
                ass_dialogue(
                    5,
                    t,
                    t2,
                    "HudLabel",
                    f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H5166FF&\\1a&H24&}}"
                    f"{draw_rect_path(HUD_BRAKE_X0, brk_top, HUD_BRAKE_X1, HUD_PEDAL_Y1)}",
                )
            )

        steer_norm = steer_deg / HUD_STEER_MAX_DEG
        steer_ang = math.radians(-90.0 + (steer_norm * 120.0))
        steer_tip_x = HUD_STEER_CX + math.cos(steer_ang) * (HUD_STEER_INNER_R - 12)
        steer_tip_y = HUD_STEER_CY + math.sin(steer_ang) * (HUD_STEER_INNER_R - 12)
        lines.append(
            ass_dialogue(
                6,
                t,
                t2,
                "HudLabel",
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H9FD6FF&\\1a&H18&}}"
                f"{draw_thick_line_path(HUD_STEER_CX, HUD_STEER_CY, steer_tip_x, steer_tip_y, 8)}",
            )
        )

        g_total = math.hypot(lat_g, lon_g)
        lines.append(
            ass_dialogue(
                8,
                t,
                t2,
                "HudMono",
                f"{{\\an5\\pos({(HUD_RPM_X0 + HUD_RPM_X1) // 2},{rpm_text_y})\\fs28\\bord2\\shad0}}{rpm:5.0f} rpm",
            )
        )
        lines.append(
            ass_dialogue(
                8,
                t,
                t2,
                "HudMono",
                f"{{\\an5\\pos({HUD_G_CX},{g_text_y})\\fs20\\bord1\\shad0}}"
                f"{lat_g:+.2f} lat  {lon_g:+.2f} lon  |g| {g_total:.2f}",
            )
        )
        lines.append(
            ass_dialogue(
                8,
                t,
                t2,
                "HudMono",
                f"{{\\an5\\pos({HUD_STEER_CX},{steer_text_y})\\fs24\\bord1\\shad0}}{steer_deg:+.0f} deg",
            )
        )
        lines.append(
            ass_dialogue(
                8,
                t,
                t2,
                "HudMono",
                f"{{\\an5\\pos({(HUD_THROTTLE_X0 + HUD_THROTTLE_X1) // 2},{pedal_text_y})\\fs18\\bord1\\shad0}}{int(round(throttle * 100)):3d}%",
            )
        )
        lines.append(
            ass_dialogue(
                8,
                t,
                t2,
                "HudMono",
                f"{{\\an5\\pos({(HUD_BRAKE_X0 + HUD_BRAKE_X1) // 2},{pedal_text_y})\\fs18\\bord1\\shad0}}{int(round(brake * 100)):3d}%",
            )
        )
        t = t2

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {HUD_PLAY_RES_X}
PlayResY: {HUD_PLAY_RES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HudLabel,Arial,24,&H00D6E6F5,&H00000000,&H00201008,&H70000000,0,0,0,0,100,100,0,0,1,2,0,7,20,20,20,1
Style: HudMono,Menlo,24,&H00F0F8FF,&H00000000,&H00201008,&H70000000,0,0,0,0,100,100,0,0,1,2,0,7,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out_ass.write_text(header + "\n".join(lines) + "\n")


def ffmpeg_escape_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_filter_complex(segs: List[RenderSegment], include_audio: bool, overlay_ass: Optional[Path] = None) -> str:
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


def render_one_pass(video: Path, segs: List[RenderSegment], out_video: Path, overlay_ass: Optional[Path] = None):
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
    ap.add_argument("--packet-summary", type=Path, default=None, help="Optional packet summary CSV for telemetry/media sync")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--target-seconds", type=int, default=90)
    ap.add_argument("--context-seconds", type=int, default=4)
    ap.add_argument("--min-segment-seconds", type=int, default=20)
    ap.add_argument("--merge-gap-seconds", type=int, default=4)
    ap.add_argument("--max-total-seconds", type=int, default=120)
    ap.add_argument("--overlay-fps", type=float, default=29.97, help="Sampling rate for telemetry HUD overlay")
    ap.add_argument("--no-overlay", action="store_true", help="Disable telemetry HUD overlay")
    ap.add_argument(
        "--full-debug-overlay",
        action="store_true",
        help="Overlay all decoded telemetry fields and bypass the HUD (debug mode)",
    )
    ap.add_argument("--run", action="store_true", help="Render output video from plan")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    plan_csv = args.out_dir / f"{stem}.highlights.csv"
    ffmpeg_cmd_txt = args.out_dir / f"{stem}.highlights.ffmpeg.txt"
    overlay_ass = args.out_dir / f"{stem}.highlights.overlay.ass"
    out_video = args.out_dir / f"{stem}.highlights.mp4"

    packet_summary = args.packet_summary
    if packet_summary is None:
        candidate = args.decoded.with_name(f"{stem}.packet_summary.csv")
        if candidate.exists():
            packet_summary = candidate

    segs, scores, active_start, start_floor = build_plan(
        video=args.video,
        decoded=args.decoded,
        target_seconds=args.target_seconds,
        context_seconds=args.context_seconds,
        min_segment_seconds=args.min_segment_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        max_total_seconds=args.max_total_seconds,
    )

    mapper = load_time_mapper(packet_summary)
    media_duration = probe_duration(args.video)
    render_segs = telemetry_to_render_segments(segs, mapper, media_duration)

    write_plan_csv(plan_csv, segs, render_segs, scores)
    with_overlay = not args.no_overlay
    if with_overlay:
        write_overlay_ass(
            args.decoded,
            render_segs,
            overlay_ass,
            args.overlay_fps,
            mapper,
            full_debug_overlay=args.full_debug_overlay,
        )

    include_audio = has_audio_stream(args.video)
    fc = build_filter_complex(render_segs, include_audio, overlay_ass=overlay_ass if with_overlay else None)
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
    print(f"time_mapper={'yes' if mapper is not None else 'no'}")
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
