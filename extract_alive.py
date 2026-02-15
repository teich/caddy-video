#!/usr/bin/env python3
import argparse
import csv
import json
import mmap
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

NUMERIC_SIZES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 8, 8: 8, 9: 4, 10: 8}


@dataclass
class ChannelDef:
    channel_id: int
    raw_type: int


@dataclass
class RateDef:
    interval_ticks: int
    channels: List[ChannelDef]


@dataclass
class NumericCalibration:
    gain: float
    offset: float
    minimum: Union[int, float]
    maximum: Union[int, float]


@dataclass
class BitFieldEntry:
    entry_tag: str
    entry_value: Union[int, float]


@dataclass
class BitFieldDef:
    tag: str
    mask: Union[int, float]
    default_tag: str
    default_value: Union[int, float]
    entries: List[BitFieldEntry]


@dataclass
class ChannelProp:
    channel_id: int
    tag: str
    quantity_id: int
    quantity_tag: str
    calibration_type: int
    calibrated_data_type: int
    numeric_calibration: Optional[NumericCalibration]
    bitfields: List[BitFieldDef]


def be_u32(data: memoryview, pos: int) -> int:
    return int.from_bytes(data[pos:pos + 4], "big")


def be_u64(data: memoryview, pos: int) -> int:
    return int.from_bytes(data[pos:pos + 8], "big")


def read_c_string(data: memoryview, pos: int, end: int) -> Tuple[str, int]:
    i = pos
    while i < end and data[i] != 0:
        i += 1
    if i >= end:
        raise ValueError("unterminated string")
    raw = bytes(data[pos:i])
    return raw.decode("utf-8"), i + 1


def iter_boxes(data: memoryview, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size = be_u32(data, pos)
        fourcc = bytes(data[pos + 4:pos + 8])
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = be_u64(data, pos + 8)
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield pos, size, fourcc, header
        pos += size


def find_child_box(data: memoryview, start: int, end: int, fourcc: bytes) -> Optional[Tuple[int, int, int]]:
    for pos, size, typ, hdr in iter_boxes(data, start, end):
        if typ == fourcc:
            return pos, size, hdr
    return None


def parse_adco_from_mp4(mp4_path: Path):
    with mp4_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = memoryview(mm)
        try:
            moov = find_child_box(data, 0, len(data), b"moov")
            if not moov:
                raise ValueError("moov not found")
            moov_pos, moov_size, moov_hdr = moov
            moov_start = moov_pos + moov_hdr
            moov_end = moov_pos + moov_size

            adco_entry = None
            for trak_pos, trak_size, trak_type, trak_hdr in iter_boxes(data, moov_start, moov_end):
                if trak_type != b"trak":
                    continue
                trak_start = trak_pos + trak_hdr
                trak_end = trak_pos + trak_size

                mdia = find_child_box(data, trak_start, trak_end, b"mdia")
                if not mdia:
                    continue
                mdia_pos, mdia_size, mdia_hdr = mdia
                mdia_start = mdia_pos + mdia_hdr
                mdia_end = mdia_pos + mdia_size

                hdlr = find_child_box(data, mdia_start, mdia_end, b"hdlr")
                if not hdlr:
                    continue
                hdlr_pos, hdlr_size, hdlr_hdr = hdlr
                hdlr_payload = hdlr_pos + hdlr_hdr
                handler_type = bytes(data[hdlr_payload + 8:hdlr_payload + 12])
                if handler_type != b"adrv":
                    continue

                minf = find_child_box(data, mdia_start, mdia_end, b"minf")
                if not minf:
                    continue
                minf_pos, minf_size, minf_hdr = minf

                stbl = find_child_box(data, minf_pos + minf_hdr, minf_pos + minf_size, b"stbl")
                if not stbl:
                    continue
                stbl_pos, stbl_size, stbl_hdr = stbl

                stsd = find_child_box(data, stbl_pos + stbl_hdr, stbl_pos + stbl_size, b"stsd")
                if not stsd:
                    continue
                stsd_pos, stsd_size, stsd_hdr = stsd
                p = stsd_pos + stsd_hdr
                stsd_end = stsd_pos + stsd_size
                entry_count = be_u32(data, p + 4)
                p += 8
                for _ in range(entry_count):
                    entry_size = be_u32(data, p)
                    entry_type = bytes(data[p + 4:p + 8])
                    if entry_type == b"adco":
                        adco_entry = (p, entry_size)
                        break
                    p += entry_size
                if adco_entry:
                    break

            if not adco_entry:
                raise ValueError("adco sample entry not found")

            adco_pos, adco_size = adco_entry
            adco_data_start = adco_pos + 16
            adco_data_end = adco_pos + adco_size
            children = {}
            for pos, size, typ, hdr in iter_boxes(data, adco_data_start, adco_data_end):
                children[typ.decode("ascii")] = (pos + hdr, pos + size)

            return children, data
        except Exception:
            mm.close()
            raise


def decode_number(data: memoryview, pos: int, dtype: int) -> Tuple[Union[int, float], int]:
    if dtype == 1:
        return struct.unpack_from(">b", data, pos)[0], pos + 1
    if dtype == 2:
        return data[pos], pos + 1
    if dtype == 3:
        return struct.unpack_from(">h", data, pos)[0], pos + 2
    if dtype == 4:
        return struct.unpack_from(">H", data, pos)[0], pos + 2
    if dtype == 5:
        return struct.unpack_from(">i", data, pos)[0], pos + 4
    if dtype == 6:
        return struct.unpack_from(">I", data, pos)[0], pos + 4
    if dtype == 7:
        return struct.unpack_from(">q", data, pos)[0], pos + 8
    if dtype == 8:
        return struct.unpack_from(">Q", data, pos)[0], pos + 8
    if dtype == 9:
        return struct.unpack_from(">f", data, pos)[0], pos + 4
    if dtype == 10:
        return struct.unpack_from(">d", data, pos)[0], pos + 8
    raise ValueError(f"unsupported NumericDataType: {dtype}")


def parse_adud(data: memoryview, start: int, end: int) -> Dict[int, str]:
    units = {}
    p = start
    while p < end:
        if p + 2 > end:
            break
        unit_id = int.from_bytes(data[p:p + 2], "big")
        p += 2
        unit_tag, p = read_c_string(data, p, end)
        units[unit_id] = unit_tag
    return units


def parse_adcr(data: memoryview, start: int, end: int) -> Tuple[List[RateDef], Dict[int, int]]:
    p = start
    rate_defs = []
    channel_raw_types: Dict[int, int] = {}
    num_rate_tables = data[p]
    p += 1
    for _ in range(num_rate_tables):
        _table_id = data[p]
        p += 1
        num_defs = data[p]
        p += 1
        for _ in range(num_defs):
            interval = be_u64(data, p)
            p += 8
            num_channels = int.from_bytes(data[p:p + 2], "big")
            p += 2
            channels = []
            for _ in range(num_channels):
                channel_id = int.from_bytes(data[p:p + 2], "big")
                raw_type = data[p + 2]
                p += 3
                channels.append(ChannelDef(channel_id=channel_id, raw_type=raw_type))
                prior = channel_raw_types.get(channel_id)
                if prior is None:
                    channel_raw_types[channel_id] = raw_type
                elif prior != raw_type:
                    raise ValueError(f"channel {channel_id} has inconsistent raw types: {prior} vs {raw_type}")
            rate_defs.append(RateDef(interval_ticks=interval, channels=channels))
    return rate_defs, channel_raw_types


def parse_adcp(data: memoryview, start: int, end: int, units: Dict[int, str], channel_raw_types: Dict[int, int]) -> Dict[int, ChannelProp]:
    props: Dict[int, ChannelProp] = {}
    p = start
    while p < end:
        if p + 2 > end:
            break
        channel_id = int.from_bytes(data[p:p + 2], "big")
        p += 2
        tag, p = read_c_string(data, p, end)
        quantity_id = int.from_bytes(data[p:p + 2], "big")
        p += 2
        calibration_type = data[p]
        p += 1
        calibrated_data_type = data[p]
        p += 1

        numeric_cal = None
        bitfields: List[BitFieldDef] = []

        if calibration_type == 1:
            gain = struct.unpack_from(">d", data, p)[0]
            offset = struct.unpack_from(">d", data, p + 8)[0]
            p += 16
            raw_type = channel_raw_types.get(channel_id, calibrated_data_type)
            minimum, p = decode_number(data, p, raw_type)
            maximum, p = decode_number(data, p, raw_type)
            numeric_cal = NumericCalibration(gain=gain, offset=offset, minimum=minimum, maximum=maximum)
        elif calibration_type == 2:
            num_bitfields = data[p]
            p += 1
            for _ in range(num_bitfields):
                bf_tag, p = read_c_string(data, p, end)
                mask, p = decode_number(data, p, calibrated_data_type)
                default_tag, p = read_c_string(data, p, end)
                default_value, p = decode_number(data, p, calibrated_data_type)
                num_entries = data[p]
                p += 1
                entries = []
                for _ in range(num_entries):
                    entry_tag, p = read_c_string(data, p, end)
                    entry_value, p = decode_number(data, p, calibrated_data_type)
                    entries.append(BitFieldEntry(entry_tag=entry_tag, entry_value=entry_value))
                bitfields.append(
                    BitFieldDef(
                        tag=bf_tag,
                        mask=mask,
                        default_tag=default_tag,
                        default_value=default_value,
                        entries=entries,
                    )
                )
        else:
            raise ValueError(f"unknown calibration type {calibration_type} for channel {channel_id}")

        props[channel_id] = ChannelProp(
            channel_id=channel_id,
            tag=tag,
            quantity_id=quantity_id,
            quantity_tag=units.get(quantity_id, ""),
            calibration_type=calibration_type,
            calibrated_data_type=calibrated_data_type,
            numeric_calibration=numeric_cal,
            bitfields=bitfields,
        )
    return props


def parse_advi(data: memoryview, start: int, end: int):
    p = start
    fmt_major = int.from_bytes(data[p:p + 2], "big")
    fmt_minor = int.from_bytes(data[p + 2:p + 4], "big")
    p += 4

    def read_ver3(off: int):
        major = int.from_bytes(data[off:off + 2], "big")
        minor = int.from_bytes(data[off + 2:off + 4], "big")
        build = int.from_bytes(data[off + 4:off + 6], "big")
        return f"{major}.{minor}.{build}"

    mmp = read_ver3(p)
    p += 6
    vip = read_ver3(p)
    p += 6
    app = read_ver3(p)
    p += 6
    source_tag, _ = read_c_string(data, p, end)
    return {
        "format_version": f"{fmt_major}.{fmt_minor}",
        "mmp_source_version": mmp,
        "vip_source_version": vip,
        "app_source_version": app,
        "source_tag": source_tag,
    }


def run_cmd(cmd: List[str], cwd: Path):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout


def find_adco_stream_index(mp4_path: Path, cwd: Path) -> int:
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(mp4_path),
        ],
        cwd,
    )
    doc = json.loads(out)
    for s in doc.get("streams", []):
        if s.get("codec_type") == "data" and s.get("codec_tag_string") == "adco":
            return int(s["index"])
    raise RuntimeError("adco data stream not found")


def demux_telemetry(mp4_path: Path, stream_index: int, out_bin: Path, cwd: Path):
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-map",
            f"0:{stream_index}",
            "-c",
            "copy",
            "-f",
            "data",
            str(out_bin),
        ],
        cwd,
    )


def export_packet_csv(mp4_path: Path, stream_index: int, packet_csv: Path, cwd: Path):
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,duration_time,size,flags",
            "-of",
            "csv=p=0",
            str(mp4_path),
        ],
        cwd,
    )
    packet_csv.write_text(out)


def parse_packets(packet_csv: Path):
    packets = []
    with packet_csv.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            packets.append(
                {
                    "pts_time": float(row[0]),
                    "dts_time": float(row[1]),
                    "duration_time": float(row[2]),
                    "size": int(row[3]),
                    "flags": row[4] if len(row) > 4 else "",
                }
            )
    return packets


def apply_calibration(raw_value: Union[int, float], channel: ChannelProp):
    if channel.calibration_type == 1 and channel.numeric_calibration is not None:
        cal = channel.numeric_calibration
        clamped = raw_value
        if clamped < cal.minimum:
            clamped = cal.minimum
        if clamped > cal.maximum:
            clamped = cal.maximum
        value = clamped * cal.gain + cal.offset
        return value, None

    if channel.calibration_type == 2 and channel.bitfields:
        bf = channel.bitfields[0]
        masked = raw_value & bf.mask
        for entry in bf.entries:
            if masked == entry.entry_value:
                return masked, entry.entry_tag
        return masked, bf.default_tag

    return raw_value, None


def decode_updates(
    telemetry_bin: Path,
    packets,
    rate_defs: List[RateDef],
    channels: Dict[int, ChannelProp],
    updates_csv: Path,
    packet_summary_csv: Path,
):
    fastest = min(rd.interval_ticks for rd in rate_defs)
    with telemetry_bin.open("rb") as f:
        telem = f.read()

    expected_sizes = [p["size"] for p in packets]
    if sum(expected_sizes) != len(telem):
        raise ValueError(
            f"binary size mismatch: packet sizes sum={sum(expected_sizes)} bin={len(telem)}"
        )

    with updates_csv.open("w", newline="") as uf, packet_summary_csv.open("w", newline="") as pf:
        uw = csv.writer(uf)
        pw = csv.writer(pf)

        uw.writerow(
            [
                "packet_index",
                "packet_pts_time",
                "packet_timestamp_ticks",
                "sample_timestamp_ticks",
                "sample_time_sec",
                "channel_id",
                "channel_tag",
                "quantity_tag",
                "raw_value",
                "calibrated_value",
                "label",
            ]
        )

        pw.writerow(
            [
                "packet_index",
                "packet_size",
                "header_timestamp_ticks",
                "header_payload_len",
                "extra_bytes",
                "decoded_payload_bytes",
                "decoded_ticks",
                "packet_pts_time",
                "packet_duration_time",
            ]
        )

        offset = 0
        for i, pkt in enumerate(packets):
            pkt_size = pkt["size"]
            chunk = telem[offset:offset + pkt_size]
            offset += pkt_size

            if len(chunk) != pkt_size:
                raise ValueError(f"short packet read at index {i}")
            if pkt_size < 14:
                raise ValueError(f"packet {i} too small for header: {pkt_size}")

            ts_ticks = int.from_bytes(chunk[0:8], "big")
            _flags = int.from_bytes(chunk[8:10], "big")
            payload_len = int.from_bytes(chunk[10:14], "big")

            if 14 + payload_len > pkt_size:
                raise ValueError(f"packet {i} payload length overrun: {payload_len} > {pkt_size - 14}")

            payload = memoryview(chunk[14:14 + payload_len])
            extra = pkt_size - 14 - payload_len

            ptr = 0
            tick = 0
            while ptr < payload_len:
                sample_ticks = ts_ticks + tick * fastest
                start_ptr = ptr
                for rate_def in rate_defs:
                    if sample_ticks % rate_def.interval_ticks != 0:
                        continue
                    for chdef in rate_def.channels:
                        raw, ptr = decode_number(payload, ptr, chdef.raw_type)
                        ch = channels.get(chdef.channel_id)
                        if ch is None:
                            continue
                        calibrated, label = apply_calibration(raw, ch)
                        uw.writerow(
                            [
                                i,
                                f"{pkt['pts_time']:.6f}",
                                ts_ticks,
                                sample_ticks,
                                f"{sample_ticks / 10_000_000:.6f}",
                                ch.channel_id,
                                ch.tag,
                                ch.quantity_tag,
                                raw,
                                calibrated,
                                label or "",
                            ]
                        )
                consumed = ptr - start_ptr
                if consumed <= 0:
                    raise ValueError(f"decoder did not consume bytes at packet {i}, tick {tick}")
                tick += 1

            pw.writerow(
                [
                    i,
                    pkt_size,
                    ts_ticks,
                    payload_len,
                    extra,
                    ptr,
                    tick,
                    f"{pkt['pts_time']:.6f}",
                    f"{pkt['duration_time']:.6f}",
                ]
            )


def write_channels_csv(channels: Dict[int, ChannelProp], out_path: Path):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "channel_id",
                "tag",
                "quantity_id",
                "quantity_tag",
                "calibration_type",
                "calibrated_data_type",
                "gain",
                "offset",
                "raw_min",
                "raw_max",
            ]
        )
        for cid in sorted(channels):
            ch = channels[cid]
            if ch.numeric_calibration is not None:
                gain = ch.numeric_calibration.gain
                offset = ch.numeric_calibration.offset
                raw_min = ch.numeric_calibration.minimum
                raw_max = ch.numeric_calibration.maximum
            else:
                gain = ""
                offset = ""
                raw_min = ""
                raw_max = ""
            w.writerow(
                [
                    ch.channel_id,
                    ch.tag,
                    ch.quantity_id,
                    ch.quantity_tag,
                    ch.calibration_type,
                    ch.calibrated_data_type,
                    gain,
                    offset,
                    raw_min,
                    raw_max,
                ]
            )


def main():
    ap = argparse.ArgumentParser(description="Extract and decode AliveDrive telemetry from MP4")
    ap.add_argument("mp4", type=Path, help="Input MP4 file")
    ap.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory")
    args = ap.parse_args()

    mp4_path = args.mp4.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_index = find_adco_stream_index(mp4_path, cwd=out_dir)

    telemetry_bin = out_dir / f"{mp4_path.stem}.telemetry.bin"
    packet_csv = out_dir / f"{mp4_path.stem}.telemetry_packets.csv"
    demux_telemetry(mp4_path, stream_index, telemetry_bin, cwd=out_dir)
    export_packet_csv(mp4_path, stream_index, packet_csv, cwd=out_dir)

    children, mp4_mem = parse_adco_from_mp4(mp4_path)
    try:
        advi = parse_advi(mp4_mem, *children["advi"])
        units = parse_adud(mp4_mem, *children["adud"])
        rate_defs, raw_types = parse_adcr(mp4_mem, *children["adcr"])
        channels = parse_adcp(mp4_mem, *children["adcp"], units=units, channel_raw_types=raw_types)
    finally:
        # parse_adco_from_mp4 keeps mmap alive via memoryview; releasing memoryview closes mmap
        mp4_mem.release()

    packets = parse_packets(packet_csv)

    channels_csv = out_dir / f"{mp4_path.stem}.channels.csv"
    updates_csv = out_dir / f"{mp4_path.stem}.decoded_updates.csv"
    packet_summary_csv = out_dir / f"{mp4_path.stem}.packet_summary.csv"
    meta_json = out_dir / f"{mp4_path.stem}.meta.json"

    write_channels_csv(channels, channels_csv)
    decode_updates(
        telemetry_bin=telemetry_bin,
        packets=packets,
        rate_defs=rate_defs,
        channels=channels,
        updates_csv=updates_csv,
        packet_summary_csv=packet_summary_csv,
    )

    meta_json.write_text(
        json.dumps(
            {
                "input": str(mp4_path),
                "stream_index": stream_index,
                "advi": advi,
                "packet_count": len(packets),
                "channel_count": len(channels),
                "rate_definition_count": len(rate_defs),
                "fastest_interval_ticks": min(rd.interval_ticks for rd in rate_defs),
            },
            indent=2,
        )
    )

    print(f"wrote {telemetry_bin}")
    print(f"wrote {packet_csv}")
    print(f"wrote {channels_csv}")
    print(f"wrote {updates_csv}")
    print(f"wrote {packet_summary_csv}")
    print(f"wrote {meta_json}")


if __name__ == "__main__":
    main()
