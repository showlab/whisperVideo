#!/home/siyuan/miniconda3/envs/whisperv/bin/python
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics
import subprocess

import torch


DATASET_DEFAULT = \
    "/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset"


VIDEO_EXTS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".m4v", ".ts", ".mxf",
}


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def require_ffprobe() -> None:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ERROR: ffprobe not found in PATH. Please install ffmpeg/ffprobe.", file=sys.stderr)
        sys.exit(2)


def ffprobe_json(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe JSON parse error: {e}")


def parse_fraction(frac: Optional[str]) -> Optional[float]:
    if not frac:
        return None
    if "/" in frac:
        num, den = frac.split("/", 1)
        try:
            numf = float(num)
            denf = float(den)
            return numf / denf if denf != 0 else None
        except Exception:
            return None
    try:
        return float(frac)
    except Exception:
        return None


def extract_video_info(meta: Dict[str, Any]) -> Dict[str, Any]:
    fmt = meta.get("format", {})
    streams: List[Dict[str, Any]] = meta.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]

    duration = None
    # Prefer container duration
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"]) if fmt["duration"] is not None else None
        except Exception:
            duration = None
    # Else try any video stream duration
    if duration is None:
        for s in vstreams:
            try:
                d = s.get("duration")
                if d is not None:
                    duration = float(d)
                    break
            except Exception:
                pass
    # Else try frames/rate
    if duration is None and vstreams:
        s0 = vstreams[0]
        nb_frames = s0.get("nb_frames")
        fps = parse_fraction(s0.get("avg_frame_rate") or s0.get("r_frame_rate"))
        try:
            if nb_frames is not None and fps and float(fps) > 0:
                duration = float(nb_frames) / float(fps)
        except Exception:
            pass

    # Resolution and fps/codec for primary video stream
    width = height = None
    fps = None
    vcodec = None
    pix_fmt = None
    bit_rate_v = None
    nb_frames_v = None
    if vstreams:
        s0 = vstreams[0]
        width = s0.get("width")
        height = s0.get("height")
        fps = parse_fraction(s0.get("avg_frame_rate") or s0.get("r_frame_rate"))
        vcodec = s0.get("codec_name")
        pix_fmt = s0.get("pix_fmt")
        try:
            bit_rate_v = int(s0.get("bit_rate")) if s0.get("bit_rate") is not None else None
        except Exception:
            bit_rate_v = None
        try:
            nb_frames_v = int(s0.get("nb_frames")) if s0.get("nb_frames") is not None else None
        except Exception:
            nb_frames_v = None

    # Audio summary (choose first audio stream for detailed fields)
    acodec = None
    asr = None
    ach = None
    bit_rate_a = None
    if astreams:
        a0 = astreams[0]
        acodec = a0.get("codec_name")
        try:
            asr = int(a0.get("sample_rate")) if a0.get("sample_rate") is not None else None
        except Exception:
            asr = None
        ach = a0.get("channels")
        try:
            bit_rate_a = int(a0.get("bit_rate")) if a0.get("bit_rate") is not None else None
        except Exception:
            bit_rate_a = None

    container_bit_rate = None
    try:
        container_bit_rate = int(fmt.get("bit_rate")) if fmt.get("bit_rate") is not None else None
    except Exception:
        container_bit_rate = None

    return {
        "duration": duration,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": vcodec,
            "pix_fmt": pix_fmt,
            "bit_rate": bit_rate_v,
            "nb_frames": nb_frames_v,
        },
        "audio": {
            "present": bool(astreams),
            "codec": acodec,
            "sample_rate": asr,
            "channels": ach,
            "bit_rate": bit_rate_a,
            "num_streams": len(astreams),
        },
        "container": {
            "format_name": fmt.get("format_name"),
            "bit_rate": container_bit_rate,
            "size": int(fmt.get("size")) if fmt.get("size") is not None else None,
        },
        "streams_count": {
            "video": len(vstreams),
            "audio": len(astreams),
            "other": max(0, len(streams) - len(vstreams) - len(astreams)),
        },
    }


def safe_probe(path: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        meta = ffprobe_json(path)
        info = extract_video_info(meta)
        # Validate required fields
        if info["duration"] is None:
            return path, None, "missing_duration"
        if info["video"]["width"] is None or info["video"]["height"] is None:
            return path, None, "missing_resolution"
        return path, info, None
    except Exception as e:
        return path, None, f"error:{type(e).__name__}:{e}"


def collect_video_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_video_file(p):
                out.append(p)
    return out


def group_keys(dataset_root: Path, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        rel = file_path.relative_to(dataset_root)
    except Exception:
        return None, None
    parts = rel.parts
    show = parts[0] if len(parts) >= 1 else None
    episode = parts[1] if len(parts) >= 2 else None
    return show, episode


def summarize_nums(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0}
    ys = sorted(xs)
    return {
        "count": float(len(ys)),
        "mean": float(statistics.fmean(ys)),
        "median": float(statistics.median(ys)),
        "min": float(ys[0]),
        "max": float(ys[-1]),
        "p10": float(ys[max(0, int(0.10 * (len(ys) - 1)))]),
        "p90": float(ys[max(0, int(0.90 * (len(ys) - 1)))]),
    }


def format_seconds(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive video stats via ffprobe (parallel with torch.multiprocessing)")
    parser.add_argument("--dataset", type=Path, default=Path(DATASET_DEFAULT), help="Dataset root directory")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel workers")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    require_ffprobe()

    root: Path = args.dataset
    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset path not found: {root}", file=sys.stderr)
        sys.exit(3)

    print(f"Scanning videos under: {root}")
    vfiles = collect_video_files(root)
    if not vfiles:
        print("ERROR: no video files found.", file=sys.stderr)
        sys.exit(4)
    print(f"Found {len(vfiles)} videos. Probing with {args.workers} workers...")

    mp = torch.multiprocessing
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(safe_probe, [str(p) for p in vfiles])

    infos: Dict[str, Dict[str, Any]] = {}
    errors: List[Tuple[str, str]] = []
    for path, info, err in results:
        if err is not None:
            errors.append((path, err))
        elif info is not None:
            infos[path] = info

    if errors:
        print(f"Encountered {len(errors)} files with errors.")
        for p, e in errors[:20]:
            print(f"  - {p} :: {e}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")

    if not infos:
        print("ERROR: no video metadata parsed successfully.", file=sys.stderr)
        sys.exit(5)

    # Aggregate statistics
    durations = [v["duration"] for v in infos.values() if v["duration"] is not None]
    fps_values = [v["video"]["fps"] for v in infos.values() if v["video"]["fps"] is not None]
    widths = [v["video"]["width"] for v in infos.values() if v["video"]["width"] is not None]
    heights = [v["video"]["height"] for v in infos.values() if v["video"]["height"] is not None]
    vcodecs = {}
    resolutions = {}
    containers = {}
    has_audio = 0
    asr_values = []
    ach_values = []
    for info in infos.values():
        vcodec = info["video"]["codec"]
        res = (info["video"]["width"], info["video"]["height"])
        container = info["container"]["format_name"]
        if vcodec:
            vcodecs[vcodec] = vcodecs.get(vcodec, 0) + 1
        if all(res):
            resolutions[res] = resolutions.get(res, 0) + 1
        if container:
            containers[container] = containers.get(container, 0) + 1
        if info["audio"]["present"]:
            has_audio += 1
            if info["audio"]["sample_rate"]:
                asr_values.append(info["audio"]["sample_rate"])
            if info["audio"]["channels"]:
                ach_values.append(info["audio"]["channels"])

    # Group by show/episode
    per_show: Dict[str, List[float]] = {}
    per_episode: Dict[Tuple[str, str], List[float]] = {}
    for fpath, info in infos.items():
        show, ep = group_keys(root, Path(fpath))
        if show is not None:
            per_show.setdefault(show, []).append(info["duration"])
        if show is not None and ep is not None:
            per_episode.setdefault((show, ep), []).append(info["duration"])

    dur_sum = summarize_nums(durations)
    fps_sum = summarize_nums(fps_values)

    print("")
    print("=== Global Video Summary ===")
    print(f"Videos parsed: {int(dur_sum['count'])} / {len(vfiles)}")
    total_sec = float(sum(durations)) if durations else 0.0
    print(f"Total Duration: {format_seconds(total_sec)} ({total_sec:.2f}s)")
    print(
        f"Duration mean/median: {dur_sum['mean']:.2f}s / {dur_sum['median']:.2f}s; "
        f"min/max: {dur_sum['min']:.2f}s / {dur_sum['max']:.2f}s; "
        f"P10/P90: {dur_sum['p10']:.2f}s / {dur_sum['p90']:.2f}s"
    )
    print(
        f"FPS mean/median: {fps_sum['mean']:.2f} / {fps_sum['median']:.2f}; "
        f"min/max: {fps_sum['min']:.2f} / {fps_sum['max']:.2f}"
    )
    print(f"Resolution samples: {len(resolutions)} unique; common: {sorted(resolutions.items(), key=lambda x: -x[1])[:5]}")
    print(f"Video codecs top: {sorted(vcodecs.items(), key=lambda x: -x[1])[:5]}")
    print(f"Containers top: {sorted(containers.items(), key=lambda x: -x[1])[:5]}")
    print(f"Files with audio: {has_audio} ({has_audio/len(infos):.1%})")
    if asr_values:
        try:
            print(f"Audio sample rates unique: {sorted({int(x) for x in asr_values})}")
            print(f"Audio channels unique: {sorted({int(x) for x in ach_values})}")
        except Exception:
            pass

    print("")
    print("=== Per Show (duration sums) ===")
    for show in sorted(per_show.keys()):
        ssum = float(sum(per_show[show])) if per_show[show] else 0.0
        print(f"{show}: videos={len(per_show[show])}, total={format_seconds(ssum)} ({ssum:.2f}s)")

    print("")
    print("=== Per Episode (duration sums) ===")
    for (show, ep) in sorted(per_episode.keys(), key=lambda x: (x[0], x[1])):
        ssum = float(sum(per_episode[(show, ep)])) if per_episode[(show, ep)] else 0.0
        print(f"{show}/{ep}: videos={len(per_episode[(show, ep)])}, total={format_seconds(ssum)} ({ssum:.2f}s)")

    if args.json_out is not None:
        payload = {
            "dataset_root": str(root),
            "global": {
                "videos_total": len(vfiles),
                "videos_parsed": int(dur_sum["count"]),
                "duration_sec_sum": total_sec,
                "duration_stats": dur_sum,
                "fps_stats": fps_sum,
                "resolutions": {f"{int(w)}x{int(h)}": c for (w, h), c in resolutions.items() if w and h},
                "video_codecs": vcodecs,
                "containers": containers,
                "audio_present_count": has_audio,
                "audio_sample_rates": sorted({int(x) for x in asr_values}) if asr_values else [],
                "audio_channels": sorted({int(x) for x in ach_values}) if ach_values else [],
            },
            "per_show": {k: {"videos": len(v), "duration_sec_sum": float(sum(v))} for k, v in per_show.items()},
            "per_episode": {f"{k[0]}/{k[1]}": {"videos": len(v), "duration_sec_sum": float(sum(v))} for k, v in per_episode.items()},
            "files": infos,
            "errors": errors,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Wrote JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

