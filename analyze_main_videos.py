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
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"]) if fmt["duration"] is not None else None
        except Exception:
            duration = None
    if duration is None:
        for s in vstreams:
            try:
                d = s.get("duration")
                if d is not None:
                    duration = float(d)
                    break
            except Exception:
                pass
    if duration is None and vstreams:
        s0 = vstreams[0]
        nb_frames = s0.get("nb_frames")
        fps = parse_fraction(s0.get("avg_frame_rate") or s0.get("r_frame_rate"))
        try:
            if nb_frames is not None and fps and float(fps) > 0:
                duration = float(nb_frames) / float(fps)
        except Exception:
            pass

    width = height = None
    fps = None
    vcodec = None
    pix_fmt = None
    if vstreams:
        s0 = vstreams[0]
        width = s0.get("width")
        height = s0.get("height")
        fps = parse_fraction(s0.get("avg_frame_rate") or s0.get("r_frame_rate"))
        vcodec = s0.get("codec_name")
        pix_fmt = s0.get("pix_fmt")

    acodec = None
    asr = None
    ach = None
    if astreams:
        a0 = astreams[0]
        acodec = a0.get("codec_name")
        try:
            asr = int(a0.get("sample_rate")) if a0.get("sample_rate") is not None else None
        except Exception:
            asr = None
        ach = a0.get("channels")

    return {
        "duration": duration,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": vcodec,
            "pix_fmt": pix_fmt,
        },
        "audio": {
            "present": bool(astreams),
            "codec": acodec,
            "sample_rate": asr,
            "channels": ach,
        },
        "container": {
            "format_name": fmt.get("format_name"),
        },
        "streams_count": {
            "video": len(vstreams),
            "audio": len(astreams),
        },
    }


def safe_probe(path: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        meta = ffprobe_json(path)
        info = extract_video_info(meta)
        if info["duration"] is None:
            return path, None, f"missing_duration"
        if info["video"]["width"] is None or info["video"]["height"] is None:
            return path, None, f"missing_resolution"
        return path, info, None
    except Exception as e:
        return path, None, f"error:{type(e).__name__}:{e}"


def collect_main_videos(root: Path) -> List[Path]:
    out: List[Path] = []
    shows = [p for p in root.iterdir() if p.is_dir()]
    for show in shows:
        eps = [p for p in show.iterdir() if p.is_dir()]
        for ep in eps:
            p = ep / "avi" / "video.avi"
            if p.exists() and p.is_file():
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
    parser = argparse.ArgumentParser(description="Analyze main episode videos only (avi/video.avi per episode)")
    parser.add_argument("--dataset", type=Path, default=Path(DATASET_DEFAULT), help="Dataset root directory")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel workers")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    require_ffprobe()

    root: Path = args.dataset
    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset path not found: {root}", file=sys.stderr)
        sys.exit(3)

    # Enumerate episodes and verify presence of main video.
    episodes: List[Tuple[str, str]] = []
    for show in sorted([p for p in root.iterdir() if p.is_dir()]):
        for ep in sorted([p for p in show.iterdir() if p.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else x.name):
            episodes.append((show.name, ep.name))

    main_files = collect_main_videos(root)
    if not main_files:
        print("ERROR: no main videos found (expected avi/video.avi under each episode).", file=sys.stderr)
        sys.exit(4)

    # Check coverage
    missing = []
    for show, ep in episodes:
        p = root / show / ep / "avi" / "video.avi"
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: missing main video files for some episodes:")
        for m in missing:
            print("  -", m)
        sys.exit(5)

    print(f"Found {len(main_files)} main episode videos. Probing with {args.workers} workers...")

    mp = torch.multiprocessing
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(safe_probe, [str(p) for p in main_files])

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

    if len(infos) != len(main_files):
        print("ERROR: some main videos failed to parse; refusing to summarize incompletely.", file=sys.stderr)
        sys.exit(6)

    # Aggregate
    durations = []
    fps_values = []
    widths = []
    heights = []
    vcodecs: Dict[str, int] = {}
    containers: Dict[str, int] = {}
    audio_present = 0
    asr_values: List[int] = []
    ach_values: List[int] = []
    for path, info in infos.items():
        durations.append(info["duration"])  # already validated non-None
        if info["video"]["fps"] is not None:
            fps_values.append(info["video"]["fps"])
        widths.append(info["video"]["width"])  # non-None
        heights.append(info["video"]["height"])  # non-None
        vc = info["video"]["codec"]
        if vc:
            vcodecs[vc] = vcodecs.get(vc, 0) + 1
        cont = info["container"]["format_name"]
        if cont:
            containers[cont] = containers.get(cont, 0) + 1
        if info["audio"]["present"]:
            audio_present += 1
            if info["audio"]["sample_rate"]:
                try:
                    asr_values.append(int(info["audio"]["sample_rate"]))
                except Exception:
                    pass
            if info["audio"]["channels"]:
                try:
                    ach_values.append(int(info["audio"]["channels"]))
                except Exception:
                    pass

    dur_sum = summarize_nums(durations)
    fps_sum = summarize_nums(fps_values)
    total_sec = float(sum(durations))

    # Group per show/episode
    per_episode: Dict[str, Dict[str, Any]] = {}
    per_show: Dict[str, List[float]] = {}
    for fpath in infos.keys():
        show, ep = group_keys(root, Path(fpath))
        key = f"{show}/{ep}"
        per_episode[key] = {
            "duration_sec": infos[fpath]["duration"],
            "width": infos[fpath]["video"]["width"],
            "height": infos[fpath]["video"]["height"],
            "fps": infos[fpath]["video"]["fps"],
            "vcodec": infos[fpath]["video"]["codec"],
            "container": infos[fpath]["container"]["format_name"],
            "audio_present": infos[fpath]["audio"]["present"],
            "audio_sr": infos[fpath]["audio"]["sample_rate"],
            "audio_ch": infos[fpath]["audio"]["channels"],
            "path": fpath,
        }
        per_show.setdefault(show, []).append(infos[fpath]["duration"])

    print("")
    print("=== Main Episode Videos Summary ===")
    print(f"Episodes: {len(main_files)} (shows: {len(per_show)})")
    print(f"Total Duration: {format_seconds(total_sec)} ({total_sec:.2f}s)")
    print(
        f"Duration mean/median: {dur_sum['mean']:.2f}s / {dur_sum['median']:.2f}s; "
        f"min/max: {dur_sum['min']:.2f}s / {dur_sum['max']:.2f}s; P10/P90: {dur_sum['p10']:.2f}s / {dur_sum['p90']:.2f}s"
    )
    print(
        f"FPS mean/median: {fps_sum['mean']:.2f} / {fps_sum['median']:.2f}; "
        f"min/max: {fps_sum['min']:.2f} / {fps_sum['max']:.2f}"
    )
    print(f"Video codecs: {sorted(vcodecs.items(), key=lambda x: -x[1])}")
    print(f"Containers: {sorted(containers.items(), key=lambda x: -x[1])}")
    print(f"Audio present: {audio_present}/{len(main_files)} ({audio_present/len(main_files):.1%})")
    if asr_values:
        print(f"Audio sample rates unique: {sorted({int(x) for x in asr_values})}")
    if ach_values:
        print(f"Audio channels unique: {sorted({int(x) for x in ach_values})}")

    print("")
    print("=== Per Show (duration sums) ===")
    for show in sorted(per_show.keys()):
        ssum = float(sum(per_show[show]))
        print(f"{show}: episodes={len(per_show[show])}, total={format_seconds(ssum)} ({ssum:.2f}s)")

    print("")
    print("=== Per Episode (main video) ===")
    for key in sorted(per_episode.keys(), key=lambda k: (k.split('/')[0], int(k.split('/')[1]) if k.split('/')[1].isdigit() else k.split('/')[1])):
        info = per_episode[key]
        print(
            f"{key}: dur={format_seconds(info['duration_sec'])} ({info['duration_sec']:.2f}s), "
            f"res={info['width']}x{info['height']} @ {info['fps']:.2f}fps, "
            f"vcodec={info['vcodec']}, audio={'Y' if info['audio_present'] else 'N'}"
        )

    if args.json_out is not None:
        payload = {
            "dataset_root": str(root),
            "episodes": len(main_files),
            "global": {
                "duration_sec_sum": total_sec,
                "duration_stats": dur_sum,
                "fps_stats": fps_sum,
                "video_codecs": vcodecs,
                "containers": containers,
                "audio_present_count": audio_present,
                "audio_sample_rates": sorted({int(x) for x in asr_values}) if asr_values else [],
                "audio_channels": sorted({int(x) for x in ach_values}) if ach_values else [],
            },
            "per_show": {k: {"episodes": len(v), "duration_sec_sum": float(sum(v))} for k, v in per_show.items()},
            "per_episode": per_episode,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Wrote JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()
