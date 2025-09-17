#!/home/siyuan/miniconda3/envs/whisperv/bin/python
import argparse
import os
import sys
import json
from pathlib import Path
import statistics
import wave
from typing import Dict, List, Tuple, Optional

import torch


DATASET_DEFAULT = \
    "/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset"


def is_audio_file(p: Path) -> bool:
    ext = p.suffix.lower()
    return ext in {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


def audio_format_supported(p: Path) -> bool:
    # For now we only implement precise handling for WAV using stdlib.
    # If other formats are present, we error out explicitly to avoid assumptions.
    return p.suffix.lower() == ".wav"


def compute_wav_info(path: str) -> Tuple[str, float, int, int]:
    # Returns (path, duration_sec, sample_rate, n_channels)
    with wave.open(path, "rb") as w:
        nframes = w.getnframes()
        fr = w.getframerate()
        ch = w.getnchannels()
        duration = nframes / float(fr) if fr > 0 else 0.0
        return path, duration, fr, ch


def safe_compute_info(path: str) -> Tuple[str, Optional[Tuple[str, float, int, int]], Optional[str]]:
    p = Path(path)
    try:
        if not audio_format_supported(p):
            return path, None, f"unsupported_format:{p.suffix.lower()}"
        info = compute_wav_info(path)
        return path, info, None
    except Exception as e:
        return path, None, f"error:{type(e).__name__}:{e}"


def collect_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_audio_file(p):
                files.append(p)
    return files


def group_keys(dataset_root: Path, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Expect structure: root/show/episode/.../file
    try:
        rel = file_path.relative_to(dataset_root)
    except Exception:
        return None, None
    parts = rel.parts
    show = parts[0] if len(parts) >= 1 else None
    episode = parts[1] if len(parts) >= 2 else None
    return show, episode


def summarize(durations: List[float]) -> Dict[str, float]:
    if not durations:
        return {
            "count": 0,
            "total_sec": 0.0,
            "mean_sec": 0.0,
            "median_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
            "p10_sec": 0.0,
            "p90_sec": 0.0,
        }
    durations_sorted = sorted(durations)
    return {
        "count": float(len(durations)),
        "total_sec": float(sum(durations)),
        "mean_sec": float(statistics.fmean(durations)),
        "median_sec": float(statistics.median(durations)),
        "min_sec": float(durations_sorted[0]),
        "max_sec": float(durations_sorted[-1]),
        "p10_sec": float(durations_sorted[max(0, int(0.10 * (len(durations_sorted) - 1)))]),
        "p90_sec": float(durations_sorted[max(0, int(0.90 * (len(durations_sorted) - 1)))]),
    }


def format_seconds(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi human talking dataset: durations and counts"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DATASET_DEFAULT),
        help="Path to dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel workers (torch.multiprocessing)",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )

    args = parser.parse_args()
    root: Path = args.dataset

    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset path not found: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"Scanning dataset under: {root}")
    audio_files = collect_audio_files(root)
    if not audio_files:
        print("ERROR: no audio files found (expected e.g., .wav)", file=sys.stderr)
        sys.exit(3)

    unsupported_exts = sorted({p.suffix.lower() for p in audio_files if not audio_format_supported(p)})
    if unsupported_exts:
        examples = [str(p) for p in audio_files if p.suffix.lower() in unsupported_exts][:5]
        print(
            "ERROR: found unsupported audio formats. This script currently only supports .wav\n"
            f"Unsupported extensions: {unsupported_exts}\n"
            f"Examples: {json.dumps(examples, ensure_ascii=False, indent=2)}",
            file=sys.stderr,
        )
        sys.exit(4)

    # Parallel compute using torch.multiprocessing
    print(f"Found {len(audio_files)} audio files. Computing durations with {args.workers} workers...")
    mp = torch.multiprocessing
    # Avoid forcing a different start method globally; rely on default to keep it simple.
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(safe_compute_info, [str(p) for p in audio_files])

    # Gather results
    durations_by_file: Dict[str, float] = {}
    sample_rates: List[int] = []
    channels: List[int] = []
    errors: List[Tuple[str, str]] = []
    for path, info, err in results:
        if err is not None:
            errors.append((path, err))
        elif info is not None:
            _path, dur, sr, ch = info
            durations_by_file[path] = dur
            sample_rates.append(sr)
            channels.append(ch)

    if errors:
        print(f"Encountered {len(errors)} files with errors:")
        for p, e in errors[:20]:
            print(f"  - {p} :: {e}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")

    if not durations_by_file:
        print("ERROR: no durations computed successfully.", file=sys.stderr)
        sys.exit(5)

    # Group by show/episode
    per_show: Dict[str, List[float]] = {}
    per_episode: Dict[Tuple[str, str], List[float]] = {}
    for f, dur in durations_by_file.items():
        show, episode = group_keys(root, Path(f))
        if show is not None:
            per_show.setdefault(show, []).append(dur)
        if show is not None and episode is not None:
            per_episode.setdefault((show, episode), []).append(dur)

    # Global summary
    global_durations = list(durations_by_file.values())
    global_summary = summarize(global_durations)

    # Sample rate and channel stats (if present)
    sr_summary = {
        "unique_sample_rates": sorted(list({int(sr) for sr in sample_rates})),
        "common_sample_rate": int(statistics.mode(sample_rates)) if sample_rates else None,
    }
    ch_summary = {
        "unique_channels": sorted(list({int(ch) for ch in channels})),
        "common_channels": int(statistics.mode(channels)) if channels else None,
    }

    # Print summary
    print("")
    print("=== Global Summary ===")
    print(f"Files: {int(global_summary['count'])}")
    print(f"Total Duration: {format_seconds(global_summary['total_sec'])} ({global_summary['total_sec']:.2f}s)")
    print(
        f"Mean/Median: {global_summary['mean_sec']:.2f}s / {global_summary['median_sec']:.2f}s; "
        f"Min/Max: {global_summary['min_sec']:.2f}s / {global_summary['max_sec']:.2f}s; "
        f"P10/P90: {global_summary['p10_sec']:.2f}s / {global_summary['p90_sec']:.2f}s"
    )
    print(f"Sample Rates: {sr_summary['unique_sample_rates']} (mode={sr_summary['common_sample_rate']})")
    print(f"Channels: {ch_summary['unique_channels']} (mode={ch_summary['common_channels']})")

    # Per-show summary
    print("")
    print("=== Per Show Summary ===")
    for show in sorted(per_show.keys()):
        s = summarize(per_show[show])
        print(
            f"{show}: files={int(s['count'])}, total={format_seconds(s['total_sec'])} "
            f"({s['total_sec']:.2f}s), mean={s['mean_sec']:.2f}s, median={s['median_sec']:.2f}s"
        )

    # Per-episode summary
    print("")
    print("=== Per Episode Summary (show/episode) ===")
    # Sort by show then episode natural order
    for (show, episode) in sorted(per_episode.keys(), key=lambda x: (x[0], x[1])):
        s = summarize(per_episode[(show, episode)])
        print(
            f"{show}/{episode}: files={int(s['count'])}, total={format_seconds(s['total_sec'])} "
            f"({s['total_sec']:.2f}s), mean={s['mean_sec']:.2f}s, median={s['median_sec']:.2f}s"
        )

    # Optional JSON output
    if args.json_out is not None:
        payload = {
            "dataset_root": str(root),
            "global": global_summary,
            "sample_rates": sr_summary,
            "channels": ch_summary,
            "per_show": {k: summarize(v) for k, v in per_show.items()},
            "per_episode": {f"{k[0]}/{k[1]}": summarize(v) for k, v in per_episode.items()},
            "errors": errors,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Wrote JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

