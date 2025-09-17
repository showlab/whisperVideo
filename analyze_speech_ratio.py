#!/home/siyuan/miniconda3/envs/whisperv/bin/python
import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch


DATASET_DEFAULT = \
    "/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset"


def require_bins() -> None:
    # Require ffmpeg for decoding audio
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ERROR: ffmpeg not found in PATH. Please install ffmpeg.", file=sys.stderr)
        sys.exit(2)
    try:
        import webrtcvad  # noqa: F401
    except Exception:
        print("ERROR: python package 'webrtcvad' not found. Install with: pip install webrtcvad", file=sys.stderr)
        sys.exit(2)


def collect_main_videos(root: Path) -> List[Path]:
    out: List[Path] = []
    for show in sorted([p for p in root.iterdir() if p.is_dir()]):
        for ep in sorted([p for p in show.iterdir() if p.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else x.name):
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


def read_audio_pcm_s16le(video_path: str, sr: int = 16000, channels: int = 1) -> bytes:
    # Decode to raw PCM 16-bit little-endian via ffmpeg
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", str(channels), "-ar", str(sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'ignore').strip()}")
    return proc.stdout


def compute_speech_ratio_for_pcm(raw: bytes, sr: int, frame_ms: int, vad_level: int) -> Tuple[float, float, int, int]:
    """
    Returns: (duration_sec, speech_sec, total_frames, speech_frames)
    """
    import webrtcvad

    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be one of {10,20,30}")
    vad = webrtcvad.Vad(vad_level)

    bytes_per_sample = 2  # s16le
    frame_bytes = int(sr * (frame_ms / 1000.0)) * bytes_per_sample
    if frame_bytes == 0:
        raise ValueError("computed frame_bytes == 0; check sr/frame_ms")

    total_frames = 0
    speech_frames = 0
    # iterate over contiguous frames; drop tail if incomplete
    for i in range(0, len(raw) - (len(raw) % frame_bytes), frame_bytes):
        frame = raw[i : i + frame_bytes]
        is_speech = vad.is_speech(frame, sr)
        total_frames += 1
        if is_speech:
            speech_frames += 1

    duration_sec = (len(raw) / bytes_per_sample) / float(sr)
    speech_sec = speech_frames * (frame_ms / 1000.0)
    return duration_sec, speech_sec, total_frames, speech_frames


def process_one(video_path: str, sr: int, frame_ms: int, vad_level: int) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = read_audio_pcm_s16le(video_path, sr=sr, channels=1)
        duration_sec, speech_sec, total_frames, speech_frames = compute_speech_ratio_for_pcm(
            raw, sr=sr, frame_ms=frame_ms, vad_level=vad_level
        )
        ratio = 0.0
        if duration_sec > 0:
            # Prefer frame-based ratio to avoid rounding drift
            ratio = (speech_frames / total_frames) if total_frames > 0 else 0.0
        out = {
            "duration_sec": float(duration_sec),
            "speech_sec": float(speech_sec),
            "speech_ratio": float(ratio),
            "total_frames": int(total_frames),
            "speech_frames": int(speech_frames),
            "sample_rate": int(sr),
            "frame_ms": int(frame_ms),
            "vad_level": int(vad_level),
            "path": video_path,
        }
        return video_path, out, None
    except Exception as e:
        return video_path, None, f"error:{type(e).__name__}:{e}"


def format_seconds(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute speech ratio for each episode main video (avi/video.avi)")
    parser.add_argument("--dataset", type=Path, default=Path(DATASET_DEFAULT), help="Dataset root directory")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel workers (torch.multiprocessing)")
    parser.add_argument("--sr", type=int, default=16000, help="Decode sample rate for VAD (Hz)")
    parser.add_argument("--frame_ms", type=int, default=30, help="VAD frame size in ms (10/20/30)")
    parser.add_argument("--vad_level", type=int, default=2, help="WebRTC VAD aggressiveness (0-3)")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    require_bins()

    root: Path = args.dataset
    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset path not found: {root}", file=sys.stderr)
        sys.exit(3)

    # List episodes and verify coverage
    episodes: List[Tuple[str, str]] = []
    for show in sorted([p for p in root.iterdir() if p.is_dir()]):
        for ep in sorted([p for p in show.iterdir() if p.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else x.name):
            episodes.append((show.name, ep.name))

    main_files = collect_main_videos(root)
    if not main_files:
        print("ERROR: no main videos found (expected avi/video.avi under each episode).", file=sys.stderr)
        sys.exit(4)

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

    print(f"Found {len(main_files)} main videos. Computing speech ratio with {args.workers} workers...")

    mp = torch.multiprocessing
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(
            process_one,
            [(str(p), int(args.sr), int(args.frame_ms), int(args.vad_level)) for p in main_files],
        )

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
        print("ERROR: some episodes failed speech ratio computation; refusing to summarize incompletely.", file=sys.stderr)
        sys.exit(6)

    # Group by show/episode
    per_episode: Dict[str, Dict[str, Any]] = {}
    per_show_duration_sum: Dict[str, float] = {}
    per_show_speech_sum: Dict[str, float] = {}
    for fpath, info in infos.items():
        show, ep = group_keys(root, Path(fpath))
        key = f"{show}/{ep}"
        per_episode[key] = info
        per_show_duration_sum[show] = per_show_duration_sum.get(show, 0.0) + info["duration_sec"]
        per_show_speech_sum[show] = per_show_speech_sum.get(show, 0.0) + info["speech_sec"]

    total_duration = sum(v["duration_sec"] for v in infos.values())
    total_speech = sum(v["speech_sec"] for v in infos.values())
    global_ratio = (total_speech / total_duration) if total_duration > 0 else 0.0

    print("")
    print("=== Speech Ratio (main videos) ===")
    print(f"Episodes: {len(main_files)} | Total duration: {format_seconds(total_duration)} ({total_duration:.2f}s)")
    print(f"Global speech: {format_seconds(total_speech)} ({total_speech:.2f}s) | Ratio: {global_ratio:.3f}")

    print("")
    print("=== Per Show (duration-weighted ratio) ===")
    for show in sorted(per_show_duration_sum.keys()):
        dur = per_show_duration_sum[show]
        sp = per_show_speech_sum[show]
        ratio = (sp / dur) if dur > 0 else 0.0
        print(f"{show}: duration={format_seconds(dur)} ({dur:.2f}s), speech={format_seconds(sp)} ({sp:.2f}s), ratio={ratio:.3f}")

    print("")
    print("=== Per Episode ===")
    for key in sorted(per_episode.keys(), key=lambda k: (k.split('/')[0], int(k.split('/')[1]) if k.split('/')[1].isdigit() else k.split('/')[1])):
        info = per_episode[key]
        print(
            f"{key}: dur={format_seconds(info['duration_sec'])} ({info['duration_sec']:.2f}s), "
            f"speech={format_seconds(info['speech_sec'])} ({info['speech_sec']:.2f}s), ratio={info['speech_ratio']:.3f}"
        )

    if args.json_out is not None:
        payload = {
            "dataset_root": str(root),
            "episodes": len(main_files),
            "global": {
                "duration_sec_sum": total_duration,
                "speech_sec_sum": total_speech,
                "speech_ratio": global_ratio,
            },
            "per_show": {
                show: {
                    "duration_sec_sum": per_show_duration_sum[show],
                    "speech_sec_sum": per_show_speech_sum[show],
                    "speech_ratio": (per_show_speech_sum[show] / per_show_duration_sum[show]) if per_show_duration_sum[show] > 0 else 0.0,
                }
                for show in per_show_duration_sum.keys()
            },
            "per_episode": per_episode,
            "params": {
                "sample_rate": int(args.sr),
                "frame_ms": int(args.frame_ms),
                "vad_level": int(args.vad_level),
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("")
        print(f"Wrote JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

