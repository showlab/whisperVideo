#!/home/siyuan/miniconda3/envs/whisperv/bin/python
import argparse
import os
import sys
import glob
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch


DATASET_DEFAULT = \
    "/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset"


@dataclass
class Episode:
    show: str
    episode: str
    path: str  # episode directory path


def find_episodes(root: str) -> List[Episode]:
    def has_audio(p: str) -> bool:
        return os.path.isfile(os.path.join(p, "avi", "audio.wav"))

    eps: List[Episode] = []
    root = os.path.abspath(root)
    if has_audio(root):
        # Treat root as a single episode (show/episode inferred from last two parts)
        parts = root.rstrip("/").split("/")
        if len(parts) < 2:
            raise RuntimeError(f"Episode path too shallow: {root}")
        show = parts[-2]
        episode = parts[-1]
        eps.append(Episode(show=show, episode=episode, path=root))
        return eps

    lvl1 = [p for p in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(p)]
    # If level-1 folders are episodes directly
    lvl1_eps = [p for p in lvl1 if has_audio(p)]
    if lvl1_eps:
        for ep in lvl1_eps:
            parts = ep.rstrip("/").split("/")
            show = parts[-2]
            episode = parts[-1]
            eps.append(Episode(show=show, episode=episode, path=ep))
        return eps

    # Otherwise, expect show -> episode
    for show_dir in lvl1:
        lvl2 = [p for p in sorted(glob.glob(os.path.join(show_dir, "*"))) if os.path.isdir(p)]
        for ep in lvl2:
            if has_audio(ep):
                parts = ep.rstrip("/").split("/")
                show = parts[-2]
                episode = parts[-1]
                eps.append(Episode(show=show, episode=episode, path=ep))
    if not eps:
        raise RuntimeError(f"No episodes found under dataset root: {root}")
    return eps


def ffprobe_duration_seconds(audio_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({proc.returncode}) for {audio_path}: {proc.stderr.decode('utf-8','ignore')}")
    try:
        return float(proc.stdout.decode("utf-8").strip())
    except Exception as e:
        raise RuntimeError(f"Unable to parse ffprobe duration for {audio_path}: {e}")


def read_rttm_intervals(rttm_path: str) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            # SPEAKER <file-id> <chan> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
            tbeg = float(parts[3])
            tdur = float(parts[4])
            if tdur <= 0:
                continue
            start = float(tbeg)
            end = float(tbeg + tdur)
            if end > start:
                intervals.append((start, end))
    intervals.sort(key=lambda x: (x[0], x[1]))
    return intervals


def union_length(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += (ce - cs)
            cs, ce = s, e
    total += (ce - cs)
    return float(total)


def format_hms(sec: float) -> str:
    s = int(round(sec))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h}:{m:02d}:{s:02d}"


def process_one_episode(ep: Episode) -> Tuple[str, str, float, float]:
    audio = os.path.join(ep.path, "avi", "audio.wav")
    rttm = os.path.join(ep.path, "result", "google_stt", "segments.rttm")
    if not os.path.isfile(audio):
        raise RuntimeError(f"Missing audio: {audio}")
    if not os.path.isfile(rttm):
        raise RuntimeError(f"Missing RTTM: {rttm}")
    dur = ffprobe_duration_seconds(audio)
    if dur <= 0:
        raise RuntimeError(f"Non-positive duration from ffprobe for {audio}")
    intervals = read_rttm_intervals(rttm)
    speech = union_length(intervals)
    return ep.show, ep.episode, dur, speech


def main():
    parser = argparse.ArgumentParser(description="Compute speech ratio using Google STT RTTM (union over speakers per episode)")
    parser.add_argument("--dataset_root", type=str, default=DATASET_DEFAULT)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    eps = find_episodes(args.dataset_root)
    # Enforce complete coverage: every episode must have RTTM
    missing = []
    for ep in eps:
        if not os.path.isfile(os.path.join(ep.path, "result", "google_stt", "segments.rttm")):
            missing.append(f"{ep.show}/{ep.episode}")
    if missing:
        print("ERROR: Missing RTTM for episodes:")
        for m in missing:
            print("  -", m)
        sys.exit(2)

    mp = torch.multiprocessing
    with mp.Pool(processes=int(args.workers)) as pool:
        rows = pool.map(process_one_episode, eps)

    # Aggregate
    per_show: Dict[str, Dict[str, float]] = {}
    per_ep: List[Dict[str, object]] = []
    total_dur = 0.0
    total_speech = 0.0
    for show, episode, dur, speech in rows:
        per_ep.append({
            "show": show,
            "episode": episode,
            "duration_sec": dur,
            "speech_sec": speech,
            "speech_ratio": float(speech / dur) if dur > 0 else float('nan'),
        })
        g = per_show.setdefault(show, {"duration_sec": 0.0, "speech_sec": 0.0})
        g["duration_sec"] += dur
        g["speech_sec"] += speech
        total_dur += dur
        total_speech += speech

    # Print report
    print("=== Speech Ratio from Google RTTM (union per episode) ===")
    print(f"Episodes: {len(rows)}")
    print("")
    print("Per show:")
    for show in sorted(per_show.keys()):
        dur = per_show[show]["duration_sec"]
        sp = per_show[show]["speech_sec"]
        ratio = (sp / dur) if dur > 0 else float('nan')
        print(f"  {show}: duration={format_hms(dur)} ({dur:.2f}s), speech={format_hms(sp)} ({sp:.2f}s), ratio={ratio*100:.2f}%")
    print("")
    overall_ratio = (total_speech / total_dur) if total_dur > 0 else float('nan')
    print(f"Total: duration={format_hms(total_dur)} ({total_dur:.2f}s), speech={format_hms(total_speech)} ({total_speech:.2f}s), ratio={overall_ratio*100:.2f}%")

    if args.json_out:
        payload = {
            "dataset_root": args.dataset_root,
            "episodes": len(rows),
            "per_show": {
                show: {
                    "duration_sec": per_show[show]["duration_sec"],
                    "speech_sec": per_show[show]["speech_sec"],
                    "speech_ratio": (per_show[show]["speech_sec"] / per_show[show]["duration_sec"]) if per_show[show]["duration_sec"] > 0 else None,
                }
                for show in per_show
            },
            "total": {
                "duration_sec": total_dur,
                "speech_sec": total_speech,
                "speech_ratio": overall_ratio,
            },
            "per_episode": per_ep,
        }
        outp = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON: {outp}")


if __name__ == "__main__":
    main()

