#!/home/siyuan/miniconda3/envs/whisperv/bin/python
"""Clean Google STT ground-truth diarization by deduplicating overlapping copies.

Rewrites result/google_stt/{words.json, segments.csv, segments.rttm} to remove the
spurious multi-speaker duplicates emitted by Google STT diarization.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from google_stt_gt import WordInfo, episode_id_from_path, words_to_segments, write_rttm

DATASET_DEFAULT = Path("/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset")
EPS_GT_REL = Path("result/google_stt")
WORDS_FILE = "words.json"
CSV_FILE = "segments.csv"
RTTM_FILE = "segments.rttm"


@dataclass
class CleanStats:
    words_before: int
    words_after: int
    zero_dropped: int
    duplicate_dropped: int
    ghost_speakers: Tuple[int, ...]
    speakers_before: int
    speakers_after: int
    segments_before: int
    segments_after: int
    multi_frac_before: float
    multi_frac_after: float
    mean_concurrency_before: float
    mean_concurrency_after: float
    max_concurrency_before: int
    max_concurrency_after: int


def discover_episodes(dataset_root: Path) -> List[Path]:
    dataset_root = dataset_root.resolve()
    if (dataset_root / EPS_GT_REL / WORDS_FILE).is_file():
        return [dataset_root]
    episodes: List[Path] = []
    for show_dir in sorted(dataset_root.glob("*")):
        if not show_dir.is_dir():
            continue
        for ep_dir in sorted(show_dir.glob("*")):
            if not ep_dir.is_dir():
                continue
            if (ep_dir / EPS_GT_REL / WORDS_FILE).is_file():
                episodes.append(ep_dir)
    if not episodes:
        raise RuntimeError(f"No episodes with google_stt/{WORDS_FILE} under {dataset_root}")
    return episodes


def load_words(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in {path}, got {type(data)}")
    out: List[Dict] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def normalize_word(word: str) -> str:
    return " ".join(word.strip().lower().split())


def clean_word_entries(
    words: Sequence[Dict],
    dedup_digits: int = 3,
    ghost_ratio_thresh: float = 0.95,
) -> Tuple[List[Dict], int, int, Tuple[int, ...]]:
    groups: Dict[Tuple[int, int, str], List[Tuple[float, float, str, int]]] = {}
    zero_dropped = 0
    for raw in words:
        try:
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", 0.0))
        except Exception:
            continue
        if end <= start:
            zero_dropped += 1
            continue
        word = str(raw.get("word", ""))
        norm_word = normalize_word(word)
        try:
            speaker_raw = raw.get("speaker", raw.get("speakerTag", 0))
            speaker = int(speaker_raw)
        except Exception:
            speaker = 0
        key = (round(start, dedup_digits), round(end, dedup_digits), norm_word)
        groups.setdefault(key, []).append((start, end, word, speaker))

    if not groups:
        return [], zero_dropped, 0, tuple()

    group_count = len(groups)
    speaker_presence: Counter = Counter()
    for entries in groups.values():
        speaker_presence.update({sp for *_, sp in entries})
    ghost_speakers = tuple(
        sorted(sp for sp, cnt in speaker_presence.items() if cnt / group_count >= ghost_ratio_thresh)
    )

    cleaned: List[Dict] = []
    duplicate_dropped = 0

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        entries = groups[key]
        kept: Dict[int, Tuple[float, float, str]] = {}
        for start, end, word, speaker in entries:
            if speaker in ghost_speakers and len(entries) > 1:
                duplicate_dropped += 1
                continue
            if speaker in kept:
                duplicate_dropped += 1
                continue
            kept[speaker] = (start, end, word)
        if not kept:
            start, end, word, speaker = entries[0]
            kept[speaker] = (start, end, word)
        duplicate_dropped += max(0, len(entries) - len(kept))
        for speaker, (start, end, word) in kept.items():
            cleaned.append({
                "word": word,
                "start": start,
                "end": end,
                "speaker": speaker,
            })

    cleaned.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return cleaned, zero_dropped, duplicate_dropped, ghost_speakers


def renumber_speakers(words: List[Dict]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    next_id = 0
    for w in words:
        sp = int(w.get("speaker", 0))
        if sp not in mapping:
            mapping[sp] = next_id
            next_id += 1
        w["speaker"] = mapping[sp]
    return mapping


def words_to_wordinfo(words: Sequence[Dict]) -> List[WordInfo]:
    out: List[WordInfo] = []
    for w in words:
        out.append(
            WordInfo(
                word=str(w.get("word", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker=int(w.get("speaker", 0)),
            )
        )
    return out


def load_segments_csv(csv_path: Path) -> List[Dict]:
    if not csv_path.is_file():
        return []
    out: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row.get("start", 0.0))
                end = float(row.get("end", 0.0))
            except Exception:
                continue
            if end <= start:
                continue
            speaker = int(row.get("speaker", 0))
            text = row.get("text", "")
            out.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return out


def concurrency_stats(segments: Sequence[Dict]) -> Tuple[float, float, int]:
    events: List[Tuple[float, int]] = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        if e <= s:
            continue
        events.append((s, +1))
        events.append((e, -1))
    if not events:
        return 0.0, 0.0, 0
    events.sort(key=lambda x: (x[0], -x[1]))
    active = 0
    last_time: float | None = None
    total = 0.0
    overlap = 0.0
    acc_active = 0.0
    max_active = 0
    for t, delta in events:
        if last_time is not None and t > last_time:
            dur = t - last_time
            if active > 0:
                total += dur
                acc_active += active * dur
                if active >= 2:
                    overlap += dur
            if active > max_active:
                max_active = active
        active += delta
        if active > max_active:
            max_active = active
        last_time = t
    frac = overlap / total if total > 0 else 0.0
    mean = acc_active / total if total > 0 else 0.0
    return frac, mean, max_active


def write_segments_csv(segments: Sequence[Dict], csv_path: Path):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker", "text"])
        for seg in segments:
            writer.writerow([
                f"{float(seg['start']):.3f}",
                f"{float(seg['end']):.3f}",
                int(seg["speaker"]),
                seg.get("text", ""),
            ])


def save_words(words: Sequence[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "word": str(w.get("word", "")),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "speaker": int(w.get("speaker", 0)),
            }
            for w in words
        ], f, ensure_ascii=False, indent=2)


def process_episode(ep_dir: Path, create_backup: bool) -> CleanStats:
    gt_dir = ep_dir / EPS_GT_REL
    words_path = gt_dir / WORDS_FILE
    csv_path = gt_dir / CSV_FILE
    rttm_path = gt_dir / RTTM_FILE

    words_raw = load_words(words_path)
    original_segments = load_segments_csv(csv_path)
    speakers_before = len({seg["speaker"] for seg in original_segments})
    multi_before, mean_before, max_before = (0.0, 0.0, 0)
    if original_segments:
        multi_before, mean_before, max_before = concurrency_stats(original_segments)

    cleaned_words, zero_dropped, duplicate_dropped, ghost_speakers = clean_word_entries(words_raw)
    renumber_speakers(cleaned_words)
    word_infos = words_to_wordinfo(cleaned_words)
    cleaned_segments = words_to_segments(word_infos)

    seg_mapping: Dict[int, int] = {}
    next_sp = 0
    for seg in cleaned_segments:
        sp = int(seg.get("speaker", 0))
        if sp not in seg_mapping:
            seg_mapping[sp] = next_sp
            next_sp += 1
        seg["speaker"] = seg_mapping[sp]

    multi_after, mean_after, max_after = (0.0, 0.0, 0)
    if cleaned_segments:
        multi_after, mean_after, max_after = concurrency_stats(cleaned_segments)

    if create_backup:
        backup_path = words_path.with_suffix(".orig.json")
        if not backup_path.exists():
            with open(backup_path, "w", encoding="utf-8") as bf:
                json.dump(words_raw, bf, ensure_ascii=False, indent=2)

    save_words(cleaned_words, words_path)
    write_segments_csv(cleaned_segments, csv_path)
    _, _, file_id = episode_id_from_path(str(ep_dir))
    write_rttm(cleaned_segments, str(rttm_path), file_id)

    speakers_after = len({seg["speaker"] for seg in cleaned_segments})

    return CleanStats(
        words_before=len(words_raw),
        words_after=len(cleaned_words),
        zero_dropped=zero_dropped,
        duplicate_dropped=duplicate_dropped,
        ghost_speakers=ghost_speakers,
        speakers_before=speakers_before,
        speakers_after=speakers_after,
        segments_before=len(original_segments),
        segments_after=len(cleaned_segments),
        multi_frac_before=multi_before,
        multi_frac_after=multi_after,
        mean_concurrency_before=mean_before,
        mean_concurrency_after=mean_after,
        max_concurrency_before=max_before,
        max_concurrency_after=max_after,
    )


def main():
    parser = argparse.ArgumentParser(description="Clean Google STT diarization outputs in-place")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT,
                        help="Dataset root or single episode directory")
    parser.add_argument("--episode", type=Path, default=None,
                        help="Optional single-episode directory; overrides --dataset")
    parser.add_argument("--no_backup", action="store_true", help="Do not write words.orig.json backup")
    args = parser.parse_args()

    if args.episode is not None:
        episodes = [args.episode.resolve()]
    else:
        episodes = discover_episodes(args.dataset)

    total_stats: List[CleanStats] = []
    for ep in episodes:
        stats = process_episode(ep, create_backup=not args.no_backup)
        total_stats.append(stats)
        rel = ep.relative_to(args.dataset if args.episode is None else ep)
        ghost_str = ",".join(str(g) for g in stats.ghost_speakers) if stats.ghost_speakers else "-"
        print(
            f"[CLEAN] {rel}: words {stats.words_before}->{stats.words_after} "
            f"(zero {stats.zero_dropped}, drop {stats.duplicate_dropped}), "
            f"segments {stats.segments_before}->{stats.segments_after}, "
            f"speakers {stats.speakers_before}->{stats.speakers_after}, "
            f"ghost={ghost_str}, multi>=2 {stats.multi_frac_before:.3f}->{stats.multi_frac_after:.3f}, "
            f"meanAct {stats.mean_concurrency_before:.3f}->{stats.mean_concurrency_after:.3f}, "
            f"maxAct {stats.max_concurrency_before}->{stats.max_concurrency_after}"
        )

    if len(total_stats) > 1:
        def avg(vals: Iterable[float]) -> float:
            vals = list(vals)
            return float(sum(vals) / len(vals)) if vals else 0.0
        print("\nSummary across episodes:")
        print(
            f"  words: total_before={sum(s.words_before for s in total_stats)} "
            f"total_after={sum(s.words_after for s in total_stats)}"
        )
        print(
            f"  multi >=2: mean_before={avg(s.multi_frac_before for s in total_stats):.3f} "
            f"mean_after={avg(s.multi_frac_after for s in total_stats):.3f}"
        )
        print(
            f"  mean concurrency: mean_before={avg(s.mean_concurrency_before for s in total_stats):.3f} "
            f"mean_after={avg(s.mean_concurrency_after for s in total_stats):.3f}"
        )


if __name__ == "__main__":
    main()
