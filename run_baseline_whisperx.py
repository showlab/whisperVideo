import os
import sys
import glob
import argparse
import pickle
import pysrt
import traceback
from datetime import timedelta

import pandas as pd
import whisperx


def seconds_to_srt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def generate_srt_from_segments(segments, output_srt_path: str):
    subs = pysrt.SubRipFile()

    for idx, segment in enumerate(segments):
        # Prefer segment-level speaker if present, else fallback to word-level majority speaker
        speaker = segment.get("speaker")
        if not speaker:
            # Tally word-level speakers if available
            word_speakers = {}
            for w in segment.get("words", []) or []:
                spk = w.get("speaker")
                if spk is None:
                    continue
                word_speakers[spk] = word_speakers.get(spk, 0) + (w.get("end", 0) - w.get("start", 0))
            if word_speakers:
                speaker = max(word_speakers.items(), key=lambda kv: kv[1])[0]
            else:
                speaker = "SPEAKER_XX"

        text = segment.get("text", "").strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))

        subs.append(
            pysrt.SubRipItem(
                index=idx + 1,
                start=seconds_to_srt_time(start),
                end=seconds_to_srt_time(end),
                text=f"{speaker}: {text}",
            )
        )

    subs.save(output_srt_path, encoding='utf-8')


def _to_diarize_df(diarize_segments):
    """Normalize diarization to a pandas DataFrame with [start, end, speaker].
    Supports pyannote.core.Annotation, list[dict], or already-DataFrame.
    """
    # Already a DataFrame
    if hasattr(diarize_segments, "__class__") and diarize_segments.__class__.__name__ == "DataFrame":
        return diarize_segments

    # pyannote Annotation -> rows
    try:
        itertracks = getattr(diarize_segments, "itertracks", None)
        if callable(itertracks):
            rows = []
            for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
                rows.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(speaker),
                })
            return pd.DataFrame(rows)
    except Exception:
        pass

    # list of dicts
    if isinstance(diarize_segments, (list, tuple)) and diarize_segments and isinstance(diarize_segments[0], dict):
        rows = []
        for d in diarize_segments:
            rows.append({
                "start": float(d.get("start", 0.0)),
                "end": float(d.get("end", 0.0)),
                "speaker": str(d.get("speaker", "SPEAKER_XX")),
            })
        return pd.DataFrame(rows)

    raise TypeError(f"Unsupported diarization type: {type(diarize_segments)}")


def process_episode_whisperx(episode_root: str, device: str = "cuda", batch_size: int = 16, compute_type: str = "float16", hf_token: str | None = None):
    avi_dir = os.path.join(episode_root, 'avi')
    result_dir = os.path.join(episode_root, 'result')
    if not os.path.isdir(avi_dir):
        raise FileNotFoundError(f"Missing avi dir: {avi_dir}")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Missing result dir: {result_dir}")

    audio_file = os.path.join(avi_dir, 'audio.wav')
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Missing audio file: {audio_file}")

    # Prepare output directory for baseline artifacts
    baseline_dir = os.path.join(result_dir, 'baseline_whisperx')
    os.makedirs(baseline_dir, exist_ok=True)
    seg_pkl = os.path.join(baseline_dir, 'segments.pkl')

    # Skip if already computed
    if os.path.isfile(seg_pkl):
        return {"segments_pkl": seg_pkl, "skipped": True}

    # 1) Transcribe
    model = whisperx.load_model("small", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2) Align
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3) Diarize whole audio
    try:
        diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token=hf_token)
        try:
            diarize_segments = diarize_model(audio)
        except Exception:
            diarize_segments = diarize_model(audio_file)
        diarize_df = _to_diarize_df(diarize_segments)
        result = whisperx.assign_word_speakers(diarize_df, result)
    except Exception:
        # Fallback to pyannote pipeline API if whisperx one not available
        try:
            from pyannote.audio import Pipeline
            diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            diarize_segments = diarize_model(audio_file)
            diarize_df = _to_diarize_df(diarize_segments)
            result = whisperx.assign_word_speakers(diarize_df, result)
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")

    # Persist segments for later evaluation
    with open(seg_pkl, 'wb') as f:
        pickle.dump(result["segments"], f)

    # Skip writing baseline SRT by request; keep segments for evaluation
    return {
        "segments_pkl": seg_pkl,
    }


def find_episodes(dataset_root: str):
    """Discover episodes robustly from a root that may be:
    - the dataset root containing show folders
    - a single show folder containing multiple episode folders
    - a single episode folder containing an `avi/` with `audio.wav`

    An episode is any directory that directly contains `avi/audio.wav`.
    """
    def has_avi_dir(p: str) -> bool:
        return os.path.isdir(os.path.join(p, 'avi')) and os.path.isfile(os.path.join(p, 'avi', 'audio.wav'))

    root = dataset_root

    # Case 1: dataset_root is already an episode dir
    if has_avi_dir(root):
        return [root]

    # Case 2: dataset_root is a show dir (its children are episodes)
    lvl1 = [p for p in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(p)]
    lvl1_episodes = [p for p in lvl1 if has_avi_dir(p)]
    if lvl1_episodes:
        return sorted(lvl1_episodes)

    # Case 3: dataset_root is the dataset dir (children are shows; grandchildren are episodes)
    episodes = []
    for show in lvl1:
        lvl2 = [p for p in sorted(glob.glob(os.path.join(show, '*'))) if os.path.isdir(p)]
        for ep in lvl2:
            if has_avi_dir(ep):
                episodes.append(ep)
    return sorted(episodes)


def main():
    parser = argparse.ArgumentParser(description="Run WhisperX baseline (audio-only) on episodes.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset",
        help=(
            "Root path which can be: dataset root (contains show folders), "
            "a single show folder (contains episode folders), or a single episode folder (contains avi/)."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    args = parser.parse_args()

    # Distributed setup via torchrun (one process per GPU)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    episodes = find_episodes(args.dataset_root)
    if not episodes:
        raise RuntimeError("No episodes found under dataset root")

    if world_size > 1:
        rank = int(os.environ.get("RANK", local_rank))
        episodes = episodes[rank::world_size]
        if not episodes:
            print(f"Rank {rank} has no episodes to process.")
            return

    failures = []
    for ep in episodes:
        try:
            print(f"[Baseline] Processing episode: {ep}")
            out = process_episode_whisperx(
                ep,
                device=args.device,
                batch_size=args.batch_size,
                compute_type=args.compute_type,
                hf_token=args.hf_token,
            )
            if isinstance(out, dict) and out.get("skipped"):
                print(f"[Baseline] Skipped (exists): {ep}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR processing {ep}: {e}\n{tb}")
            try:
                with open(os.path.join(ep, 'result', 'baseline_error_last.txt'), 'w') as ef:
                    ef.write(str(e) + "\n" + tb)
            except Exception:
                pass
            failures.append((ep, f"{e}\n{tb}"))

    if failures:
        print("\nBaseline completed with failures:")
        for ep, err in failures:
            print(f"- {ep}: {err}")
        # In multi-GPU runs, avoid aborting all ranks
        if world_size > 1:
            return
        else:
            sys.exit(1)
    else:
        print("\nBaseline finished successfully for all episodes.")


if __name__ == "__main__":
    main()
