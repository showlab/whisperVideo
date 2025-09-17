import os
import sys
import glob
import argparse
import json
import datetime
import numbers
from dataclasses import dataclass
from typing import List, Tuple


def sec_from_duration(d):
    """Convert a duration to seconds.

    Handles both protobuf Duration objects (with .seconds/.nanos)
    and Python datetime.timedelta, which newer google-cloud libraries
    may return via proto-plus wrappers.
    """
    # Case 1: protobuf Duration-like with seconds/nanos attributes
    if hasattr(d, "seconds"):
        seconds = float(getattr(d, "seconds", 0.0))
        nanos = float(getattr(d, "nanos", 0.0))
        return seconds + nanos / 1e9

    # Case 2: Python timedelta
    if isinstance(d, datetime.timedelta):
        return float(d.total_seconds())

    # Case 3: already numeric seconds
    if isinstance(d, numbers.Number):
        return float(d)

    raise TypeError(f"Unsupported duration type: {type(d)}; expected Duration or timedelta")


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    speaker: int


def gather_words_from_response(response) -> List[WordInfo]:
    words: List[WordInfo] = []
    for res in response.results:
        if not res.alternatives:
            continue
        alt = res.alternatives[0]
        if not getattr(alt, "words", None):
            continue
        for w in alt.words:
            if (getattr(w, "start_time", None) is None) or (getattr(w, "end_time", None) is None):
                continue
            st = sec_from_duration(w.start_time)
            et = sec_from_duration(w.end_time)
            speaker = int(getattr(w, "speaker_tag", 0) or 0)
            words.append(WordInfo(word=w.word, start=st, end=et, speaker=speaker))
    if not words:
        return words

    dedup_order: List[Tuple[str, float, float]] = []
    dedup_map: dict[Tuple[str, float, float], WordInfo] = {}

    for info in words:
        key = (info.word, info.start, info.end)
        existing = dedup_map.get(key)
        if existing is None:
            dedup_map[key] = info
            dedup_order.append(key)
            continue

        # Prefer diarized speaker tags (>0) over the placeholder 0.
        if existing.speaker == 0 and info.speaker != 0:
            dedup_map[key] = info
        elif existing.speaker != 0 and info.speaker == 0:
            continue
        else:
            dedup_map[key] = info

    deduped_words = [dedup_map[key] for key in dedup_order]
    deduped_words.sort(key=lambda x: (x.start, x.end))
    return deduped_words


def words_to_segments(words: List[WordInfo], max_gap: float = 0.5) -> List[dict]:
    segs = []
    if not words:
        return segs
    cur_speaker = words[0].speaker
    cur_start = words[0].start
    cur_end = words[0].end
    cur_text = [words[0].word]
    for w in words[1:]:
        gap = w.start - cur_end
        if w.speaker == cur_speaker and gap <= max_gap:
            cur_end = max(cur_end, w.end)
            cur_text.append(w.word)
        else:
            segs.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "speaker": int(cur_speaker),
                "text": " ".join(cur_text),
            })
            cur_speaker = w.speaker
            cur_start = w.start
            cur_end = w.end
            cur_text = [w.word]
    segs.append({
        "start": float(cur_start),
        "end": float(cur_end),
        "speaker": int(cur_speaker),
        "text": " ".join(cur_text),
    })
    return segs


def write_rttm(segments: List[dict], rttm_path: str, file_id: str):
    with open(rttm_path, "w") as f:
        for seg in segments:
            start = float(seg["start"])
            dur = float(seg["end"]) - float(seg["start"])
            spk = str(seg.get("speaker", "SPEAKER_XX"))
            line = f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            f.write(line)


def episode_id_from_path(ep_path: str) -> Tuple[str, str, str]:
    parts = ep_path.rstrip("/").split("/")
    show = parts[-2]
    episode = parts[-1]
    file_id = f"{show}_{episode}"
    return show, episode, file_id


def recognize_with_gcs(gcs_uri: str, language_code: str, min_speakers: int, max_speakers: int,
                        model: str, enhanced: bool):
    try:
        from google.cloud import speech
    except Exception as e:
        raise RuntimeError(f"google-cloud-speech is not installed or unavailable: {e}")

    client = speech.SpeechClient()

    diar_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=min_speakers if min_speakers else 2,
        max_speaker_count=max_speakers if max_speakers else 8,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code,
        enable_word_time_offsets=True,
        diarization_config=diar_config,
        use_enhanced=enhanced,
        model=model,
    )

    audio = speech.RecognitionAudio(uri=gcs_uri)
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()
    return response


def recognize_with_content(local_path: str, language_code: str, min_speakers: int, max_speakers: int,
                           model: str, enhanced: bool, encoding_override: str = None):
    try:
        from google.cloud import speech
    except Exception as e:
        raise RuntimeError(f"google-cloud-speech is not installed or unavailable: {e}")

    with open(local_path, "rb") as f:
        content = f.read()

    client = speech.SpeechClient()

    diar_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=min_speakers if min_speakers else 2,
        max_speaker_count=max_speakers if max_speakers else 8,
    )

    # Detect encoding from file extension if override not provided
    ext = os.path.splitext(local_path)[1].lower()
    if encoding_override:
        enc_name = encoding_override.upper()
    elif ext in (".wav", ".lin", ".pcm"):
        enc_name = "LINEAR16"
    elif ext == ".flac":
        enc_name = "FLAC"
    elif ext == ".mp3":
        enc_name = "MP3"
    else:
        # Let API autodetect if unknown
        enc_name = None

    config = speech.RecognitionConfig(
        encoding=getattr(speech.RecognitionConfig.AudioEncoding, enc_name) if enc_name else speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        language_code=language_code,
        enable_word_time_offsets=True,
        diarization_config=diar_config,
        use_enhanced=enhanced,
        model=model,
    )

    audio = speech.RecognitionAudio(content=content)
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()
    return response


def transcode_audio(src_wav: str, dst_path: str, codec: str) -> str:
    """Transcode using ffmpeg. codec: 'flac' or 'mp3'"""
    import subprocess as sp
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if codec == "flac":
        cmd = [
            "ffmpeg", "-y", "-i", src_wav,
            "-vn", "-ac", "1", "-ar", "16000", dst_path,
            "-loglevel", "error",
        ]
    elif codec == "mp3":
        cmd = [
            "ffmpeg", "-y", "-i", src_wav,
            "-vn", "-ac", "1", "-ar", "16000", "-codec:a", "libmp3lame", "-qscale:a", "4", dst_path,
            "-loglevel", "error",
        ]
    else:
        raise ValueError("codec must be 'flac' or 'mp3'")
    ret = sp.call(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {' '.join(cmd)}")
    return dst_path


def maybe_upload_to_gcs(local_path: str, bucket: str, dst_path: str) -> str:
    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError(f"google-cloud-storage is not installed or unavailable: {e}")

    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(dst_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{dst_path}"


def main():
    parser = argparse.ArgumentParser(description="Generate GT-like diarization via Google STT for all episodes.")
    parser.add_argument("--dataset_root", type=str,
                        default="/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset")
    parser.add_argument("--language_code", type=str, default="en-US")
    parser.add_argument("--min_speakers", type=int, default=2)
    parser.add_argument("--max_speakers", type=int, default=8)
    parser.add_argument("--model", type=str, default="video")
    parser.add_argument("--enhanced", action="store_true")

    parser.add_argument("--gcs_uri_template", type=str, default=None,
                        help="Template like gs://bucket/folder/{show}/{episode}/audio.wav")
    parser.add_argument("--upload_to_gcs", action="store_true")
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--gcs_prefix", type=str, default="whisperv_stt")
    parser.add_argument("--local_transcode", type=str, default="auto", choices=["auto","flac","mp3","none"],
                        help="When using local content, optionally transcode WAV to reduce payload (auto chooses flac then mp3 if needed)")
    parser.add_argument("--max_payload_mb", type=float, default=9.5,
                        help="If local content file exceeds this size, transcode when --local_transcode is not 'none'")

    args = parser.parse_args()

    def has_avi_dir(p: str) -> bool:
        return os.path.isdir(os.path.join(p, "avi")) and os.path.isfile(os.path.join(p, "avi", "audio.wav"))

    # Discover episodes robustly: support root -> show -> episode, show -> episode, or episode-only
    episodes = []
    root = args.dataset_root
    if has_avi_dir(root):
        episodes = [root]
    else:
        lvl1 = [p for p in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(p)]
        # If any lvl1 has avi/, treat lvl1 as episodes
        lvl1_episodes = [p for p in lvl1 if has_avi_dir(p)]
        if lvl1_episodes:
            episodes = lvl1_episodes
        else:
            # Otherwise look one level deeper
            for d in lvl1:
                lvl2 = [p for p in sorted(glob.glob(os.path.join(d, "*"))) if os.path.isdir(p)]
                for ep in lvl2:
                    if has_avi_dir(ep):
                        episodes.append(ep)

    if not episodes:
        raise RuntimeError(f"No episodes found under dataset_root: {args.dataset_root}. Expected structure like <episode>/avi/audio.wav")

    for ep in episodes:
        show, episode, file_id = episode_id_from_path(ep)
        avi_dir = os.path.join(ep, "avi")
        result_dir = os.path.join(ep, "result", "google_stt")
        os.makedirs(result_dir, exist_ok=True)
        audio_path = os.path.join(avi_dir, "audio.wav")
        if not os.path.isfile(audio_path):
            print(f"[Google STT] Missing audio: {audio_path}")
            continue

        seg_csv = os.path.join(result_dir, "segments.csv")
        rttm_path = os.path.join(result_dir, "segments.rttm")
        words_json = os.path.join(result_dir, "words.json")
        if os.path.isfile(seg_csv) and os.path.isfile(rttm_path) and os.path.isfile(words_json):
            print(f"[Google STT] Skipping (exists): {ep}")
            continue

        try:
            if args.gcs_uri_template:
                gcs_uri = args.gcs_uri_template.format(show=show, episode=episode)
                print(f"[Google STT] Recognize via GCS URI: {gcs_uri}")
                response = recognize_with_gcs(gcs_uri, args.language_code,
                                              args.min_speakers, args.max_speakers,
                                              args.model, args.enhanced)
            elif args.upload_to_gcs:
                if not args.gcs_bucket:
                    raise RuntimeError("--upload_to_gcs requires --gcs_bucket")
                dst_path = f"{args.gcs_prefix}/{show}/{episode}/audio.wav"
                gcs_uri = maybe_upload_to_gcs(audio_path, args.gcs_bucket, dst_path)
                print(f"[Google STT] Uploaded to {gcs_uri}, start recognition")
                response = recognize_with_gcs(gcs_uri, args.language_code,
                                              args.min_speakers, args.max_speakers,
                                              args.model, args.enhanced)
            else:
                # Local content: enforce payload limit by optional transcoding
                src = audio_path
                size_mb = os.path.getsize(src) / (1024*1024.0)
                enc_override = None
                if size_mb > args.max_payload_mb and args.local_transcode != "none":
                    out_dir = os.path.join(result_dir, "_tmp")
                    os.makedirs(out_dir, exist_ok=True)
                    if args.local_transcode in ("auto", "flac"):
                        dst = os.path.join(out_dir, "audio.flac")
                        try:
                            transcode_audio(src, dst, "flac")
                            if os.path.getsize(dst)/(1024*1024.0) <= args.max_payload_mb:
                                src = dst; enc_override = "FLAC"
                            else:
                                raise RuntimeError("FLAC still too large")
                        except Exception:
                            if args.local_transcode == "auto":
                                dst = os.path.join(out_dir, "audio.mp3")
                                transcode_audio(src, dst, "mp3")
                                src = dst; enc_override = "MP3"
                            else:
                                raise
                    elif args.local_transcode == "mp3":
                        dst = os.path.join(out_dir, "audio.mp3")
                        transcode_audio(src, dst, "mp3")
                        src = dst; enc_override = "MP3"

                print(f"[Google STT] Recognize via local content: {src}")
                response = recognize_with_content(src, args.language_code,
                                                  args.min_speakers, args.max_speakers,
                                                  args.model, args.enhanced,
                                                  encoding_override=enc_override)

            words = gather_words_from_response(response)
            with open(words_json, "w") as jf:
                json.dump([
                    {"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker}
                    for w in words
                ], jf)

            segs = words_to_segments(words)
            with open(seg_csv, "w") as cf:
                cf.write("start,end,speaker,text\n")
                for s in segs:
                    text = s["text"].replace("\n", " ").replace("\r", " ").replace(",", " ")
                    cf.write(f"{s['start']:.3f},{s['end']:.3f},{s['speaker']},{text}\n")

            write_rttm(segs, rttm_path, file_id)
            print(f"[Google STT] Wrote: {seg_csv}, {rttm_path}, {words_json}")

        except Exception as e:
            err_path = os.path.join(result_dir, "error_last.txt")
            with open(err_path, "w") as ef:
                ef.write(str(e))
            print(f"[Google STT] ERROR {ep}: {e}")


if __name__ == "__main__":
    main()
