import os
import sys
import glob
import pickle
import argparse
import subprocess
import traceback
import numpy as np
from scipy import signal

# Import pipeline utilities; support running from repo root or whisperv/ cwd
# This module defines a global `args` object; we override its paths per-episode.
try:
    from whisperv import inference_folder as inf  # when run from repo root
    from whisperv.identity_verifier import IdentityVerifier
except ImportError:
    import inference_folder as inf  # when run from whisperv/ directory
    from identity_verifier import IdentityVerifier


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _ensure_frames(video_path, frames_dir, n_threads=4):
    os.makedirs(frames_dir, exist_ok=True)
    jpgs = glob.glob(os.path.join(frames_dir, '*.jpg'))
    if len(jpgs) > 0:
        return
    cmd = (
        f"ffmpeg -y -i {video_path} -qscale:v 2 -threads {n_threads} "
        f"-f image2 {os.path.join(frames_dir, '%06d.jpg')} -loglevel panic"
    )
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed extracting frames for {video_path}")


def _prepare_scores(result_dir):
    scores_p = os.path.join(result_dir, 'scores.pckl')
    if os.path.exists(scores_p):
        return scores_p
    smooth_p = os.path.join(result_dir, 'smooth_scores.pckl')
    if os.path.exists(smooth_p):
        # Reuse existing smooth scores as scores
        scores = _load_pickle(smooth_p)
        _save_pickle(scores, scores_p)
        return scores_p
    return None


def process_episode(episode_root, n_threads=4):
    # Expected structure: {episode_root}/{avi,crop,result[,frame]}
    avi_dir = os.path.join(episode_root, 'avi')
    crop_dir = os.path.join(episode_root, 'crop')
    result_dir = os.path.join(episode_root, 'result')
    frame_dir = os.path.join(episode_root, 'frame')

    def log(msg):
        print(f"[episode] {episode_root}: {msg}")
        try:
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, 'experiment_log.txt'), 'a') as lf:
                lf.write(str(msg) + "\n")
        except Exception:
            pass

    if not os.path.isdir(avi_dir):
        raise FileNotFoundError(f"Missing avi dir: {avi_dir}")
    if not os.path.isdir(crop_dir):
        raise FileNotFoundError(f"Missing crop dir: {crop_dir}")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Missing result dir: {result_dir}")

    video_path = os.path.join(avi_dir, 'video.avi')
    audio_path = os.path.join(avi_dir, 'audio.wav')
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Missing video file: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    # Ensure frames exist if we need visualization overlays
    log("ensure frames")
    _ensure_frames(video_path, frame_dir, n_threads=n_threads)

    # Configure inference_folder paths to reuse existing artifacts
    inf.args.pyaviPath = avi_dir
    inf.args.pyframesPath = frame_dir
    inf.args.pyworkPath = result_dir
    inf.args.pycropPath = crop_dir
    inf.args.videoFilePath = video_path
    inf.args.audioFilePath = audio_path
    inf.args.nDataLoaderThread = n_threads

    # Load required base artifacts
    log("load tracks.pckl")
    tracks_p = os.path.join(result_dir, 'tracks.pckl')
    if not os.path.exists(tracks_p):
        raise FileNotFoundError(f"Missing face tracks: {tracks_p}")
    vidTracks = _load_pickle(tracks_p)

    # Ensure vidTracks have cropFile and proc_track fields; map to existing crop clips
    log("scan crop/*.avi")
    crop_avis = sorted(glob.glob(os.path.join(crop_dir, '*.avi')))
    if not crop_avis:
        raise FileNotFoundError(f"No crop clips found in {crop_dir}")

    def _compute_proc_track(track):
        # Reproduce smoothing used in crop_video for visualization
        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:
            det = np.asarray(det)
            s = max((det[3] - det[1]), (det[2] - det[0])) / 2.0
            y = (det[1] + det[3]) / 2.0
            x = (det[0] + det[2]) / 2.0
            dets['s'].append(float(s))
            dets['y'].append(float(y))
            dets['x'].append(float(x))
        # Apply median filters
        if len(dets['s']) >= 13:
            dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
            dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
            dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
        return dets

    # Normalize structure to have 'track', 'proc_track', 'cropFile'
    normalized = []
    # Validate length compatibility if needed
    if isinstance(vidTracks, list) and len(vidTracks) == len(crop_avis):
        for i, t in enumerate(vidTracks):
            entry = {}
            if isinstance(t, dict) and 'track' in t:
                entry['track'] = t['track']
                entry['proc_track'] = t.get('proc_track') or _compute_proc_track(t['track'])
            elif isinstance(t, dict) and all(k in t for k in ('frame', 'bbox')):
                entry['track'] = t
                entry['proc_track'] = _compute_proc_track(t)
            else:
                raise RuntimeError("Unsupported tracks.pckl format: cannot find track frames/bboxes")
            # Assign cropFile base path (without extension, as expected by downstream code)
            base = os.path.splitext(crop_avis[i])[0]
            entry['cropFile'] = base
            normalized.append(entry)
        vidTracks = normalized
    else:
        # If structure already contains cropFile, trust it; else error
        if isinstance(vidTracks, list) and vidTracks and isinstance(vidTracks[0], dict) and 'cropFile' in vidTracks[0]:
            pass
        else:
            raise RuntimeError("tracks.pckl size does not match crop/*.avi count and no cropFile present")

    # Active Speaker Detection scores (compute before identities for ASD-gated embeddings)
    log("prepare/load ASD scores")
    scores_p = _prepare_scores(result_dir)
    if scores_p and os.path.exists(scores_p):
        scores = _load_pickle(scores_p)
    else:
        av_clips = sorted(glob.glob(os.path.join(crop_dir, '*.avi')))
        if len(av_clips) == 0:
            raise FileNotFoundError(f"No crop clips found in {crop_dir}")
        log("run TalkNet ASD")
        scores = inf.evaluate_network(av_clips, inf.args)
        scores_p = os.path.join(result_dir, 'scores.pckl')
        _save_pickle(scores, scores_p)

    # Identity assignment: cluster face tracks into stable visual identities (auto-K, constraints) with ASD gating
    id_tracks_p = os.path.join(result_dir, 'tracks_identity.pckl')
    log("identity assignment (visual constrained clustering)")
    if os.path.exists(id_tracks_p):
        annotated_tracks = _load_pickle(id_tracks_p)
    else:
        try:
            from .identity_cluster import cluster_visual_identities
        except Exception:
            from identity_cluster import cluster_visual_identities  # when run from whisperv/ cwd
        annotated_tracks = cluster_visual_identities(vidTracks, scores_list=scores)
        _save_pickle(annotated_tracks, id_tracks_p)

    # WhisperX transcription + alignment + diarization (constrained by speaker count)
    # Estimate speakers from two modalities, then take the smaller:
    #  - Visual identities (VID_*) count across all tracks
    #  - Audio-visual active identities: VID_* identities that have any ASD-positive frames
    vis_ids = set()
    for tr in annotated_tracks:
        ident = tr.get('identity')
        if isinstance(ident, str) and ident.startswith('VID_'):
            vis_ids.add(ident)
    K_vis = max(1, len(vis_ids))

    # Compute number of speaking identities from ASD scores
    active_ids = set()
    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity')
        if not (isinstance(ident, str) and ident.startswith('VID_')):
            continue
        if i >= len(scores):
            continue
        sc = scores[i]
        if sc is None or len(sc) == 0:
            continue
        # Mark identity active if any frame in this track has positive ASD score
        if any((s_val > 0) for s_val in sc):
            active_ids.add(ident)
    K_asd = len(active_ids) if active_ids else K_vis

    # Choose smaller K across modalities, then set diarization range
    K = max(1, min(K_vis, K_asd))
    # Policy: min = 1, max = K
    min_spk = 1
    max_spk = K
    log(f"whisperx diarization with min/max speakers = {min_spk}/{max_spk} (K={K}, K_vis={K_vis}, K_asd={K_asd})")
    raw_diar_p = os.path.join(result_dir, 'raw_diriazation_constrained.pckl')
    if os.path.exists(raw_diar_p):
        raw_segments = _load_pickle(raw_diar_p)
        raw_results = {"segments": raw_segments}
    else:
        raw_results = inf.speech_diarization(min_speakers=min_spk, max_speakers=max_spk)
        _save_pickle(raw_results["segments"], raw_diar_p)

    # Match speakers to visual identities and correct
    log("match speakers to visual identities")
    matched_p = os.path.join(result_dir, 'matched_diriazation.pckl')
    if os.path.exists(matched_p):
        corrected_results = _load_pickle(matched_p)
    else:
        matched_results = inf.match_speaker_identity(
            annotated_tracks, scores, raw_results["segments"], fps=25
        )
        corrected_results = inf.autofill_and_correct_matches(matched_results)
        _save_pickle(corrected_results, matched_p)

    # Visual-guided diarization refinement
    log("refining diarization with visual activity + identities")
    refined_p = os.path.join(result_dir, 'refined_diriazation.pckl')
    if os.path.exists(refined_p):
        refined_segments = _load_pickle(refined_p)
    else:
        refined_segments = inf.refine_diarization_with_visual(
            annotated_tracks, scores, raw_results["segments"], fps=25, tau=0.3, min_seg=0.08, merge_gap=0.2
        )
        _save_pickle(refined_segments, refined_p)

    # ASR-aligned boundary snapping refinement (DER-oriented)
    log("refining diarization by ASR-aligned boundary snapping")
    refined_asr_p = os.path.join(result_dir, 'refined_diriazation_asr.pckl')
    if os.path.exists(refined_asr_p):
        refined_asr_segments = _load_pickle(refined_asr_p)
    else:
        refined_asr_segments = inf.refine_diarization_boundaries(
            raw_results["segments"], pad=0.05, gap_split=0.25, min_seg=0.15, merge_gap=0.10
        )
        _save_pickle(refined_asr_segments, refined_asr_p)

    # Visualization and SRT subtitle generation (disabled by request)
    log("skip visualization + SRT generation")


def find_episodes(dataset_root):
    """Discover episodes under dataset_root.
    Supports both full-root (root -> show -> episode) and single-episode path.
    """
    # Single-episode directory: contains expected subfolders like 'avi' and 'result'
    if os.path.isdir(os.path.join(dataset_root, 'avi')) and os.path.isdir(os.path.join(dataset_root, 'result')):
        return [dataset_root]

    # Show/episode tree
    shows = []
    for d in sorted(os.listdir(dataset_root)):
        show_dir = os.path.join(dataset_root, d)
        if os.path.isdir(show_dir):
            shows.append(show_dir)
    episodes = []
    for show in shows:
        for e in sorted(os.listdir(show)):
            ep_dir = os.path.join(show, e)
            if os.path.isdir(ep_dir) and os.path.isdir(os.path.join(ep_dir, 'avi')) and os.path.isdir(os.path.join(ep_dir, 'result')):
                episodes.append(ep_dir)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Run WhisperV experiment pipeline on all episodes.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset",
        help="Root of the processed dataset (contains show folders)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for ffmpeg-related steps",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="HuggingFace token for diarization models (pyannote)",
    )
    args = parser.parse_args()

    # Propagate HF token to environment for downstream modules
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_TOKEN"] = args.hf_token

    # Distributed setup via torchrun (one process per GPU)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Constrain each process to a single GPU and set default device
    if world_size > 1:
        # Defer torch import until after environment is set
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    dataset_root = args.dataset_root
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    episodes = find_episodes(dataset_root)
    if len(episodes) == 0:
        raise RuntimeError("No episodes found under dataset root.")

    # Shard episodes across ranks (each process handles a disjoint subset)
    if world_size > 1:
        rank = int(os.environ.get("RANK", local_rank))
        episodes = episodes[rank::world_size]
        if not episodes:
            print(f"Rank {rank} has no episodes to process.")
            return

    failures = []
    for ep in episodes:
        try:
            print(f"Processing episode: {ep}")
            process_episode(ep, n_threads=args.threads)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR processing {ep}: {e}\n{tb}")
            # Also persist the traceback to episode's result dir
            try:
                with open(os.path.join(ep, 'result', 'error_last.txt'), 'w') as ef:
                    ef.write(str(e) + "\n" + tb)
            except Exception:
                pass
            failures.append((ep, f"{e}\n{tb}"))

    if failures:
        print("\nCompleted with failures:")
        for ep, err in failures:
            print(f"- {ep}: {err}")
        # In multi-GPU runs, avoid aborting the entire job; return success to let other ranks finish
        if world_size > 1:
            return
        else:
            sys.exit(1)
    else:
        print("\nAll episodes processed successfully.")


if __name__ == "__main__":
    main()
