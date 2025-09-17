import os
import sys
import json
import time
import glob
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Strictly no dummies/mocks: fail early on missing deps or credentials
try:
    from google.cloud import videointelligence_v1 as vi
except Exception as e:
    raise RuntimeError(
        "Missing dependency: google-cloud-videointelligence.\n"
        "Install it in your /home/siyuan/miniconda3/envs/whisperv env:\n"
        "  /home/siyuan/miniconda3/envs/whisperv/bin/python -m pip install google-cloud-videointelligence\n"
        f"Import error: {e}"
    )

import torch


# Constants consistent with the existing pipeline
FPS = 25.0


@dataclass
class Episode:
    show: str
    episode: str
    root: str  # absolute path to dataset root

    @property
    def ep_dir(self) -> str:
        return os.path.join(self.root, self.show, self.episode)

    @property
    def avi_dir(self) -> str:
        return os.path.join(self.ep_dir, "avi")

    @property
    def video_path(self) -> str:
        return os.path.join(self.avi_dir, "video.avi")

    @property
    def result_dir(self) -> str:
        return os.path.join(self.ep_dir, "result")

    def assert_layout(self) -> None:
        if not os.path.isdir(self.avi_dir):
            raise FileNotFoundError(f"Missing avi dir: {self.avi_dir}")
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Missing video file: {self.video_path}")
        os.makedirs(self.result_dir, exist_ok=True)


def find_episodes(dataset_root: str) -> List[Episode]:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    shows = [d for d in sorted(os.listdir(dataset_root)) if os.path.isdir(os.path.join(dataset_root, d))]
    episodes: List[Episode] = []
    for show in shows:
        show_dir = os.path.join(dataset_root, show)
        for e in sorted(os.listdir(show_dir)):
            ep_dir = os.path.join(show_dir, e)
            if os.path.isdir(ep_dir):
                episodes.append(Episode(show=show, episode=e, root=dataset_root))
    if not episodes:
        raise RuntimeError("No episodes found under dataset root")
    return episodes


def _duration_to_seconds(dur) -> float:
    # Supports google.protobuf.duration_pb2.Duration
    # Duration stores seconds + nanos
    sec = float(getattr(dur, "seconds", 0.0))
    nanos = float(getattr(dur, "nanos", 0.0))
    return sec + nanos / 1e9


def _episode_to_gcs_uri(ep: Episode, prefix: str) -> str:
    # Map local path relative to dataset_root to GCS path under prefix
    # Required layout parity: {prefix}/{show}/{episode}/avi/video.avi
    prefix = prefix.rstrip("/")
    return f"{prefix}/{ep.show}/{ep.episode}/avi/video.avi"


def _parse_gcs_uri_prefix(uri: str) -> Tuple[str, str]:
    """Split a gs://bucket/prefix into (bucket, prefix_without_leading_slash)."""
    if not uri.startswith("gs://"):
        raise ValueError("gcs_uri_prefix must start with gs://")
    rest = uri[5:]  # strip gs://
    parts = rest.split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    return bucket, path.rstrip("/")


def annotate_person_tracks(client: vi.VideoIntelligenceServiceClient,
                           *,
                           input_uri: Optional[str] = None,
                           input_content: Optional[bytes] = None,
                           timeout: int = 3600) -> List[Dict]:
    if not input_uri and not input_content:
        raise ValueError("Either input_uri or input_content must be provided")
    if input_uri and input_content:
        raise ValueError("Provide only one of input_uri or input_content")

    features = [vi.Feature.OBJECT_TRACKING]

    if input_uri:
        operation = client.annotate_video(input_uri=input_uri, features=features)
    else:
        operation = client.annotate_video(input_content=input_content, features=features)

    result = operation.result(timeout=timeout)
    if not result.annotation_results:
        raise RuntimeError("Empty annotation_results returned by GCP Video Intelligence")

    ann = result.annotation_results[0]
    tracks_out: List[Dict] = []
    skipped_no_timing = 0
    for idx, obj in enumerate(getattr(ann, "object_annotations", [])):
        # Only keep person
        ent = getattr(obj, "entity", None)
        label = getattr(ent, "description", "") if ent else ""
        if label.lower() != "person":
            continue

        # Prefer track_id if provided; else fall back to deterministic annotation index
        tid = getattr(obj, "track_id", None)
        if tid is not None and tid != "":
            id_str = f"GCP_T{tid}"
        else:
            id_str = f"GCP_A{idx}"

        segs: List[Tuple[float, float]] = []
        # Primary segment
        seg = getattr(obj, "segment", None)
        if seg is not None:
            s = _duration_to_seconds(getattr(seg, "start_time_offset", 0.0))
            e = _duration_to_seconds(getattr(seg, "end_time_offset", 0.0))
            if e > s:
                segs.append((s, e))
        # Fallback: infer from frames if segment missing
        if not segs:
            frames = list(getattr(obj, "frames", []))
            if frames:
                t0 = _duration_to_seconds(getattr(frames[0], "time_offset", 0.0))
                t1 = _duration_to_seconds(getattr(frames[-1], "time_offset", 0.0))
                if t1 > t0:
                    segs.append((t0, t1))

        # Collect frames (optional; not required for ID mapping)
        # We keep minimal payload: id + segments
        if not segs:
            # Rarely, an object annotation may lack both segment and frames timing.
            # Skip this object without fabricating data.
            skipped_no_timing += 1
            continue

        tracks_out.append({
            "id": id_str,
            "entity": label,
            "segments": segs,
        })

    # Optionally, log skipped count to stderr for transparency
    if skipped_no_timing > 0:
        sys.stderr.write(f"[GCP-ID] Skipped {skipped_no_timing} object(s) with no timing info.\n")

    return tracks_out


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def _load_tracks_pckl(tracks_pckl: str):
    import pickle
    with open(tracks_pckl, 'rb') as f:
        return pickle.load(f)


def _save_pickle(obj, path: str) -> None:
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def assign_gcp_identities_to_tracks(vidTracks: List[Dict], person_tracks: List[Dict], fps: float = FPS) -> List[Dict]:
    """Assign GCP person track IDs to existing face tracks by max time overlap.
    - vidTracks entries must have keys: 'track' with 'frame' list, and 'cropFile'
    - Output list mirrors input but adds 'identity' string or None.
    """
    out = []
    # Build episode-level person segments
    pts: List[Tuple[str, Tuple[float, float]]] = []
    for pt in person_tracks:
        tid = pt["id"]
        for s, e in pt["segments"]:
            if e > s:
                pts.append((tid, (float(s), float(e))))

    for t in vidTracks:
        if not isinstance(t, dict) or "track" not in t or "cropFile" not in t:
            raise RuntimeError("vidTracks entry missing required keys 'track' and 'cropFile'")
        frames = t["track"].get("frame")
        if frames is None or len(frames) == 0:
            # No frames → cannot assign ID; keep None strictly
            t2 = dict(t)
            t2["identity"] = None
            out.append(t2)
            continue
        s = min(frames) / fps
        e = (max(frames) + 1) / fps
        best_id = None
        best_ov = 0.0
        for tid, seg in pts:
            ov = _overlap((s, e), seg)
            if ov > best_ov:
                best_ov = ov
                best_id = tid
        t2 = dict(t)
        # Require at least 10% overlap of the track duration to accept assignment
        dur = max(1e-6, e - s)
        if best_id is not None and (best_ov / dur) >= 0.10:
            t2["identity"] = best_id
        else:
            t2["identity"] = None
        out.append(t2)
    return out


def process_episode_worker(ep: Episode,
                           mode: str,
                           gcs_prefix: Optional[str],
                           upload_to_gcs: bool,
                           gcs_bucket: Optional[str],
                           alt_gcs_prefix: Optional[str],
                           bytes_max_mb: int,
                           write_tracks_identity: bool,
                           overwrite_tracks_identity: bool,
                           timeout_s: int) -> Tuple[str, str]:
    """Process a single episode. Returns (episode_dir, status). Raises on hard errors."""
    ep.assert_layout()

    vi_client = vi.VideoIntelligenceServiceClient()

    # Prepare input
    input_uri = None
    input_content = None
    if mode == "gcs":
        if upload_to_gcs:
            # Determine bucket + prefix
            if gcs_prefix:
                bkt, base = _parse_gcs_uri_prefix(gcs_prefix)
            else:
                if not gcs_bucket or not alt_gcs_prefix:
                    raise RuntimeError("When --upload_to_gcs is set, provide either --gcs_uri_prefix or both --gcs_bucket and --gcs_prefix")
                bkt, base = gcs_bucket, alt_gcs_prefix.strip("/")

            dst_path = f"{base}/{ep.show}/{ep.episode}/avi/video.avi" if base else f"{ep.show}/{ep.episode}/avi/video.avi"
            # Upload local file to GCS
            try:
                from google.cloud import storage
            except Exception as e:
                raise RuntimeError(
                    "Missing dependency: google-cloud-storage required for --upload_to_gcs.\n"
                    "Install: /home/siyuan/miniconda3/envs/whisperv/bin/python -m pip install google-cloud-storage\n"
                    f"Import error: {e}"
                )
            storage_client = storage.Client()
            storage_bucket = storage_client.bucket(bkt)
            blob = storage_bucket.blob(dst_path)
            blob.upload_from_filename(ep.video_path)
            input_uri = f"gs://{bkt}/{dst_path}"
        else:
            if not gcs_prefix:
                raise RuntimeError("--gcs_uri_prefix is required for input_mode=gcs (or use --upload_to_gcs)")
            input_uri = _episode_to_gcs_uri(ep, gcs_prefix)
    elif mode == "bytes":
        fsz = os.path.getsize(ep.video_path)
        if fsz > bytes_max_mb * 1024 * 1024:
            raise RuntimeError(
                f"Video too large for bytes mode: {fsz/1024/1024:.1f} MB > {bytes_max_mb} MB. Use --input_mode gcs."
            )
        with open(ep.video_path, "rb") as f:
            input_content = f.read()
    else:
        raise ValueError("input_mode must be 'gcs' or 'bytes'")

    # Annotate
    person_tracks = annotate_person_tracks(vi_client, input_uri=input_uri, input_content=input_content, timeout=timeout_s)

    # Persist raw GCP results (JSON)
    out_json = os.path.join(ep.result_dir, "gcp_visual_identity.json")
    payload = {
        "episode": {"show": ep.show, "episode": ep.episode},
        "video": os.path.relpath(ep.video_path, ep.root),
        "input_mode": mode,
        "tracks": person_tracks,
        "generated_at": int(time.time()),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f)

    # Optionally assign to existing face tracks
    if write_tracks_identity:
        tp = os.path.join(ep.result_dir, "tracks.pckl")
        if not os.path.exists(tp):
            raise FileNotFoundError(f"Cannot assign identities: missing {tp}")
        vidTracks = _load_tracks_pckl(tp)

        # Normalize structure: ensure entries have 'track' and 'cropFile'
        norm = []
        # If vidTracks already contains dicts with cropFile, keep them
        if isinstance(vidTracks, list) and vidTracks and isinstance(vidTracks[0], dict) and "cropFile" in vidTracks[0]:
            norm = vidTracks
        else:
            # Attempt to align crop/*.avi to track order
            crop_avis = sorted(glob.glob(os.path.join(ep.ep_dir, "crop", "*.avi")))
            if not crop_avis:
                raise FileNotFoundError(f"No crop clips found in {os.path.join(ep.ep_dir, 'crop')}")
            if not isinstance(vidTracks, list) or len(vidTracks) != len(crop_avis):
                raise RuntimeError("tracks.pckl format/count does not match crop/*.avi files")
            for i, t in enumerate(vidTracks):
                if not isinstance(t, dict):
                    raise RuntimeError("Unsupported tracks.pckl entry (expected dict)")
                if "track" in t:
                    tr = t["track"]
                elif all(k in t for k in ("frame", "bbox")):
                    tr = t
                else:
                    raise RuntimeError("tracks.pckl missing per-track 'track' or ('frame','bbox') keys")
                base = os.path.splitext(crop_avis[i])[0]
                norm.append({"track": tr, "cropFile": base})

        annotated = assign_gcp_identities_to_tracks(norm, person_tracks, fps=FPS)

        out_name = "tracks_identity_gcp.pckl"
        if overwrite_tracks_identity:
            out_name = "tracks_identity.pckl"
        _save_pickle(annotated, os.path.join(ep.result_dir, out_name))

    return (ep.ep_dir, "ok")


def _worker(payload):
    (
        ep,
        mode,
        gcs_prefix,
        upload_to_gcs,
        gcs_bucket,
        alt_gcs_prefix,
        bytes_max_mb,
        write_id,
        overwrite_id,
        timeout_s,
    ) = payload
    return process_episode_worker(
        ep,
        mode,
        gcs_prefix,
        upload_to_gcs,
        gcs_bucket,
        alt_gcs_prefix,
        bytes_max_mb,
        write_id,
        overwrite_id,
        timeout_s,
    )


def main():
    parser = argparse.ArgumentParser(description="Label per-episode visual identities using Google Video Intelligence (OBJECT_TRACKING).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root of dataset (e.g., /workspace/.../multi_human_talking_dataset)",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["gcs", "bytes"],
        required=True,
        help="Use 'gcs' with --gcs_uri_prefix (recommended), or 'bytes' for small videos",
    )
    parser.add_argument(
        "--gcs_uri_prefix",
        type=str,
        default=None,
        help="GCS prefix mapping dataset root (e.g., gs://bucket/multi_human_talking_dataset)",
    )
    parser.add_argument(
        "--upload_to_gcs",
        action="store_true",
        help="Upload local videos to GCS before annotation (auto-computes per-episode path)",
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default=None,
        help="Target GCS bucket name if --upload_to_gcs and --gcs_uri_prefix not provided",
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default=None,
        help="Target GCS prefix if --upload_to_gcs and --gcs_uri_prefix not provided",
    )
    parser.add_argument(
        "--bytes_max_mb",
        type=int,
        default=150,
        help="Hard limit for input_mode=bytes; above this, raise error",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel episode workers (torch.multiprocessing)",
    )
    parser.add_argument(
        "--write_tracks_identity",
        action="store_true",
        help="Also map GCP person IDs to face tracks and save tracks_identity_gcp.pckl",
    )
    parser.add_argument(
        "--overwrite_tracks_identity",
        action="store_true",
        help="Write to tracks_identity.pckl instead of *_gcp.pckl (careful: overwrites)",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=3600,
        help="Per-episode annotate_video timeout seconds",
    )

    args = parser.parse_args()

    # Enforce credentials presence (no silent fallbacks)
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred or not os.path.isfile(cred):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS not set or file missing. Export it before running."
        )

    # Enumerate episodes
    episodes = find_episodes(args.dataset_root)

    # Sequential processing (align with google_stt_gt.py behavior)
    total = len(episodes)
    ok = 0
    failures: List[Tuple[str, str]] = []
    for ep in episodes:
        try:
            ep_dir, status = process_episode_worker(
                ep,
                args.input_mode,
                args.gcs_uri_prefix,
                bool(args.upload_to_gcs),
                args.gcs_bucket,
                args.gcs_prefix,
                int(args.bytes_max_mb),
                bool(args.write_tracks_identity),
                bool(args.overwrite_tracks_identity),
                int(args.timeout_s),
            )
            print(f"[GCP-ID] {ep_dir}: {status}")
            ok += 1
        except Exception as e:
            # Persist episode-local error log and continue
            err = f"{e}"
            try:
                os.makedirs(os.path.join(ep.ep_dir, 'result'), exist_ok=True)
                with open(os.path.join(ep.ep_dir, 'result', 'error_last.txt'), 'w') as ef:
                    ef.write(err)
            except Exception:
                pass
            failures.append((ep.ep_dir, err))
            print(f"[GCP-ID][ERROR] {ep.ep_dir}: {e}")

    if failures:
        print(f"Completed {ok}/{total} episodes; failures: {len(failures)}")
        for ep_dir, err in failures:
            print(f" - {ep_dir}: {err}")
        # Keep exit code 1 to surface failures (no silent fallback)
        sys.exit(1)
    else:
        print(f"All {total} episodes labeled successfully.")


if __name__ == "__main__":
    # Use the specified Python interpreter when invoking manually:
    #   /home/siyuan/miniconda3/envs/whisperv/bin/python whisperv/gcp_visual_identity_label.py ...
    main()
