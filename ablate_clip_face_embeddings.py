#!/home/siyuan/miniconda3/envs/whisperv/bin/python
import os
import sys
import glob
import pickle
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

# Respect the repo guidance: no mock/fallback results.
# We accept either open_clip or clip as backends; if neither is available or
# model weights are unavailable, we raise a hard error.


class ClipFaceEmbedder:
    def __init__(
        self,
        device: str = "cuda",
        backend: str = "auto",   # one of: auto, open_clip, clip
        model: Optional[str] = None,
        pretrained: Optional[str] = None,
        batch_size: int = 32,
        max_samples: int = 15,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for CLIP but no GPU is available.")

        self.device = torch.device(device if device == "cuda" else "cpu")
        self.batch_size = int(batch_size)
        self.max_samples = int(max_samples)

        # Lazy imports with strict failure if neither backend is present
        self.backend = backend
        self.model = None
        self.preprocess = None

        # Backend selection
        if backend not in {"auto", "open_clip", "clip"}:
            raise ValueError(f"Invalid backend: {backend}")

        open_clip = None
        clip = None
        if backend in {"auto", "open_clip"}:
            try:
                import open_clip as _open_clip  # type: ignore
                open_clip = _open_clip
            except Exception:
                if backend == "open_clip":
                    raise RuntimeError("open_clip backend requested but 'open_clip' is not installed.")
        if backend in {"auto", "clip"}:
            try:
                import clip as _clip  # type: ignore
                clip = _clip
            except Exception:
                if backend == "clip":
                    raise RuntimeError("clip backend requested but 'clip' (openai-clip) is not installed.")

        # Initialize model and preprocess transforms
        if backend == "open_clip" or (backend == "auto" and open_clip is not None):
            # Defaults for open_clip
            model_name = model or "ViT-B-32"
            pretrained_tag = pretrained or "laion2b_s34b_b79k"
            try:
                mdl, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load open_clip model '{model_name}' pretrained='{pretrained_tag}'."
                    " Ensure weights are available locally or install the model properly."
                ) from e
            self.model = mdl.to(self.device).eval()
            self.preprocess = preprocess
            self.backend = "open_clip"

        elif backend == "clip" or (backend == "auto" and clip is not None):
            # Defaults for openai-clip
            model_name = model or "ViT-B/32"
            try:
                mdl, preprocess = clip.load(model_name, device=str(self.device))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load clip model '{model_name}'. Ensure weights are available/cached."
                ) from e
            self.model = mdl.eval()
            self.preprocess = preprocess
            self.backend = "clip"
        else:
            raise RuntimeError(
                "No CLIP backend available. Install one of: 'open_clip' or 'clip'."
            )

    def _sample_positions(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return []
        # Spread-out positions similar to IdentityVerifier default behavior
        positions = [
            0,
            total_frames // 4,
            total_frames // 2,
            (3 * total_frames) // 4,
            max(0, total_frames - 1),
        ]
        # Trim to max_samples uniformly if necessary
        if len(positions) > self.max_samples:
            step = max(1, len(positions) // self.max_samples)
            positions = positions[::step][: self.max_samples]
        return sorted(set(int(min(max(0, p), total_frames - 1)) for p in positions))

    def _load_key_frames(self, track_file: str, frame_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(track_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
        
        positions = self._sample_positions(total_frames) if not frame_indices else frame_indices
        tensors: List[torch.Tensor] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                tensor = self.preprocess(transforms.ToPILImage()(frame_rgb))  # 3xHxW float tensor
            except Exception:
                continue
            tensors.append(tensor)
        cap.release()
        return tensors

    @torch.no_grad()
    def _embed_batch(self, batch_tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not batch_tensors:
            return None
        x = torch.stack(batch_tensors, dim=0).to(self.device)
        # open_clip and openai-clip both expose encode_image
        feats = self.model.encode_image(x)
        feats = F.normalize(feats, p=2, dim=1)
        return feats

    def track_embedding(self, track_file: str, frame_indices: Optional[List[int]] = None) -> Optional[torch.Tensor]:
        frames = self._load_key_frames(track_file, frame_indices)
        if not frames:
            return None
        embeds_all: List[torch.Tensor] = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            emb = self._embed_batch(batch)
            if emb is not None:
                embeds_all.append(emb)
        if not embeds_all:
            return None
        E = torch.cat(embeds_all, dim=0)  # (K, D)
        e = E.mean(dim=0, keepdim=True)
        e = F.normalize(e, p=2, dim=1)  # (1, D)
        return e


def _load_tracks_identity(pywork_dir: str) -> List[dict]:
    f = os.path.join(pywork_dir, "tracks_identity.pckl")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Run whisperv/inference_folder.py to generate identities first.")
    with open(f, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"tracks_identity.pckl at {f} is empty or malformed.")
    return data


def _gather_episodes(root: str) -> List[Tuple[str, str]]:
    """
    Find all episodes that contain both 'pycrop' and 'pywork/tracks_identity.pckl'.
    Returns list of tuples: (episode_dir, pywork_dir)
    """
    episodes: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "pywork":
            parent = os.path.dirname(dirpath)
            pycrop = os.path.join(parent, "pycrop")
            id_file = os.path.join(dirpath, "tracks_identity.pckl")
            if os.path.exists(pycrop) and os.path.exists(id_file):
                episodes.append((parent, dirpath))
    if not episodes:
        raise RuntimeError(
            f"No episodes with pycrop and tracks_identity.pckl found under {root}."
            " Ensure you have run inference to generate face crops and identities."
        )
    return sorted(episodes)


def _compute_episode_embeddings(
    pywork_dir: str,
    clip_embedder: ClipFaceEmbedder,
    facenet_embedder,  # IdentityVerifier instance
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[str]]]:
    """
    Compute per-track embeddings for CLIP and Facenet for one episode.
    Returns (E_clip [N,Dc], E_face [N,Df], labels [len N]) where labels are identities (can be None).
    """
    tracks = _load_tracks_identity(pywork_dir)
    crop_files: List[str] = []
    labels: List[Optional[str]] = []
    for t in tracks:
        crop_base = t.get("cropFile")
        if not crop_base:
            continue
        crop_avi = crop_base + ".avi"
        if not os.path.exists(crop_avi):
            # Try absolute path join in case relative paths are stored
            crop_avi = os.path.join(os.path.dirname(pywork_dir), "pycrop", os.path.basename(crop_base) + ".avi")
            if not os.path.exists(crop_avi):
                raise FileNotFoundError(f"Crop video not found: {crop_base}.avi")
        crop_files.append(crop_avi)
        labels.append(t.get("identity"))

    if not crop_files:
        raise RuntimeError(f"No crop files found for episode at {pywork_dir}")

    # Compute embeddings
    E_clip: List[torch.Tensor] = []
    E_face: List[torch.Tensor] = []

    for cf in crop_files:
        e_clip = clip_embedder.track_embedding(cf)
        if e_clip is None:
            raise RuntimeError(f"Failed to compute CLIP embedding for {cf}")
        E_clip.append(e_clip)

        e_face = facenet_embedder._track_embedding(cf)
        if e_face is None:
            raise RuntimeError(f"Failed to compute Facenet embedding for {cf}")
        E_face.append(e_face)

    E_clip_t = torch.cat(E_clip, dim=0)
    E_face_t = torch.cat(E_face, dim=0)
    return E_clip_t, E_face_t, labels


def _pairwise_stats(E: torch.Tensor, labels: List[Optional[str]]):
    """Compute pairwise cosine similarities and summary stats for same vs different identities.
    Returns dict with counts and means.
    """
    if E.numel() == 0 or E.size(0) < 2:
        raise RuntimeError("Not enough embeddings to compute pairwise stats.")
    # Ensure normalized
    E = F.normalize(E, p=2, dim=1)
    S = E @ E.T  # (N,N)

    # Collect upper-triangular pairs
    L = labels
    same_vals: List[float] = []
    diff_vals: List[float] = []
    n = E.size(0)
    for i in range(n):
        li = L[i]
        if li is None:
            continue
        for j in range(i + 1, n):
            lj = L[j]
            if lj is None:
                continue
            v = float(S[i, j].item())
            if li == lj:
                same_vals.append(v)
            else:
                diff_vals.append(v)

    if not same_vals or not diff_vals:
        raise RuntimeError("Insufficient same/different identity pairs to compute stats.")

    same_t = torch.tensor(same_vals)
    diff_t = torch.tensor(diff_vals)
    return {
        "n_same": int(same_t.numel()),
        "n_diff": int(diff_t.numel()),
        "mean_same": float(same_t.mean().item()),
        "mean_diff": float(diff_t.mean().item()),
        "median_same": float(same_t.median().item()),
        "median_diff": float(diff_t.median().item()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation: CLIP vs Facenet face embeddings similarity")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset",
        help="Root directory containing episodes (must include pycrop and pywork/tracks_identity.pckl)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP embedding")
    parser.add_argument("--max_samples", type=int, default=15, help="Max frames sampled per track for embedding")
    parser.add_argument(
        "--clip_backend",
        type=str,
        default="auto",
        choices=["auto", "open_clip", "clip"],
        help="CLIP backend to use",
    )
    parser.add_argument("--clip_model", type=str, default=None, help="Model name (backend-specific)")
    parser.add_argument("--clip_pretrained", type=str, default=None, help="Pretrained tag (open_clip only)")
    args = parser.parse_args()

    # Gather episodes
    episodes = _gather_episodes(args.dataset_root)
    print(f"Found {len(episodes)} episodes with crops + identities under: {args.dataset_root}")

    # Initialize embedders
    from identity_verifier import IdentityVerifier  # Facenet baseline
    fn_embedder = IdentityVerifier(device=args.device, batch_size=args.batch_size)
    clip_embedder = ClipFaceEmbedder(
        device=args.device,
        backend=args.clip_backend,
        model=args.clip_model,
        pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Aggregate stats
    agg_same_clip = []
    agg_diff_clip = []
    agg_same_face = []
    agg_diff_face = []

    for ep_dir, pywork_dir in episodes:
        rel = os.path.relpath(ep_dir, args.dataset_root)
        print(f"Processing episode: {rel}")

        E_clip, E_face, labels = _compute_episode_embeddings(pywork_dir, clip_embedder, fn_embedder)

        # Per-episode stats
        s_clip = _pairwise_stats(E_clip, labels)
        s_face = _pairwise_stats(E_face, labels)
        print(
            f"  CLIP   - n_same={s_clip['n_same']}, n_diff={s_clip['n_diff']}, "
            f"mean_same={s_clip['mean_same']:.4f}, mean_diff={s_clip['mean_diff']:.4f}"
        )
        print(
            f"  Facenet- n_same={s_face['n_same']}, n_diff={s_face['n_diff']}, "
            f"mean_same={s_face['mean_same']:.4f}, mean_diff={s_face['mean_diff']:.4f}"
        )

        # For global summary, collect raw pairwise values
        # Recompute pairwise to collect lists
        def collect_pairs(E: torch.Tensor, L: List[Optional[str]]):
            E = F.normalize(E, p=2, dim=1)
            S = E @ E.T
            same_vals = []
            diff_vals = []
            n = E.size(0)
            for i in range(n):
                li = L[i]
                if li is None:
                    continue
                for j in range(i + 1, n):
                    lj = L[j]
                    if lj is None:
                        continue
                    v = float(S[i, j].item())
                    if li == lj:
                        same_vals.append(v)
                    else:
                        diff_vals.append(v)
            return same_vals, diff_vals

        s_vals_c, d_vals_c = collect_pairs(E_clip, labels)
        s_vals_f, d_vals_f = collect_pairs(E_face, labels)
        agg_same_clip.extend(s_vals_c)
        agg_diff_clip.extend(d_vals_c)
        agg_same_face.extend(s_vals_f)
        agg_diff_face.extend(d_vals_f)

    # Global summary across all episodes
    if not agg_same_clip or not agg_diff_clip or not agg_same_face or not agg_diff_face:
        raise RuntimeError("Insufficient pairs aggregated across episodes; cannot summarize.")

    clip_same = torch.tensor(agg_same_clip)
    clip_diff = torch.tensor(agg_diff_clip)
    face_same = torch.tensor(agg_same_face)
    face_diff = torch.tensor(agg_diff_face)

    print("\n=== Global Summary (All Episodes) ===")
    print(
        "CLIP   : "
        f"n_same={clip_same.numel()}, n_diff={clip_diff.numel()}, "
        f"mean_same={clip_same.mean().item():.4f}, mean_diff={clip_diff.mean().item():.4f}, "
        f"median_same={clip_same.median().item():.4f}, median_diff={clip_diff.median().item():.4f}"
    )
    print(
        "Facenet: "
        f"n_same={face_same.numel()}, n_diff={face_diff.numel()}, "
        f"mean_same={face_same.mean().item():.4f}, mean_diff={face_diff.mean().item():.4f}, "
        f"median_same={face_same.median().item():.4f}, median_diff={face_diff.median().item():.4f}"
    )


if __name__ == "__main__":
    main()

