import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import os
# Embedding backends: default to MagFace; allow override via IDENTITY_EMBEDDER env
def _build_embedder(device: str = "cuda", batch_size: int = 16):
    backend = os.environ.get("IDENTITY_EMBEDDER", "magface").strip().lower()
    if backend == "magface":
        try:
            from .embedders.magface_embedder import MagFaceEmbedder
        except Exception:
            from embedders.magface_embedder import MagFaceEmbedder
        return MagFaceEmbedder(device=device, batch_size=batch_size, backbone=os.environ.get("MAGFACE_BACKBONE", "iresnet100"))
    elif backend == "facenet":
        try:
            from .identity_verifier import IdentityVerifier
        except Exception:
            from identity_verifier import IdentityVerifier
        return IdentityVerifier(device=device, batch_size=batch_size)
    else:
        raise RuntimeError(f"Unsupported IDENTITY_EMBEDDER backend: {backend}")


def _frames_to_bbox_map(track: Dict) -> Dict[int, Tuple[float, float, float, float]]:
    frames = track["track"]["frame"]
    bboxes = track["track"]["bbox"]
    # frames/bboxes can be numpy arrays; ensure Python types
    frames_list = frames.tolist() if hasattr(frames, "tolist") else list(frames)
    bboxes_list = bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
    return {int(f): tuple(map(float, bb)) for f, bb in zip(frames_list, bboxes_list)}


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


@torch.no_grad()
def cluster_visual_identities(
    vidTracks: List[Dict],
    device: str = "cuda",
    batch_size: int = 16,
    face_sim_thresh: float = 0.60,
    must_iou_thresh: float = 0.70,
    cannot_iou_thresh: float = 0.10,
    min_overlap_frames: int = 3,
    scores_list: Optional[List[List[float]]] = None,
    # ASD gating parameters (frame-level score threshold, min consecutive frames, min ratio over track)
    asd_score_thresh: float = 0.20,
    asd_min_consec: int = 8,
    asd_min_ratio: float = 0.10,
) -> List[Dict]:
    """Cluster face tracks into stable episode-level identities using only visual cues.

    - Embeddings: facenet via IdentityVerifier (averaged key-frames per cropFile)
    - Must-link: overlapping-in-time with high bbox IoU (same face duplicated), force union
    - Cannot-link: co-visible with low IoU (different persons), forbid merging
    - Greedy agglomerative merging by cosine similarity with threshold and constraints

    Returns a new list mirroring vidTracks but 'identity' set to a stable label: 'Person_1', ...
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 1) Per-track embedding (single prototype per track)
    embedder = _build_embedder(device=device, batch_size=batch_size)
    embs: List[Optional[torch.Tensor]] = []  # (1,D) per track or None
    valid_idx: List[int] = []
    # Scheme A: skip tracks with no positive ASD frames when scores_list provided
    include_track = [True] * len(vidTracks)
    if scores_list is not None:
        include_track = []
        for i in range(len(vidTracks)):
            sc = scores_list[i] if i < len(scores_list) else None
            if not (isinstance(sc, (list, tuple)) and len(sc) > 0):
                include_track.append(False)
                continue
            # Apply stricter speaking criteria:
            #  - frame-level threshold: score > asd_score_thresh
            #  - require BOTH: consecutive positives >= asd_min_consec AND ratio >= asd_min_ratio
            pos = [1 if (float(v) > asd_score_thresh) else 0 for v in sc]
            if not any(pos):
                include_track.append(False)
                continue
            # longest consecutive ones
            max_run = 0
            run = 0
            for b in pos:
                if b:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
            ratio = float(sum(pos)) / max(1, len(pos))
            inc = (max_run >= asd_min_consec) and (ratio >= asd_min_ratio)
            include_track.append(inc)
    for i, tr in enumerate(vidTracks):
        crop_file = tr.get("cropFile")
        if not crop_file or not include_track[i]:
            track_sets.append(None)
            continue
        # Optional ASD gating: use positive-score frame indices if provided
        active_idx = None
        if scores_list is not None and i < len(scores_list):
            sc = scores_list[i]
            if isinstance(sc, (list, tuple)) and len(sc) > 0:
                idx = [k for k, v in enumerate(sc) if float(v) > asd_score_thresh]
                if len(idx) > 0:
                    active_idx = idx
        # Identity backends expose a common method name
        if hasattr(embedder, "track_embedding"):
            emb = embedder.track_embedding(crop_file + ".avi", active_indices=active_idx)
        else:
            emb = embedder._track_embedding(crop_file + ".avi", active_indices=active_idx)
        if emb is None:
            embs.append(None)
            continue
        embs.append(emb)  # (1, D)
        valid_idx.append(i)

    if not valid_idx:
        raise RuntimeError("No valid embeddings computed for any track; cannot cluster identities.")

    # 2) Build time overlap + IoU maps
    frame_maps = [_frames_to_bbox_map(tr) for tr in vidTracks]
    frames_sets = [set(m.keys()) for m in frame_maps]

    n = len(vidTracks)
    dsu = _DSU(n)
    cannot = set()  # set of frozenset({i,j})

    for i in range(n):
        if embs[i] is None or not include_track[i]:
            continue
        for j in range(i + 1, n):
            if embs[j] is None or not include_track[j]:
                continue
            # time-overlap frames
            overlap_frames = frames_sets[i].intersection(frames_sets[j])
            if len(overlap_frames) < min_overlap_frames:
                continue
            # mean IoU over overlapped frames
            ious = []
            fmap_i = frame_maps[i]
            fmap_j = frame_maps[j]
            for f in overlap_frames:
                iou = _iou(fmap_i[f], fmap_j[f])
                ious.append(iou)
            if not ious:
                continue
            miou = float(np.mean(ious))
            if miou >= must_iou_thresh:
                dsu.union(i, j)
            elif miou <= cannot_iou_thresh:
                cannot.add(frozenset((i, j)))

    # 3) Initialize groups by must-link unions
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        if embs[i] is None or not include_track[i]:
            continue
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)
    group_ids = list(groups.keys())
    # 4) Group embeddings (mean of member embs)
    grp_embs: Dict[int, torch.Tensor] = {}
    for gid, members in groups.items():
        vecs = [embs[m] for m in members if embs[m] is not None]
        if not vecs:
            raise RuntimeError("Encountered a group without embeddings; aborting to avoid fake data.")
        V = torch.cat(vecs, dim=0)  # (k, D)
        V = F.normalize(V, p=2, dim=1)
        m = F.normalize(V.mean(dim=0, keepdim=True), p=2, dim=1)  # (1, D)
        grp_embs[gid] = m

    # Helper: can we merge two groups under cannot-link constraints?
    def can_merge(a_gid: int, b_gid: int) -> bool:
        A = groups[a_gid]
        B = groups[b_gid]
        for i in A:
            for j in B:
                if frozenset((i, j)) in cannot:
                    return False
        return True

    # Helper: can we merge two groups under cannot-link constraints?
    def can_merge(a_gid: int, b_gid: int) -> bool:
        A = groups[a_gid]
        B = groups[b_gid]
        for i in A:
            for j in B:
                if frozenset((i, j)) in cannot:
                    return False
        return True

    # 5) Greedy agglomerative merging with similarity threshold & constraints
    active = set(group_ids)
    while True:
        best_pair = None
        best_sim = face_sim_thresh
        act_list = list(active)
        L = len(act_list)
        for a in range(L):
            ga = act_list[a]
            ea = grp_embs[ga]
            for b in range(a + 1, L):
                gb = act_list[b]
                if not can_merge(ga, gb):
                    continue
                eb = grp_embs[gb]
                sim = float(F.cosine_similarity(ea, eb).item())
                if sim >= best_sim:
                    best_sim = sim
                    best_pair = (ga, gb)
        if best_pair is None:
            break
        ga, gb = best_pair
        # merge gb into ga
        groups[ga].extend(groups[gb])
        # recompute emb
        vecs = [embs[m] for m in groups[ga] if embs[m] is not None]
        V = torch.cat(vecs, dim=0)
        V = F.normalize(V, p=2, dim=1)
        grp_embs[ga] = F.normalize(V.mean(dim=0, keepdim=True), p=2, dim=1)
        # deactivate gb
        active.discard(gb)
        del groups[gb]
        del grp_embs[gb]

    # 6) Assign stable identity labels
    stable_ids = {}
    for idx, gid in enumerate(sorted(active)):
        # Use Person_* naming in place of prior VID_*
        label = f"Person_{idx+1}"
        for m in groups[gid]:
            stable_ids[m] = label

    # 7) Build output annotated tracks
    annotated = []
    for i, tr in enumerate(vidTracks):
        t2 = dict(tr)
        ident = stable_ids.get(i)
        if ident is not None:
            t2["identity"] = ident
        else:
            t2["identity"] = None
        annotated.append(t2)

    return annotated
