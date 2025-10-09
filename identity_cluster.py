import os
import cv2
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import os
_EMBEDDER_CACHE = {}

# Embedding backends: default to MagFace; allow override via IDENTITY_EMBEDDER env
def _build_embedder(device: str = "cuda", batch_size: int = 16):
    backend = os.environ.get("IDENTITY_EMBEDDER", "magface").strip().lower()
    if backend == "magface":
        try:
            from .embedders.magface_embedder import MagFaceEmbedder
        except Exception:
            from embedders.magface_embedder import MagFaceEmbedder
        backbone = os.environ.get("MAGFACE_BACKBONE", "iresnet100")
        key = ("magface", device, int(batch_size), backbone)
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = MagFaceEmbedder(device=device, batch_size=batch_size, backbone=backbone)
        return _EMBEDDER_CACHE[key]
    elif backend == "facenet":
        try:
            from .identity_verifier import IdentityVerifier
        except Exception:
            from identity_verifier import IdentityVerifier
        key = ("facenet", device, int(batch_size))
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = IdentityVerifier(device=device, batch_size=batch_size)
        return _EMBEDDER_CACHE[key]
    else:
        raise RuntimeError(f"Unsupported IDENTITY_EMBEDDER backend: {backend}")


def _frames_to_bbox_map(track: Dict) -> Dict[int, Tuple[float, float, float, float]]:
    frames = track["track"]["frame"]
    bboxes = track["track"]["bbox"]
    # frames/bboxes can be numpy arrays; ensure Python types
    frames_list = frames.tolist() if hasattr(frames, "tolist") else list(frames)
    bboxes_list = bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
    return {int(f): tuple(map(float, bb)) for f, bb in zip(frames_list, bboxes_list)}

def _sample_indices(total: int, active_indices: Optional[List[int]], max_samples: int = 15) -> List[int]:
    if total <= 0:
        return []
    if active_indices and len(active_indices) > 0:
        idx = sorted(set(int(min(max(0, i), total - 1)) for i in active_indices))
        if len(idx) > max_samples:
            step = max(1, len(idx) // max_samples)
            idx = idx[::step][:max_samples]
        return idx
    # default spread
    return [0, total // 4, total // 2, (3 * total) // 4, max(0, total - 1)]

def _crop_face_bgr(image_bgr: np.ndarray, x: float, y: float, s: float, cs: float) -> Optional[np.ndarray]:
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        return None
    H, W = image_bgr.shape[:2]
    bsi = int(s * (1 + 2 * cs))
    pad_val = 110
    frame = np.pad(image_bgr, ((bsi,bsi),(bsi,bsi),(0,0)), mode='constant', constant_values=pad_val)
    my = y + bsi
    mx = x + bsi
    y1 = int(my - s)
    y2 = int(my + s * (1 + 2 * cs))
    x1 = int(mx - s * (1 + cs))
    x2 = int(mx + s * (1 + cs))
    if y2 <= y1 or x2 <= x1:
        return None
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (224,224))


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
    save_avatars_path: Optional[str] = None,
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
    # Track-level best aligned face and its quality (mag)
    track_best: Dict[int, Dict[str, object]] = {}
    valid_idx: List[int] = []
    # Scheme A: skip tracks with no positive ASD frames when scores_list provided
    include_track = [True] * len(vidTracks)
    if scores_list is not None:
        include_track = []
        for i in range(len(vidTracks)):
            sc = scores_list[i] if i < len(scores_list) else None
            # If no ASD available, do not exclude the track; just skip gating
            if not (isinstance(sc, (list, tuple)) and len(sc) > 0):
                include_track.append(True)
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
        if not include_track[i]:
            embs.append(None)
            continue
        crop_file = tr.get("cropFile")
        emb = None
        best_img = None
        best_mag = None
        # Active indices from ASD gating
        active_idx = None
        if scores_list is not None and i < len(scores_list):
            sc = scores_list[i]
            if isinstance(sc, (list, tuple)) and len(sc) > 0:
                idx = [k for k, v in enumerate(sc) if float(v) > asd_score_thresh]
                if len(idx) > 0:
                    active_idx = idx
        # 1) If crop file exists, open video and compute aligned frames for quality + embedding
        if crop_file:
            avi_path = crop_file + ".avi"
            if os.path.isfile(avi_path):
                import cv2 as _cv
                cap = _cv.VideoCapture(avi_path)
                total = int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    # select sample indices
                    positions = _sample_indices(total, active_idx, max_samples=15)
                    tensors = []
                    align_src = []
                    for pos in positions:
                        cap.set(_cv.CAP_PROP_POS_FRAMES, int(pos))
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        t = embedder._align_and_preprocess(frame)
                        if t is not None:
                            tensors.append(t)
                            align_src.append(t)  # store aligned tensor for thumbnail
                    cap.release()
                    if tensors:
                        batch = torch.stack(tensors, dim=0).to(embedder.device)
                        out = embedder.model(batch)
                        mags = torch.norm(out, p=2, dim=1)
                        out_n = F.normalize(out, p=2, dim=1)
                        # weighted average embedding as before
                        w = mags.view(-1,1)
                        emb = F.normalize((out_n * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8), p=2, dim=1)
                        # best aligned image by mag
                        j = int(torch.argmax(mags).item())
                        best_mag = float(mags[j].item())
                        at = align_src[j].cpu().numpy()  # CHW float
                        best_img = (at.transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
        # 2) Otherwise, build frames from original video + proc_track and embed in-memory
        if emb is None:
            video_path = tr.get('video_path') or tr.get('videoFilePath')
            proc = tr.get('proc_track')
            track_obj = tr.get('track', {})
            frames = track_obj.get('frame')
            if (not video_path) or (proc is None) or (frames is None):
                embs.append(None)
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                embs.append(None)
                continue
            frames_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            sel = _sample_indices(len(frames_list), active_idx, max_samples=15)
            frames_bgr = []
            cs = float(tr.get('cropScale', 0.40))
            for k in sel:
                fidx = int(frames_list[k])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, img = cap.read()
                if not ret:
                    continue
                x = float(proc['x'][k]); y = float(proc['y'][k]); s = float(proc['s'][k])
                face = _crop_face_bgr(img, x, y, s, cs)
                if face is not None:
                    frames_bgr.append(face)
            cap.release()
            if frames_bgr:
                # Align & forward to get mags and embedding
                tensors = []
                for fb in frames_bgr:
                    t = embedder._align_and_preprocess(fb)
                    if t is not None:
                        tensors.append(t)
                if tensors:
                    batch = torch.stack(tensors, dim=0).to(embedder.device)
                    out = embedder.model(batch)
                    mags = torch.norm(out, p=2, dim=1)
                    out_n = F.normalize(out, p=2, dim=1)
                    w = mags.view(-1,1)
                    emb = F.normalize((out_n * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8), p=2, dim=1)
                    j = int(torch.argmax(mags).item())
                    best_mag = float(mags[j].item())
                    at = tensors[j].cpu().numpy()
                    best_img = (at.transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
        if emb is None:
            embs.append(None)
            continue
        # Save best for this track if available
        if isinstance(best_img, np.ndarray) and best_img.size > 0 and (best_mag is not None):
            track_best[i] = { 'img': best_img, 'mag': float(best_mag) }
        embs.append(emb)
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
    # 4) Group embeddings (use sum of unit embeddings to allow O(1) centroid updates)
    grp_embs: Dict[int, torch.Tensor] = {}
    grp_sums: Dict[int, torch.Tensor] = {}
    grp_counts: Dict[int, int] = {}
    for gid, members in groups.items():
        vecs = [embs[m] for m in members if embs[m] is not None]
        if not vecs:
            raise RuntimeError("Encountered a group without embeddings; aborting to avoid fake data.")
        V = torch.cat(vecs, dim=0)  # (k, D)
        V = F.normalize(V, p=2, dim=1)  # normalize each member embedding
        S = V.sum(dim=0, keepdim=True)  # sum of unit vectors
        grp_sums[gid] = S
        grp_counts[gid] = V.size(0)
        grp_embs[gid] = F.normalize(S.clone(), p=2, dim=1)  # centroid direction

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

    # 5) Greedy agglomerative merging with similarity threshold & constraints (vectorized sims)
    active = set(group_ids)
    while True:
        act_list = list(active)
        L = len(act_list)
        if L <= 1:
            break
        # Build centroid matrix [L, D]
        C = torch.cat([grp_embs[g] for g in act_list], dim=0)  # rows are unit centroids
        # Cosine similarity matrix via dot product
        S = torch.matmul(C, C.t())  # [L, L]
        # Mask diagonal
        idx = torch.arange(L)
        S[idx, idx] = -1e9

        merged = False
        while True:
            max_val = torch.max(S)
            best_sim = float(max_val.item()) if torch.is_tensor(max_val) else float(max_val)
            if best_sim < face_sim_thresh:
                break
            pos = torch.nonzero(S == max_val, as_tuple=False)
            if pos.numel() == 0:
                break
            a_i, b_i = int(pos[0,0].item()), int(pos[0,1].item())
            ga, gb = act_list[a_i], act_list[b_i]
            if can_merge(ga, gb):
                groups[ga].extend(groups[gb])
                grp_sums[ga] = grp_sums[ga] + grp_sums[gb]
                grp_counts[ga] = grp_counts[ga] + grp_counts[gb]
                grp_embs[ga] = F.normalize(grp_sums[ga], p=2, dim=1)
                active.discard(gb)
                del groups[gb]
                del grp_embs[gb]
                del grp_sums[gb]
                del grp_counts[gb]
                merged = True
                break
            else:
                S[a_i, b_i] = -1e9
                S[b_i, a_i] = -1e9
        if not merged:
            break

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

    # 8) Build per-identity best avatar (reuse MagFace quality) and optionally persist
    if save_avatars_path:
        id_best = {}
        for gid in active:
            members = groups[gid]
            # find member with highest best_mag
            best = None
            best_m = -1.0
            for m in members:
                rec = track_best.get(m)
                if not rec:
                    continue
                if float(rec['mag']) > best_m:
                    best_m = float(rec['mag'])
                    best = rec['img']
            if best is None:
                continue
            label = None
            # find label for this group
            for k,v in stable_ids.items():
                if k in members:
                    label = v; break
            if isinstance(label, str):
                id_best[label] = best
        if id_best:
            try:
                import pickle
                with open(save_avatars_path, 'wb') as f:
                    pickle.dump(id_best, f)
            except Exception:
                pass

    return annotated
