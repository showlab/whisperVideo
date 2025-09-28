#!/home/siyuan/miniconda3/envs/whisperv/bin/python
"""
Generate LateNightShow visualization assets from WhisperV outputs.

For each episode under the dataset, select the top-4 identities by total
speaking duration (from matched diarization), and for each of their speech
segments, render one representative frame with a bounding box over the face
and save the annotated image. Also emit a CSV per episode containing
identity, start/end times, and the text.

Color choices strictly match analysis/episode_timeline_vis.py:
  - GREEN: #6ec279 (RGB 110,194,121)
  - RED:   #db3a3c (RGB 219,58,60)
  - Additional colors: Matplotlib tab10 for identities 3,4

Requirements (use the project's whisperv environment):
  - numpy, opencv-python, matplotlib, torch

Inputs per episode directory (must exist; no fallbacks):
  {episode}/
    avi/
      audio.wav
      video.avi
    result/
      tracks_identity.pckl           # used for face tracks + bboxes
      matched_diriazation.pckl       # used when --id_mode identity (default before)
      baseline_whisperx/segments.pkl # used when --id_mode speaker
      scores.pckl or smooth_scores.pckl  # ASD scores aligned to tracks
    frame/ or pyframes/
      000001.jpg, ...

Output structure (under --out_dir):
  {out_dir}/latenightshow/{episode_id}/
    colors.json                    # identity -> RGB list
    segments.csv                   # metadata rows
    IDsanitized/
      seg_00012_f001234.jpg       # annotated frame per segment

This script raises clear errors when required files are missing; no dummy or
mock data is produced.
"""

from __future__ import annotations

import os
import re
import csv
import sys
import glob
import json
import math
import pickle
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp


# Exact brand colors copied from analysis/episode_timeline_vis.py
_GREEN_BAR = (110.0/255.0, 194.0/255.0, 121.0/255.0)
_RED_BAR   = (219.0/255.0,  58.0/255.0,  60.0/255.0)


def _rgb01_to_bgr255(c: Tuple[float, float, float]) -> Tuple[int, int, int]:
    r, g, b = c
    return (int(round(b * 255.0)), int(round(g * 255.0)), int(round(r * 255.0)))


def _disp_label(id_str: str) -> str:
    # Convert internal IDs like PERSON_00 / SPEAKER_03 / VID_2 to display form "[0]", "[3]", "[2]"
    try:
        import re as _re
        m = _re.search(r"_(\d+)$", str(id_str))
        if m:
            return f"[{int(m.group(1))}]"
        # Try whole string digits
        if str(id_str).isdigit():
            return f"[{int(id_str)}]"
    except Exception:
        pass
    # Fallback: bracket the raw string
    return f"[{str(id_str)}]"


def _id_bar_color(index: int, n_ids: int):
    if index == 0:
        return _GREEN_BAR
    if index == 1:
        return _RED_BAR
    return plt.cm.get_cmap('tab10', max(3, n_ids))(index)


def _require_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")


def _require_dir(path: str, desc: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _sanitize_id(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:64] if len(s) > 64 else s


def _find_frames_dir(ep_dir: str) -> str:
    for name in ('frame', 'pyframes'):
        p = os.path.join(ep_dir, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"No frames directory under {ep_dir} (expected 'frame' or 'pyframes')")


def _frame_file_map(frames_dir: str) -> Dict[int, str]:
    files = glob.glob(os.path.join(frames_dir, '*.jpg'))
    if not files:
        raise FileNotFoundError(f"No jpg frames in {frames_dir}")
    mapping: Dict[int, str] = {}
    for fp in files:
        bn = os.path.basename(fp)
        name, ext = os.path.splitext(bn)
        try:
            idx = int(name)
        except Exception:
            # skip non-numeric names
            continue
        mapping[idx] = fp
    if not mapping:
        raise RuntimeError(f"Could not parse numeric frame indices in {frames_dir}")
    return mapping


def _top_identities_by_duration(matched_segments: List[Dict[str, Any]], top_k: int) -> List[str]:
    durations: Dict[str, float] = {}
    for seg in matched_segments:
        ID = seg.get('identity')
        if not ID or ID == 'None':
            continue
        st = float(seg.get('start_time', seg.get('start', 0.0)))
        et = float(seg.get('end_time', seg.get('end', st)))
        if et > st:
            durations[str(ID)] = durations.get(str(ID), 0.0) + (et - st)
    ordered = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    return [ID for ID, _ in ordered[:top_k]]


def _collect_segments_by_identity(matched_segments: List[Dict[str, Any]], ids: List[str]) -> Dict[str, List[Tuple[float, float, str, int]]]:
    out: Dict[str, List[Tuple[float, float, str, int]]] = {i: [] for i in ids}
    for idx, seg in enumerate(matched_segments):
        ID = str(seg.get('identity'))
        if ID not in out:
            continue
        st = float(seg.get('start_time', seg.get('start', 0.0)))
        et = float(seg.get('end_time', seg.get('end', st)))
        txt = seg.get('text', '')
        if et > st:
            out[ID].append((st, et, txt, idx))
    return out


# Globals set per-episode in worker initializer
G = {
    'ep_dir': None,
    'fps': None,
    'frames_map': None,      # Dict[int,str]
    'tracks': None,          # list of track dicts
    'id_to_tidx': None,      # Dict[str,List[int]]
    'color_map': None,       # Dict[str,Tuple[int,int,int]] BGR-255
    'out_dir': None,
    'scores': None,          # ASD scores per track
    'mode': 'identity',      # 'identity' or 'speaker'
    'top_ids': None,         # ordered list of IDs to show on every image
    'track_to_person': None, # Dict[int, str] mapping track index -> PERSON_ID
    'track_candidates': None,# Dict[int, List[Tuple[str, float]]] candidates sorted by support
}


def _init_worker(ep_dir: str, fps: float, frames_map: Dict[int, str], tracks: List[dict], color_map: Dict[str, Tuple[int,int,int]], out_dir: str,
                 scores: List, mode: str, top_ids: List[str], track_to_person: Dict[int, str] | None = None,
                 track_candidates: Dict[int, List[Tuple[str, float]]] | None = None):
    G['ep_dir'] = ep_dir
    G['fps'] = float(fps)
    G['frames_map'] = frames_map
    G['tracks'] = tracks
    # Build identity->track indices mapping once
    id_to_tidx: Dict[str, List[int]] = {}
    for tidx, t in enumerate(tracks):
        ID = str(t.get('identity', 'None'))
        if not ID or ID == 'None':
            continue
        id_to_tidx.setdefault(ID, []).append(tidx)
    G['id_to_tidx'] = id_to_tidx
    G['color_map'] = color_map
    G['out_dir'] = out_dir
    G['scores'] = scores
    G['mode'] = mode
    G['top_ids'] = list(top_ids) if top_ids is not None else []
    G['track_to_person'] = dict(track_to_person) if isinstance(track_to_person, dict) else {}
    G['track_candidates'] = dict(track_candidates) if isinstance(track_candidates, dict) else {}


def _pick_frame_for_segment_by_identity(ID: str, st: float, et: float) -> Tuple[int, Dict[str, float], int]:
    """Return (frame_number, bbox_dict) for a frame inside [st,et] for identity ID.
    bbox_dict has keys: x,y,s. Raises on failure.
    """
    if ID not in G['id_to_tidx']:
        raise RuntimeError(f"No tracks found for identity {ID}")
    fps = G['fps']
    tracks = G['tracks']
    mid = 0.5 * (st + et)
    best = None  # (abs_dt, frame_num, bbox, tidx)
    tol = 0.5 / max(1e-6, fps)
    for tidx in G['id_to_tidx'][ID]:
        t = tracks[tidx]
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        xs = t['proc_track']['x']
        ys = t['proc_track']['y']
        ss = t['proc_track']['s']
        for i, f in enumerate(frames_list):
            t_abs = float(f) / fps
            if (st - tol) <= t_abs <= (et + tol):
                dt = abs(t_abs - mid)
                cand = (dt, int(f), {'x': float(xs[i]), 'y': float(ys[i]), 's': float(ss[i])}, tidx)
                if best is None or dt < best[0]:
                    best = cand
    if best is None:
        raise RuntimeError(f"No frame found inside segment for identity {ID} in [{st:.3f},{et:.3f}]")
    return best[1], best[2], best[3]


def _pick_frame_for_segment_by_speaker(st: float, et: float) -> Tuple[int, Dict[str, float], int]:
    """Return (frame_number, bbox_dict) by selecting the track with max ASD score within [st,et].
    bbox_dict has keys: x,y,s. Raises if no track has frames overlapping the segment.
    """
    fps = G['fps']
    tracks = G['tracks']
    scores = G['scores']
    tol = 0.5 / max(1e-6, fps)
    best = None  # (score, frame_num, bbox, tidx)
    if scores is None:
        raise RuntimeError('ASD scores not loaded for speaker mode')
    for tidx, t in enumerate(tracks):
        sc = scores[tidx] if tidx < len(scores) else None
        if sc is None:
            continue
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        xs = t['proc_track']['x']
        ys = t['proc_track']['y']
        ss = t['proc_track']['s']
        # Find candidate frame indices within [st,et]
        for i, f in enumerate(frames_list):
            t_abs = float(f) / fps
            if (st - tol) <= t_abs <= (et + tol):
                # Smooth ASD over small window like inference_folder
                l0 = max(0, i - 2)
                l1 = min(len(sc), i + 3)
                if l1 <= l0:
                    continue
                try:
                    sval = float(np.mean(sc[l0:l1]))
                except Exception:
                    sval = float(sc[i]) if i < len(sc) else -1e9
                # Choose maximum ASD
                if (best is None) or (sval > best[0]):
                    best = (sval, int(f), {'x': float(xs[i]), 'y': float(ys[i]), 's': float(ss[i])}, tidx)
    if best is None:
        raise RuntimeError(f"No track frames overlap segment [{st:.3f},{et:.3f}]")
    return best[1], best[2], best[3]


def _draw_and_save(ID: str, st: float, et: float, text: str, seg_idx: int, out_base: str) -> Dict[str, Any]:
    if G['mode'] == 'identity':
        frame_num, bbox, main_tidx = _pick_frame_for_segment_by_identity(ID, st, et)
    elif G['mode'] == 'speaker':
        frame_num, bbox, main_tidx = _pick_frame_for_segment_by_speaker(st, et)
    else:
        raise RuntimeError(f"Unknown mode {G['mode']}")
    if frame_num not in G['frames_map']:
        raise RuntimeError(f"Frame image {frame_num} not found in frames directory")
    img_path = G['frames_map'][frame_num]
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = image.shape[:2]
    # Compute bbox corners (x,y is center; s is half side length)
    x = bbox['x']; y = bbox['y']; s = bbox['s']
    x0 = max(0, min(int(x - s), w - 1))
    y0 = max(0, min(int(y - s), h - 1))
    x1 = max(0, min(int(x + s), w - 1))
    y1 = max(0, min(int(y + s), h - 1))
    # First, draw all visible faces in this frame using consistent episode-level mapping,
    # possibly overriding when exactly 4 persons are visible (left-to-right -> 0,1,3,2)
    pid_main = _draw_all_faces(image, frame_num, main_tidx)
    if not pid_main:
        pid_main = G.get('track_to_person', {}).get(main_tidx, ID)
    color = G.get('color_map', {}).get(pid_main, G.get('color_map', {}).get(ID, (0, 255, 0)))
    # Then draw main overlay text (truncate long text for image label)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 5)
    short_txt = (text[:140] + '…') if len(text) > 140 else text
    # Only show identity label; no time visualization on image
    label = f"{_disp_label(pid_main)}"
    # Put ID/time above box, content at bottom (enlarged label ~3x)
    cv2.putText(image, label, (x0, max(0, y0 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 7)
    cv2.putText(image, short_txt, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Legend removed per request; no _draw_id_legend call

    # Save
    subdir = os.path.join(out_base, _sanitize_id(ID))
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"seg_{seg_idx:05d}_f{frame_num:06d}.jpg")
    ok = cv2.imwrite(out_path, image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return {
        'identity': ID,
        'segment_index': seg_idx,
        'start_time': float(st),
        'end_time': float(et),
        'text': text,
        'frame_number': int(frame_num),
        'image_path': out_path,
    }


def _draw_id_legend(image: np.ndarray, current_id: str) -> None:
    # Vertical legend at top-left; highlight current_id with thicker rectangle
    if not isinstance(G.get('top_ids'), list) or not G.get('top_ids'):
        return
    h, w = image.shape[:2]
    x = 10
    pad_y = 96
    box_w = 54
    box_h = 54
    # Ensure top of the first legend box stays inside the image
    y = box_h + 16
    # Sort IDs by numeric suffix so legend is [0],[1],[2],[3] top→bottom
    def _id_num(s: str) -> int:
        try:
            import re as _re
            m = _re.search(r"_(\d+)$", str(s))
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return 9999
    for ID in sorted(G['top_ids'], key=_id_num):
        col = G['color_map'].get(ID, (0, 255, 0))
        thickness = 6 if ID == current_id else 4
        # color box
        cv2.rectangle(image, (x, y - box_h), (x + box_w, y), col, thickness)
        # id text
        cv2.putText(image, _disp_label(ID), (x + box_w + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 2.1, col, 5)
        y += pad_y


def _draw_all_faces(image: np.ndarray, frame_num: int, main_tidx: int) -> str | None:
    # Draw all visible tracks at this frame using episode-level track->PERSON mapping for consistency.
    tracks = G['tracks']
    fps = G['fps']
    h, w = image.shape[:2]
    NEUTRAL = (180, 180, 180)  # BGR
    # Gather visible tracks and initial person assignment
    visible = []  # list of (tidx, i, x0,y0,x1,y1)
    for tidx, t in enumerate(tracks):
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        if frame_num not in frames_list:
            continue
        i = frames_list.index(frame_num)
        try:
            x = float(t['proc_track']['x'][i]); y = float(t['proc_track']['y'][i]); s = float(t['proc_track']['s'][i])
        except Exception:
            continue
        x0 = max(0, min(int(x - s), w - 1)); y0 = max(0, min(int(y - s), h - 1))
        x1 = max(0, min(int(x + s), w - 1)); y1 = max(0, min(int(y + s), h - 1))
        visible.append((tidx, i, x0, y0, x1, y1))

    if not visible:
        return None

    # Prepare base mapping and possibly frame override when exactly 4 are visible
    color_map = G.get('color_map', {})
    base_map = G.get('track_to_person', {})
    top_ids = list(G.get('top_ids', []) or [])

    assign: Dict[int, str] = {}
    if len(visible) == 4:
        # left-to-right mapping to PERSON_00, PERSON_01, PERSON_03, PERSON_02
        order = sorted(visible, key=lambda v: (v[2] + v[4]) * 0.5)  # by x-center
        targets = ['PERSON_00', 'PERSON_01', 'PERSON_03', 'PERSON_02']
        # If targets not in our color_map (unusual), fallback to top_ids [0,1,3,2]
        if not all(t in color_map for t in targets):
            if len(top_ids) >= 4:
                targets = [top_ids[0], top_ids[1], top_ids[3], top_ids[2]]
        for (slot, (tidx, i, x0, y0, x1, y1)) in enumerate(order):
            assign[tidx] = targets[min(slot, len(targets)-1)]
    else:
        # default: global mapping
        for (tidx, i, x0, y0, x1, y1) in visible:
            assign[tidx] = base_map.get(tidx)

    # Draw all using chosen mapping
    pid_main: str | None = None
    for (tidx, i, x0, y0, x1, y1) in visible:
        pid = assign.get(tidx)
        if tidx == main_tidx:
            # Defer drawing main track to caller to avoid duplicate labels; only pass back its PERSON
            if pid:
                pid_main = pid
            continue
        if pid and pid in color_map:
            col = color_map[pid]
            cv2.rectangle(image, (x0, y0), (x1, y1), col, 3)
            cv2.putText(image, _disp_label(pid), (x0, max(0, y0 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 2.4, col, 5)
        else:
            cv2.rectangle(image, (x0, y0), (x1, y1), NEUTRAL, 2)
    return pid_main


def _worker(task: Tuple[str, float, float, str, int, str]) -> Dict[str, Any]:
    ID, st, et, text, seg_idx, out_base = task
    # Targeted error capture: skip only the failing segment while reporting the error.
    try:
        return {'ok': True, 'data': _draw_and_save(ID, st, et, text, seg_idx, out_base)}
    except Exception as e:
        return {'ok': False, 'error': f"{type(e).__name__}: {e}", 'identity': ID, 'segment_index': seg_idx, 'start_time': float(st), 'end_time': float(et)}


def _has_frame_for_segment_local(id_to_tidx: Dict[str, List[int]], tracks: List[dict], fps: float, ID: str, st: float, et: float) -> bool:
    if ID not in id_to_tidx:
        return False
    tol = 0.5 / max(1e-6, fps)
    for tidx in id_to_tidx[ID]:
        t = tracks[tidx]
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for f in frames_list:
            t_abs = float(f) / fps
            if (st - tol) <= t_abs <= (et + tol):
                return True
    return False


def _has_any_frame_for_segment(tracks: List[dict], fps: float, st: float, et: float) -> bool:
    tol = 0.5 / max(1e-6, fps)
    for t in tracks:
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for f in frames_list:
            t_abs = float(f) / fps
            if (st - tol) <= t_abs <= (et + tol):
                return True
    return False


def _assign_tracks_to_persons(tracks: List[dict], scores: List, fps: float,
                              segs_by_id: Dict[str, List[Tuple[float, float, str, int]]],
                              top_ids: List[str]) -> Tuple[Dict[int, str], Dict[int, List[Tuple[str, float]]], Dict[str, Any]]:
    # Build merged intervals per PERSON ID
    merged_by_id: Dict[str, List[Tuple[float, float]]] = {}
    for pid, segs in segs_by_id.items():
        ivals = []
        for (st, et, _txt, _idx) in segs:
            if et > st:
                ivals.append((float(st), float(et)))
        ivals.sort()
        merged = []
        for s, e in ivals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        merged_by_id[pid] = [(a, b) for a, b in merged]

    # Helper: check if t lies within any interval (linear scan; top_k is small)
    def inside(ivals: List[Tuple[float, float]], t: float) -> bool:
        for a, b in ivals:
            if a - 1e-6 <= t <= b + 1e-6:
                return True
        return False

    # Accumulate ASD support per (person, track)
    support: Dict[str, Dict[int, float]] = {pid: {} for pid in merged_by_id.keys()}
    for tidx, t in enumerate(tracks):
        sc = scores[tidx] if tidx < len(scores) else None
        if sc is None:
            continue
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for i, f in enumerate(frames_list):
            t_abs = float(f) / fps
            l0 = max(0, i - 2)
            l1 = min(len(sc), i + 3)
            if l1 <= l0:
                continue
            try:
                sval = float(np.mean(sc[l0:l1]))
            except Exception:
                sval = float(sc[i]) if i < len(sc) else 0.0
            if sval <= -1e9:
                continue
            # contribute positive part only
            pos = max(0.0, sval)
            if pos <= 0.0:
                continue
            for pid, ivals in merged_by_id.items():
                if inside(ivals, t_abs):
                    support[pid][tidx] = support[pid].get(tidx, 0.0) + pos

    # For each track, compute ranked PERSON candidates by ASD support (restrict to top_ids)
    mapping: Dict[int, str] = {}
    candidates: Dict[int, List[Tuple[str, float]]] = {}
    for tidx in range(len(tracks)):
        best_pid = None
        best_val = -1.0
        ranking: List[Tuple[str, float]] = []
        for pid in top_ids:
            d = support.get(pid, {})
            v = d.get(tidx, -1.0)
            if v > 0:
                ranking.append((pid, v))
            if v > best_val:
                best_val = v
                best_pid = pid
        if best_pid is not None and best_val > 0.0:
            mapping[tidx] = best_pid
        ranking.sort(key=lambda x: x[1], reverse=True)
        candidates[tidx] = ranking

    # Fallback: tracks with no positive support → assign to nearest PERSON by time (based on merged intervals)
    def track_time_span(t: dict) -> Tuple[float, float, float]:
        frames_arr = t['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        if not frames_list:
            return 0.0, 0.0, 0.0
        s = min(frames_list) / fps
        e = max(frames_list) / fps
        c = 0.5 * (s + e)
        return s, e, c

    def dist_to_intervals(c: float, ivals: List[Tuple[float, float]]) -> float:
        dmin = float('inf')
        for a, b in ivals:
            if a <= c <= b:
                return 0.0
            dmin = min(dmin, abs(c - a), abs(c - b))
        return dmin if dmin != float('inf') else 1e9

    for tidx in range(len(tracks)):
        if tidx in mapping:
            continue
        _, _, c = track_time_span(tracks[tidx])
        best_pid = None
        best_d = float('inf')
        for pid in top_ids:
            ivals = merged_by_id.get(pid, [])
            d = dist_to_intervals(c, ivals)
            if d < best_d:
                best_d = d
                best_pid = pid
        if best_pid is not None:
            mapping[tidx] = best_pid
        if tidx not in candidates:
            candidates[tidx] = []
    # Build co-visibility graph (edge if two tracks share any frame)
    frame_sets = []
    for t in tracks:
        fa = t['track']['frame']
        fl = fa.tolist() if hasattr(fa, 'tolist') else list(fa)
        frame_sets.append(set(int(f) for f in fl))
    neighbors: Dict[int, set] = {i: set() for i in range(len(tracks))}
    for i in range(len(tracks)):
        Fi = frame_sets[i]
        if not Fi:
            continue
        for j in range(i+1, len(tracks)):
            Fj = frame_sets[j]
            if not Fj:
                continue
            if Fi.intersection(Fj):
                neighbors[i].add(j)
                neighbors[j].add(i)
    # Greedy coloring with preferences to enforce per-frame uniqueness of PERSON labels
    # Order tracks by (degree desc, best_score desc)
    def best_score(tidx: int) -> float:
        lst = candidates.get(tidx, [])
        return float(lst[0][1]) if lst else 0.0
    order = list(range(len(tracks)))
    order.sort(key=lambda t: (len(neighbors[t]), best_score(t)), reverse=True)
    assignment: Dict[int, str] = {}
    conflict_pairs = []
    for t in order:
        # build forbidden set from already-assigned neighbors
        forbidden = set(assignment[n] for n in neighbors[t] if n in assignment)
        pref = [pid for pid, _ in candidates.get(t, [])]
        if not pref:
            # fallback preference by nearest-interval distance ranking
            # compute distance per pid
            _, _, c = track_time_span(tracks[t])
            d_pairs = []
            for pid in top_ids:
                ivals = merged_by_id.get(pid, [])
                d_pairs.append((pid, dist_to_intervals(c, ivals)))
            pref = [pid for pid, _ in sorted(d_pairs, key=lambda x: x[1])]
        # pick first non-forbidden person
        pick = None
        for pid in pref:
            if pid not in forbidden:
                pick = pid
                break
        if pick is None:
            # unavoidable conflict (e.g., more than K co-visible); pick best anyway and record conflict
            pick = pref[0] if pref else (top_ids[0] if top_ids else 'PERSON_00')
            conflict_pairs.append((t, list(forbidden)))
        assignment[t] = pick
    # overwrite mapping with coloring assignment for consistency
    mapping.update(assignment)
    stats = {
        'num_tracks': len(tracks),
        'num_conflicting_assignments': len(conflict_pairs),
    }
    return mapping, candidates, stats


def process_episode(ep_dir: str, out_dir: str, top_k: int, workers: int, max_segments_per_id: int | None = None,
                    id_mode: str = 'speaker') -> None:
    # Validate inputs
    avi_dir = os.path.join(ep_dir, 'avi')
    _require_dir(avi_dir, 'avi directory')
    frames_dir = _find_frames_dir(ep_dir)
    result_dir = os.path.join(ep_dir, 'result')
    _require_dir(result_dir, 'result directory')
    tracks_p = os.path.join(result_dir, 'tracks_identity.pckl')
    _require_file(tracks_p, 'tracks_identity.pckl')
    # Load ASD scores
    scores_p = None
    for nm in ['scores.pckl', 'smooth_scores.pckl']:
        cand = os.path.join(result_dir, nm)
        if os.path.isfile(cand):
            scores_p = cand
            break
    if scores_p is None:
        raise FileNotFoundError(f"Missing ASD scores under {result_dir} (scores.pckl or smooth_scores.pckl)")

    # FPS from video
    video_path = os.path.join(avi_dir, 'video.avi')
    _require_file(video_path, 'video.avi')
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Failed to open video for FPS: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by video: {fps}")

    frames_map = _frame_file_map(frames_dir)
    tracks = _load_pickle(tracks_p)
    scores = _load_pickle(scores_p)

    # Select identities and collect segments based on mode
    if id_mode == 'identity':
        matched_p = os.path.join(result_dir, 'matched_diriazation.pckl')
        _require_file(matched_p, 'matched_diriazation.pckl')
        matched = _load_pickle(matched_p)
        top_ids = _top_identities_by_duration(matched, top_k=top_k)
        if len(top_ids) < top_k:
            raise RuntimeError(f"Episode {ep_dir} has only {len(top_ids)} identities with speech; need {top_k}")
        segs_by_id = _collect_segments_by_identity(matched, top_ids)
    elif id_mode == 'speaker':
        wx_p = os.path.join(result_dir, 'baseline_whisperx', 'segments.pkl')
        _require_file(wx_p, 'baseline_whisperx/segments.pkl')
        wx = _load_pickle(wx_p)
        # Build durations and segments per speaker
        dur: Dict[str, float] = {}
        segs_tmp: Dict[str, List[Tuple[float, float, str, int]]] = {}
        for idx, seg in enumerate(wx):
            sp = seg.get('speaker')
            if not sp:
                continue
            st = float(seg.get('start'))
            et = float(seg.get('end'))
            if et <= st:
                continue
            txt = seg.get('text', '')
            sp = str(sp)
            dur[sp] = dur.get(sp, 0.0) + (et - st)
            segs_tmp.setdefault(sp, []).append((st, et, txt, idx))
        if not dur:
            raise RuntimeError(f"No WhisperX speakers found in {wx_p}")
        # Map SPEAKER_XX -> PERSON_XX for output identity names
        ordered_sp = [sp for sp, _ in sorted(dur.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        if len(ordered_sp) < top_k:
            raise RuntimeError(f"Episode {ep_dir} has only {len(top_ids)} speakers by WhisperX; need {top_k}")
        def to_person(spname: str) -> str:
            if spname.startswith('SPEAKER_'):
                return 'PERSON_' + spname.split('SPEAKER_', 1)[1]
            return 'PERSON_' + spname
        id_map = {sp: to_person(sp) for sp in segs_tmp.keys()}
        top_ids = [id_map[sp] for sp in ordered_sp]
        segs_by_id = {id_map[sp]: segs_tmp.get(sp, []) for sp in ordered_sp}
    else:
        raise RuntimeError(f"Unknown id_mode: {id_mode}")
    # Color map in BGR-255 for OpenCV
    color_map: Dict[str, Tuple[int,int,int]] = {}
    for i, ID in enumerate(top_ids):
        c = _id_bar_color(i, len(top_ids))
        if isinstance(c, tuple) and len(c) == 4:
            c = c[:3]
        color_map[ID] = _rgb01_to_bgr255(c)

    # Build local id->track index map for pre-filtering (identity mode only)
    id_to_tidx_local: Dict[str, List[int]] = {}
    if id_mode == 'identity':
        for tidx, t in enumerate(tracks):
            IDt = str(t.get('identity', 'None'))
            if not IDt or IDt == 'None':
                continue
            id_to_tidx_local.setdefault(IDt, []).append(tidx)

    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    # Save colors.json as RGB-255
    colors_json = {ID: list(map(int, color_map[ID])) for ID in top_ids}
    with open(os.path.join(out_dir, 'colors.json'), 'w', encoding='utf-8') as f:
        json.dump(colors_json, f, indent=2)

    # Build tasks (pre-filter segments that have no reachable frame to avoid hard failures)
    tasks: List[Tuple[str, float, float, str, int, str]] = []
    skipped: List[Dict[str, Any]] = []
    for ID in top_ids:
        segs = segs_by_id.get(ID, [])
        if max_segments_per_id is not None and max_segments_per_id >= 0:
            segs = segs[:max_segments_per_id]
        for (st, et, txt, seg_idx) in segs:
            if id_mode == 'identity':
                ok = _has_frame_for_segment_local(id_to_tidx_local, tracks, fps, ID, float(st), float(et))
            else:
                ok = _has_any_frame_for_segment(tracks, fps, float(st), float(et))
            if ok:
                tasks.append((ID, float(st), float(et), txt, int(seg_idx), out_dir))
            else:
                skipped.append({'identity': ID, 'segment_index': int(seg_idx), 'start_time': float(st), 'end_time': float(et), 'reason': 'no_frame_in_segment'})

    if not tasks:
        raise RuntimeError(f"No segments to process in episode {ep_dir}")

    # Build (optional) track->PERSON mapping for drawing all IDs on frame
    track_to_person = {}
    track_candidates = {}
    if id_mode == 'speaker':
        track_to_person, track_candidates, assign_stats = _assign_tracks_to_persons(tracks, scores, fps, segs_by_id, top_ids)

    # Spawn torch multiprocessing pool per-episode
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max(1, int(workers)), initializer=_init_worker,
                  initargs=(ep_dir, fps, frames_map, tracks, color_map, out_dir, scores, id_mode, top_ids, track_to_person, track_candidates)) as pool:
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for i, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
            if isinstance(res, dict) and not res.get('ok', False):
                errors.append(res)
            else:
                results.append(res['data'] if 'data' in res else res)

    # Write CSV
    csv_path = os.path.join(out_dir, 'segments.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as wf:
        w = csv.DictWriter(wf, fieldnames=['identity','segment_index','start_time','end_time','text','frame_number','image_path'])
        w.writeheader()
        for r in sorted(results, key=lambda x: (x['identity'], x['segment_index'])):
            w.writerow(r)

    # Write an errors JSON including skipped segments and worker failures for transparency
    if skipped or errors:
        with open(os.path.join(out_dir, 'segment_errors.json'), 'w', encoding='utf-8') as f:
            json.dump({'skipped_no_frame': skipped, 'worker_errors': errors}, f, indent=2)

    # Persist mapping for review
    if track_to_person:
        with open(os.path.join(out_dir, 'track_to_person.json'), 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in track_to_person.items()}, f, indent=2)
    if id_mode == 'speaker':
        with open(os.path.join(out_dir, 'assign_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(assign_stats, f, indent=2)


def discover_episodes(show_dir: str) -> List[str]:
    eps = []
    for name in sorted(os.listdir(show_dir)):
        ep = os.path.join(show_dir, name)
        if os.path.isdir(ep):
            eps.append(ep)
    if not eps:
        raise FileNotFoundError(f"No episodes found under {show_dir}")
    return eps


def main():
    parser = argparse.ArgumentParser(description='Generate LateNightShow assets (bbox images + CSV) from WhisperV outputs.')
    parser.add_argument('--dataset', type=str, default='/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset/latenightshow',
                        help='Path to latenightshow directory containing episode subdirectories')
    parser.add_argument('--episodes', type=str, default='all',
                        help="Comma-separated episode IDs to process, or 'all' to process every episode")
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'outputs', 'latenight_assets'),
                        help='Output base directory to write assets')
    parser.add_argument('--top_k', type=int, default=4, help='Number of identities per episode')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2), help='Torch multiprocessing workers per episode')
    parser.add_argument('--max_segments_per_id', type=int, default=None, help='Optional cap per-identity; default processes all segments')
    parser.add_argument('--id_mode', type=str, choices=['speaker','identity'], default='speaker',
                        help="Identity source: 'speaker' uses WhisperX speakers; 'identity' uses matched_diriazation identities")
    args = parser.parse_args()

    _require_dir(args.dataset, 'latenightshow dataset directory')
    if args.episodes.strip().lower() == 'all':
        episodes = discover_episodes(args.dataset)
    else:
        ids = [s.strip() for s in args.episodes.split(',') if s.strip()]
        if not ids:
            raise RuntimeError('No episodes specified')
        episodes = [os.path.join(args.dataset, i) for i in ids]

    # Process each episode; if any fail, exit non-zero after reporting
    failures = []
    for ep_dir in episodes:
        ep_id = os.path.basename(ep_dir.rstrip('/'))
        out_ep = os.path.join(args.out_dir, ep_id)
        try:
            process_episode(ep_dir, out_ep, top_k=args.top_k, workers=args.workers, max_segments_per_id=args.max_segments_per_id,
                            id_mode=args.id_mode)
            print(f"Processed episode {ep_id} -> {out_ep}")
        except Exception as e:
            failures.append((ep_id, str(e)))
            print(f"ERROR episode {ep_id}: {e}", file=sys.stderr)

    if failures:
        # Also write a summary file for debugging
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'failures.json'), 'w', encoding='utf-8') as f:
            json.dump([{ 'episode': ep, 'error': err } for ep, err in failures], f, indent=2)
        raise SystemExit(f"Completed with {len(failures)} episode failures. See failures.json under {args.out_dir}")


if __name__ == '__main__':
    main()
