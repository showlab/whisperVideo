import os
import sys
import argparse
import json
import math
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import numpy as np


def read_rttm(path: str) -> List[Dict]:
    segs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # SPEAKER <file-id> <chan> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
            parts = line.split()
            if len(parts) < 9 or parts[0] != 'SPEAKER':
                continue
            tbeg = float(parts[3]); tdur = float(parts[4]); name = parts[7]
            segs.append({'start': tbeg, 'end': tbeg + tdur, 'speaker': str(name)})
    return segs


def _merge_same_speaker(ref_segs: List[Dict], gap: float) -> List[Dict]:
    """Merge adjacent segments of the same speaker when the gap <= `gap` seconds.
    Keeps order stable; assumes ref_segs are lists of dicts with keys start/end/speaker (floats/str).
    """
    if gap is None or gap <= 0.0 or not ref_segs:
        return ref_segs
    ss = sorted(ref_segs, key=lambda x: (float(x['start']), float(x['end'])))
    out: List[Dict] = []
    cur = ss[0].copy()
    for nxt in ss[1:]:
        if str(nxt['speaker']) == str(cur['speaker']) and (float(nxt['start']) - float(cur['end'])) <= float(gap):
            # merge by extending end
            cur['end'] = max(float(cur['end']), float(nxt['end']))
        else:
            out.append(cur)
            cur = nxt.copy()
    out.append(cur)
    return out


def _absorb_tiny_same_speaker(ref_segs: List[Dict], min_seg: float) -> List[Dict]:
    """Absorb very short segments (< min_seg) into adjacent same-speaker neighbor when possible.
    Does not merge across speakers to avoid corrupting labels; if no same-speaker neighbor, keep as-is.
    """
    if min_seg is None or min_seg <= 0.0 or not ref_segs:
        return ref_segs
    segs = sorted(ref_segs, key=lambda x: (float(x['start']), float(x['end'])))
    n = len(segs)
    if n <= 1:
        return segs
    out: List[Dict] = []
    i = 0
    while i < n:
        s = segs[i].copy()
        dur = float(s['end']) - float(s['start'])
        if dur >= min_seg:
            out.append(s)
            i += 1
            continue
        # try absorb into previous same-speaker
        absorbed = False
        if out and str(out[-1]['speaker']) == str(s['speaker']):
            out[-1]['end'] = max(float(out[-1]['end']), float(s['end']))
            absorbed = True
        else:
            # try absorb forward into next same-speaker
            if i + 1 < n and str(segs[i + 1]['speaker']) == str(s['speaker']):
                nxt = segs[i + 1].copy()
                s_merged = {'start': float(min(float(s['start']), float(nxt['start']))),
                            'end': float(max(float(s['end']), float(nxt['end']))),
                            'speaker': str(s['speaker'])}
                segs[i + 1] = s_merged
                absorbed = True
            else:
                # no same-speaker neighbor to absorb; keep as-is
                out.append(s)
        i += 1 if not absorbed else 1
    # ensure sorted and non-overlapping per speaker where possible
    out.sort(key=lambda x: (float(x['start']), float(x['end'])))
    return out


def smooth_gt_segments(ref_segs: List[Dict], merge_gap: Optional[float], min_seg: Optional[float]) -> List[Dict]:
    """Apply in-memory smoothing of GT segments:
    1) Merge adjacent same-speaker segments separated by gap <= merge_gap
    2) Absorb very short segments (< min_seg) into same-speaker neighbors when possible
    Returns a new list of segments.
    """
    out = [
        {
            'start': float(s.get('start', 0.0)),
            'end': float(s.get('end', 0.0)),
            'speaker': str(s.get('speaker', 'UNK')),
        }
        for s in ref_segs
        if float(s.get('end', 0.0)) > float(s.get('start', 0.0))
    ]
    if merge_gap is not None and merge_gap > 0.0:
        out = _merge_same_speaker(out, merge_gap)
    if min_seg is not None and min_seg > 0.0:
        out = _absorb_tiny_same_speaker(out, min_seg)
    return out



def load_sys_segments(pickle_path: str) -> List[Dict]:
    import pickle
    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)
    # Support list of dicts with keys start, end, speaker
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and 'start' in obj[0] and 'end' in obj[0]:
        return [
            {
                'start': float(x.get('start', x.get('start_time', 0.0))),
                'end': float(x.get('end', x.get('end_time', 0.0))),
                'speaker': str(x.get('speaker', 'UNK')),
            }
            for x in obj
            if float(x.get('end', x.get('end_time', 0.0))) > float(x.get('start', x.get('start_time', 0.0)))
        ]
    raise RuntimeError(f'Unsupported system pickle format: {pickle_path}')


def segments_to_tracks(segs: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for s in segs:
        lab = str(s['speaker'])
        out.setdefault(lab, []).append((float(s['start']), float(s['end'])))
    # normalize
    for k, v in out.items():
        v.sort()
    return out


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return max(0.0, e - s)


def total_overlap(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if e > s:
            tot += e - s
        if A[i][1] <= B[j][1]:
            i += 1
        else:
            j += 1
    return tot


def hungarian_max(cost: np.ndarray) -> List[Tuple[int, int]]:
    # maximize overlap -> use linear_sum_assignment on negative values
    from scipy.optimize import linear_sum_assignment
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(-cost)
    return list(zip(r.tolist(), c.tolist()))


def build_global_mapping(sys_segs: List[Dict], ref_segs: List[Dict]) -> Dict[str, str]:
    # Build overlap matrix between system labels and reference labels
    sys_tracks = segments_to_tracks(sys_segs)
    ref_tracks = segments_to_tracks(ref_segs)
    sys_labels = sorted(sys_tracks.keys())
    ref_labels = sorted(ref_tracks.keys())
    if not sys_labels or not ref_labels:
        return {}
    M = np.zeros((len(sys_labels), len(ref_labels)), dtype=float)
    for i, s in enumerate(sys_labels):
        for j, r in enumerate(ref_labels):
            M[i, j] = total_overlap(sys_tracks[s], ref_tracks[r])
    pairs = hungarian_max(M)
    mapping: Dict[str, str] = {}
    for i, j in pairs:
        if M[i, j] > 0.0:
            mapping[sys_labels[i]] = ref_labels[j]
    return mapping


def label_segments_by_mapping(segs: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    out = []
    for s in segs:
        lab = str(s['speaker'])
        out.append({'start': float(s['start']), 'end': float(s['end']), 'speaker': mapping.get(lab, 'UNK')})
    return out


def acc_ppc_rpc(ref_segs: List[Dict], sys_mapped: List[Dict]) -> Tuple[float, float, float]:
    # Predict per ref segment: pick system label with maximum overlap within [start,end]
    # Build per-label precision/recall over ref segments
    # index system by time
    sys_by_lab = segments_to_tracks(sys_mapped)
    # labels set
    ref_labels: List[str] = sorted(set(str(s['speaker']) for s in ref_segs))

    y_true: List[str] = []
    y_pred: List[str] = []

    for seg in ref_segs:
        s = float(seg['start']); e = float(seg['end']); lab = str(seg['speaker'])
        # compute overlap per label
        best_lab = 'UNK'; best_ov = 0.0
        for L, spans in sys_by_lab.items():
            ov = 0.0
            # sum overlap with spans intersecting [s,e]
            # coarse: iterate all; lists are small
            for a, b in spans:
                if b <= s or a >= e:
                    continue
                ov += overlap((a, b), (s, e))
            if ov > best_ov:
                best_ov = ov; best_lab = L
        y_true.append(lab)
        y_pred.append(best_lab)

    # Accuracy
    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p)) / max(1, len(y_true))

    # Per-character precision/recall over ref labels only
    precisions = []
    recalls = []
    for c in ref_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        precisions.append(prec)
        recalls.append(rec)
    ppc = float(np.mean(precisions)) if precisions else 0.0
    rpc = float(np.mean(recalls)) if recalls else 0.0
    return acc, ppc, rpc


def acc_ppc_rpc_hyp(ref_segs: List[Dict], sys_mapped: List[Dict]) -> Tuple[float, float, float]:
    """Hypothesis-anchored (system-anchored) metrics over system segments that overlap GT.
    - For each system segment, if it overlaps any reference speech, assign a GT label
      by maximum time-overlap with reference; else drop segment.
    - Compute Acc over these system segments.
    - Compute per-character precision/recall and average across reference labels.
    """
    # Index reference by label → list of spans
    ref_by_lab = segments_to_tracks(ref_segs)
    ref_labels: List[str] = sorted(ref_by_lab.keys())

    # Helper: GT label for a system segment by max overlap
    def gt_label_for_sys(seg: Dict) -> Optional[str]:
        s = float(seg['start']); e = float(seg['end'])
        best_lab = None; best_ov = 0.0
        for L, spans in ref_by_lab.items():
            # total overlap with this label's spans
            ov = 0.0
            for a, b in spans:
                if b <= s or a >= e:
                    continue
                ov += overlap((a, b), (s, e))
            if ov > best_ov:
                best_ov = ov; best_lab = L
        return best_lab if best_ov > 0.0 else None

    # Build samples: (y_true, y_pred) over sys segments with overlap
    y_true: List[str] = []
    y_pred: List[str] = []
    for seg in sys_mapped:
        gt = gt_label_for_sys(seg)
        if gt is None:
            continue  # ignore sys segment with no GT overlap
        y_true.append(gt)
        y_pred.append(str(seg['speaker']))

    if not y_true:
        return 0.0, 0.0, 0.0

    # Accuracy over sys segments
    acc = float(sum(1 for t, p in zip(y_true, y_pred) if t == p)) / max(1, len(y_true))

    # Per-character precision/recall across reference labels
    precisions = []
    recalls = []
    for c in ref_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        precisions.append(prec)
        recalls.append(rec)
    ppc = float(np.mean(precisions)) if precisions else 0.0
    rpc = float(np.mean(recalls)) if recalls else 0.0
    return acc, ppc, rpc


def der_with_pyannote(ref_segs: List[Dict], sys_mapped: List[Dict], collar: float, skip_overlap: bool) -> float:
    # Build pyannote.core Annotation for ref and sys
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except Exception as e:
        raise RuntimeError(
            'pyannote.metrics is required for DER computation.\n'
            'Install: /home/siyuan/miniconda3/envs/whisperv/bin/python -m pip install pyannote.metrics\n'
            f'Import error: {e}'
        )
    ref = Annotation()
    for s in ref_segs:
        if s['end'] <= s['start']:
            continue
        ref[Segment(s['start'], s['end'])] = s['speaker']
    hyp = Annotation()
    for s in sys_mapped:
        if s['end'] <= s['start']:
            continue
        hyp[Segment(s['start'], s['end'])] = s['speaker']
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    return float(metric(ref, hyp))


@dataclass
class Episode:
    path: str
    show: str
    episode: str


def find_episodes(dataset_root: str) -> List[Episode]:
    # If dataset_root itself is an episode dir (contains result/), return it directly
    if os.path.isdir(os.path.join(dataset_root, 'result')):
        parent, episode = os.path.split(dataset_root.rstrip('/'))
        show = os.path.basename(parent)
        return [Episode(path=dataset_root, show=show, episode=episode)]

    eps: List[Episode] = []
    # detect show/episode layout (dataset_root -> show -> episode)
    for show in sorted(os.listdir(dataset_root)):
        show_dir = os.path.join(dataset_root, show)
        if not os.path.isdir(show_dir):
            continue
        for ep in sorted(os.listdir(show_dir)):
            ep_dir = os.path.join(show_dir, ep)
            if os.path.isdir(ep_dir):
                eps.append(Episode(path=ep_dir, show=show, episode=ep))
    if not eps:
        raise RuntimeError(f'No episodes found under {dataset_root}')
    return eps


def frame_level_accuracy(ref_segs: List[Dict], sys_mapped: List[Dict]) -> float:
    """Frame-level (time-weighted) label accuracy over reference speech regions.
    - Build breakpoints at all starts/ends from ref and sys.
    - For each micro-interval, if reference has a label, count duration and correctness.
    - If multiple labels overlap (rare for STT), choose the (first) one; system similarly.
    """
    # Collect breakpoints
    bps = set()
    for s in ref_segs:
        bps.add(float(s['start'])); bps.add(float(s['end']))
    for s in sys_mapped:
        bps.add(float(s['start'])); bps.add(float(s['end']))
    if not bps:
        return 0.0
    xs = sorted(bps)
    # Index segments
    ref_sorted = sorted(ref_segs, key=lambda x: (x['start'], x['end']))
    sys_sorted = sorted(sys_mapped, key=lambda x: (x['start'], x['end']))
    i = j = 0
    correct = 0.0
    total = 0.0
    for k in range(len(xs) - 1):
        a = xs[k]; b = xs[k + 1]
        if b <= a:
            continue
        # find ref label covering [a,b)
        R = None
        while i < len(ref_sorted) and ref_sorted[i]['end'] <= a:
            i += 1
        if i < len(ref_sorted):
            s = ref_sorted[i]
            if s['start'] < b and s['end'] > a:
                R = s['speaker']
        if R is None:
            continue  # only score where reference has speech
        # find sys label covering [a,b)
        H = None
        while j < len(sys_sorted) and sys_sorted[j]['end'] <= a:
            j += 1
        if j < len(sys_sorted):
            s2 = sys_sorted[j]
            if s2['start'] < b and s2['end'] > a:
                H = s2['speaker']
        dur = b - a
        total += dur
        if H == R:
            correct += dur
    return correct / total if total > 0 else 0.0


def turn_change_f1(ref_segs: List[Dict], sys_mapped: List[Dict], tol: float = 0.25) -> Tuple[float, float, float]:
    """Turn-change detection F1, with tolerance window.
    - Extract boundaries at ref label changes (end of one seg/start of next with different label).
    - Extract boundaries at system label changes.
    - Match predicted to reference within ±tol (greedy), then compute P/R/F1.
    """
    def boundaries(segs: List[Dict]) -> List[float]:
        ss = sorted(segs, key=lambda x: (x['start'], x['end']))
        b = []
        for a, b_ in zip(ss[:-1], ss[1:]):
            if a['end'] <= b_['start']:
                # gap; count boundary at a.end if speaker changes
                if a['speaker'] != b_['speaker']:
                    b.append(float(a['end']))
            else:
                # overlap; consider mid boundary
                if a['speaker'] != b_['speaker']:
                    b.append(float(min(a['end'], b_['start'])))
        return b

    ref_b = boundaries(ref_segs)
    hyp_b = boundaries(sys_mapped)
    if not ref_b and not hyp_b:
        return 0.0, 0.0, 0.0
    ref_b = sorted(ref_b)
    hyp_b = sorted(hyp_b)
    used = [False] * len(ref_b)
    tp = 0
    for hb in hyp_b:
        # find closest unmatched reference boundary within tol
        best = None; bestd = None
        for idx, rb in enumerate(ref_b):
            if used[idx]:
                continue
            d = abs(hb - rb)
            if d <= tol and (bestd is None or d < bestd):
                bestd = d; best = idx
        if best is not None:
            used[best] = True
            tp += 1
    fp = len(hyp_b) - tp
    fn = len(ref_b) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec)) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def main():
    parser = argparse.ArgumentParser(description='Evaluate WhisperV against Google-based GT (no WER).')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--pred_file', type=str, default='refined_diriazation_asr.pckl',
                        help='Which system output file in result/ to evaluate')
    parser.add_argument('--collar', type=float, default=0.25)
    parser.add_argument('--skip_overlap', action='store_true', help='Report DER(O) by skipping overlap')
    parser.add_argument('--also_der_with_overlap', action='store_true', help='Also compute DER with overlap (in addition to DER(O))')
    parser.add_argument('--gt_merge_gap', type=float, default=None,
                        help='If set, merge adjacent GT segments of the same speaker when gap <= this (seconds)')
    parser.add_argument('--gt_min_seg', type=float, default=None,
                        help='If set, absorb GT segments shorter than this into same-speaker neighbors when possible')
    # Removed advanced flip absorption options; keep only merge_gap/min_seg smoothing
    args = parser.parse_args()

    eps = find_episodes(args.dataset_root)

    per_ep = []
    for ep in eps:
        res_dir = os.path.join(ep.path, 'result')
        rttm = os.path.join(res_dir, 'google_stt', 'segments.rttm')
        pred = os.path.join(res_dir, args.pred_file)
        if not os.path.exists(rttm):
            print(f'[EVAL][SKIP] Missing GT RTTM: {rttm}')
            continue
        if not os.path.exists(pred):
            print(f'[EVAL][SKIP] Missing system output: {pred}')
            continue
        ref_segs = read_rttm(rttm)
        # Optional GT smoothing (in-memory only): same-speaker gap merge + tiny-segment absorb
        if (args.gt_merge_gap is not None and args.gt_merge_gap > 0.0) or \
           (args.gt_min_seg is not None and args.gt_min_seg > 0.0):
            ref_segs = smooth_gt_segments(ref_segs, args.gt_merge_gap, args.gt_min_seg)
        sys_segs = load_sys_segments(pred)
        # Build global mapping and map system labels to ref label set
        mapping = build_global_mapping(sys_segs, ref_segs)
        sys_mapped = label_segments_by_mapping(sys_segs, mapping)

        # Acc/Ppc/Rpc over reference segments
        acc_ref, ppc_ref, rpc_ref = acc_ppc_rpc(ref_segs, sys_mapped)
        # Hypothesis-anchored variants over system segments
        acc_hyp, ppc_hyp, rpc_hyp = acc_ppc_rpc_hyp(ref_segs, sys_mapped)
        # Frame-level accuracy (FLA)
        fla = frame_level_accuracy(ref_segs, sys_mapped)
        # Turn-change F1
        f1_tc, p_tc, r_tc = turn_change_f1(ref_segs, sys_mapped, tol=args.collar)

        # DER(s)
        der_o = None
        der_w = None
        try:
            if args.skip_overlap:
                der_o = der_with_pyannote(ref_segs, sys_mapped, collar=args.collar, skip_overlap=True)
            if args.also_der_with_overlap or (not args.skip_overlap):
                der_w = der_with_pyannote(ref_segs, sys_mapped, collar=args.collar, skip_overlap=False)
        except RuntimeError as e:
            print(f'[EVAL][WARN] {e}')

        per_ep.append({
            'show': ep.show,
            'episode': ep.episode,
            'acc': acc_ref,
            'ppc': ppc_ref,
            'rpc': rpc_ref,
            'acc_hyp': acc_hyp,
            'ppc_hyp': ppc_hyp,
            'rpc_hyp': rpc_hyp,
            'fla': fla,
            'turn_f1': f1_tc,
            'turn_p': p_tc,
            'turn_r': r_tc,
            'der_o': der_o,
            'der': der_w,
        })
        print(f"[EVAL] {ep.show}/{ep.episode}: Acc(ref)={acc_ref:.3f} Ppc(ref)={ppc_ref:.3f} Rpc(ref)={rpc_ref:.3f} | Acc(hyp)={acc_hyp:.3f} Ppc(hyp)={ppc_hyp:.3f} Rpc(hyp)={rpc_hyp:.3f} | FLA={fla:.3f} TC-F1={f1_tc:.3f} (P={p_tc:.3f}, R={r_tc:.3f}) DER(O)={der_o if der_o is not None else 'NA'} DER={der_w if der_w is not None else 'NA'}")

    # Aggregate by show and overall (macro average per episode)
    from collections import defaultdict
    agg = defaultdict(list)
    for row in per_ep:
        agg[row['show']].append(row)
        agg['ALL'].append(row)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    print('\nSummary (macro-avg per episode):')
    for key, rows in agg.items():
        acc = avg([r['acc'] for r in rows])
        ppc = avg([r['ppc'] for r in rows])
        rpc = avg([r['rpc'] for r in rows])
        acc_h = avg([r['acc_hyp'] for r in rows])
        ppc_h = avg([r['ppc_hyp'] for r in rows])
        rpc_h = avg([r['rpc_hyp'] for r in rows])
        fla = avg([r['fla'] for r in rows])
        f1tc = avg([r['turn_f1'] for r in rows])
        ptc = avg([r['turn_p'] for r in rows])
        rtc = avg([r['turn_r'] for r in rows])
        der_o = avg([r['der_o'] for r in rows])
        der_w = avg([r['der'] for r in rows])
        print(f"  {key}: Acc(ref)={acc:.3f} Ppc(ref)={ppc:.3f} Rpc(ref)={rpc:.3f} | Acc(hyp)={acc_h:.3f} Ppc(hyp)={ppc_h:.3f} Rpc(hyp)={rpc_h:.3f} | FLA={fla:.3f} TC-F1={f1tc:.3f} (P={ptc:.3f}, R={rtc:.3f}) DER(O)={'NA' if der_o is None else f'{der_o:.3f}'} DER={'NA' if der_w is None else f'{der_w:.3f}'}")


if __name__ == '__main__':
    main()
