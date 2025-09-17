import os
import sys
import csv
import math
import argparse
from collections import defaultdict

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    HAVE_PYANNOTE = True
except Exception:
    HAVE_PYANNOTE = False

import pickle


def load_baseline_segments(baseline_pkl):
    segs = pickle.load(open(baseline_pkl, 'rb'))
    out = []
    for s in segs:
        st = float(s.get('start', 0.0))
        en = float(s.get('end', 0.0))
        lab = str(s.get('speaker', 'spk'))
        if en > st:
            out.append((st, en, lab))
    return out


def load_google_csv(csv_path):
    out = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = row.get('start') or row.get('Start') or row.get('start_time')
            en = row.get('end') or row.get('End') or row.get('end_time')
            sp = row.get('speakerTag') or row.get('speaker') or row.get('spk')
            try:
                st = float(st); en = float(en)
            except Exception:
                continue
            if en > st:
                out.append((st, en, str(sp)))
    return out


def to_annotation(segs):
    ann = Annotation()
    for s, e, l in segs:
        ann[Segment(s, e)] = l
    return ann


def load_refined_segments(ep_root):
    refined_p = os.path.join(ep_root, 'result', 'refined_diriazation.pckl')
    if not os.path.isfile(refined_p):
        raise FileNotFoundError(f"Missing refined output: {refined_p}")
    items = pickle.load(open(refined_p, 'rb'))
    out = []
    for s in items:
        st = float(s.get('start', 0.0))
        en = float(s.get('end', 0.0))
        lab = s.get('speaker')
        if lab is None:
            lab = 'UNK'
        lab = str(lab)
        if en > st:
            out.append((st, en, lab))
    return out

def load_refined_asr_segments(ep_root):
    refined_p = os.path.join(ep_root, 'result', 'refined_diriazation_asr.pckl')
    if not os.path.isfile(refined_p):
        raise FileNotFoundError(f"Missing refined ASR output: {refined_p}")
    items = pickle.load(open(refined_p, 'rb'))
    out = []
    for s in items:
        st = float(s.get('start', 0.0))
        en = float(s.get('end', 0.0))
        lab = s.get('speaker')
        if lab is None:
            lab = 'UNK'
        lab = str(lab)
        if en > st:
            out.append((st, en, lab))
    return out


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def greedy_mapping(ref, pred):
    # Build overlap matrix (ref_labels x pred_labels)
    ref_labels = sorted({l for _, _, l in ref})
    pred_labels = sorted({l for _, _, l in pred})
    idx_r = {l: i for i, l in enumerate(ref_labels)}
    idx_p = {l: i for i, l in enumerate(pred_labels)}
    M = [[0.0 for _ in pred_labels] for __ in ref_labels]

    for rs, re, rl in ref:
        for ps, pe, pl in pred:
            ov = overlap((rs, re), (ps, pe))
            if ov > 0:
                M[idx_r[rl]][idx_p[pl]] += ov

    used_r, used_p, mapping = set(), set(), {}
    while True:
        best, bestv = None, 0.0
        for i in range(len(ref_labels)):
            if i in used_r:
                continue
            for j in range(len(pred_labels)):
                if j in used_p:
                    continue
                if M[i][j] > bestv:
                    bestv = M[i][j]
                    best = (i, j)
        if not best or bestv <= 0.0:
            break
        i, j = best
        used_r.add(i)
        used_p.add(j)
        mapping[pred_labels[j]] = ref_labels[i]
    return mapping, ref_labels, pred_labels


def frame_level_agreement(ref, pred, mapping, step=0.01, collar=0.25):
    start = min(min(s for s, _, _ in ref), min(s for s, _, _ in pred))
    end = max(max(e for _, e, _ in ref), max(e for _, e, _ in pred))
    boundaries = []
    for s, e, _ in ref:
        boundaries.append(s)
        boundaries.append(e)

    def near_boundary(t):
        return any(abs(t - b) <= collar for b in boundaries)

    def lab_at(t, segs):
        for s, e, l in segs:
            if s <= t < e:
                return l
        return None

    ref_total = 0.0
    correct = 0.0
    t = start
    while t < end:
        rl = lab_at(t, ref)
        pl = lab_at(t, pred)
        mpl = mapping.get(pl)
        if rl is not None and not near_boundary(t):
            ref_total += step
            if mpl == rl:
                correct += step
        t += step
    return correct / ref_total if ref_total > 0 else 0.0


def turn_change_f1(ref, pred, tol=0.5):
    def cps(segs):
        out = []
        prev = None
        for s, e, l in sorted(segs):
            if prev is None:
                prev = (s, e, l)
                continue
            if l != prev[2]:
                out.append(s)
            prev = (s, e, l)
        return out

    rcps = sorted(cps(ref))
    pcps = sorted(cps(pred))
    used = [False] * len(pcps)
    tp = 0
    import bisect
    for r in rcps:
        i = bisect.bisect_left(pcps, r)
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(pcps) and (not used[j]) and abs(pcps[j] - r) <= tol:
                used[j] = True
                tp += 1
                break
    matched = tp
    fp = len(pcps) - matched
    fn = len(rcps) - matched
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def eval_single(ref, pred):
    mapping, ref_labels, pred_labels = greedy_mapping(ref, pred)
    fra = frame_level_agreement(ref, pred, mapping, step=0.01, collar=0.25)
    prec, rec, f1 = turn_change_f1(ref, pred, tol=0.5)
    der_skip = der_with = None
    if HAVE_PYANNOTE:
        ann_ref = to_annotation(ref)
        pred_mapped = [(s, e, mapping.get(l, f'UNK:{l}')) for s, e, l in pred]
        ann_hyp = to_annotation(pred_mapped)
        metric_skip = DiarizationErrorRate(collar=0.25, skip_overlap=True)
        metric_with = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        der_skip = metric_skip(ann_ref, ann_hyp)
        der_with = metric_with(ann_ref, ann_hyp)
    return {
        'frame_agreement': fra,
        'turn_precision': prec,
        'turn_recall': rec,
        'turn_f1': f1,
        'der_skip_overlap': der_skip,
        'der_with_overlap': der_with,
        'mapping': mapping,
        'ref_speakers': len(ref_labels),
        'pred_speakers': len(pred_labels),
    }


def eval_episode(ep_root, compare='both'):
    base_pkl = os.path.join(ep_root, 'result', 'baseline_whisperx', 'segments.pkl')
    g_csv = os.path.join(ep_root, 'result', 'google_stt', 'segments.csv')
    if not os.path.isfile(g_csv):
        raise FileNotFoundError('Missing google files')
    ref = load_google_csv(g_csv)

    out = {}
    if compare in ('baseline', 'both'):
        if not os.path.isfile(base_pkl):
            raise FileNotFoundError('Missing baseline segments.pkl')
        pred_b = load_baseline_segments(base_pkl)
        out['baseline'] = eval_single(ref, pred_b)
    if compare in ('refined', 'both'):
        try:
            pred_r = load_refined_segments(ep_root)
            out['refined'] = eval_single(ref, pred_r)
        except FileNotFoundError:
            pass
    if compare in ('refined_asr', 'both', 'all'):
        try:
            pred_ra = load_refined_asr_segments(ep_root)
            out['refined_asr'] = eval_single(ref, pred_ra)
        except FileNotFoundError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', type=str, help='Episode root, e.g., multi_human_talking_dataset/fallowshow/1')
    ap.add_argument('--dataset_root', type=str, help='If set, evaluate all episodes under this root that have google_stt/segments.csv')
    ap.add_argument('--compare', type=str, choices=['baseline','refined','refined_asr','both','all'], default='both')
    args = ap.parse_args()

    episodes = []
    if args.dataset_root:
        for show in sorted(os.listdir(args.dataset_root)):
            show_dir = os.path.join(args.dataset_root, show)
            if not os.path.isdir(show_dir):
                continue
            for ep in sorted(os.listdir(show_dir)):
                ep_dir = os.path.join(show_dir, ep)
                if not os.path.isdir(ep_dir):
                    continue
                g_csv = os.path.join(ep_dir, 'result', 'google_stt', 'segments.csv')
                if os.path.isfile(g_csv):
                    episodes.append(ep_dir)
    elif args.episode:
        episodes = [args.episode]
    else:
        print('Provide --episode or --dataset_root')
        sys.exit(1)

    # Accumulators
    sums = {
        'baseline': defaultdict(float),
        'refined': defaultdict(float),
        'refined_asr': defaultdict(float),
    }
    counts = {'baseline': 0, 'refined': 0, 'refined_asr': 0}

    for ep in episodes:
        try:
            res = eval_episode(ep, compare=args.compare)
        except Exception as e:
            print(f"ERROR {ep}: {e}")
            continue
        print(f"Episode: {ep}")
        for label in ('baseline','refined','refined_asr'):
            if label not in res:
                continue
            r = res[label]
            print(f"  [{label}] Frame-agree: {r['frame_agreement']:.3f} | Turn-F1: {r['turn_f1']:.3f} (P={r['turn_precision']:.3f}, R={r['turn_recall']:.3f})")
            if r['der_skip_overlap'] is not None:
                print(f"  [{label}] DER skip/with overlap: {r['der_skip_overlap']:.3f}/{r['der_with_overlap']:.3f}")
            sums[label]['frame_agreement'] += r['frame_agreement']
            sums[label]['turn_precision'] += r['turn_precision']
            sums[label]['turn_recall'] += r['turn_recall']
            sums[label]['turn_f1'] += r['turn_f1']
            if r['der_skip_overlap'] is not None:
                sums[label]['der_skip_overlap'] += r['der_skip_overlap']
                sums[label]['der_with_overlap'] += r['der_with_overlap']
            counts[label] += 1

    # Macro averages
    for label in ('baseline','refined','refined_asr'):
        n = counts[label]
        if n == 0:
            continue
        print(f"\nMacro-avg over {n} episodes [{label}] :")
        print(f"  Frame-agree: {sums[label]['frame_agreement']/n:.3f}")
        print(f"  Turn-F1: {sums[label]['turn_f1']/n:.3f} (P={sums[label]['turn_precision']/n:.3f}, R={sums[label]['turn_recall']/n:.3f})")
        if 'der_skip_overlap' in sums[label]:
            print(f"  DER skip/with overlap: {sums[label]['der_skip_overlap']/n:.3f}/{sums[label]['der_with_overlap']/n:.3f}")


if __name__ == '__main__':
    main()
