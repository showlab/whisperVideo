import os
import sys
import argparse
import glob
import pickle
from typing import List, Dict, Optional, Tuple
import re


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_episodes(dataset_root: str, allowed_shows: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """Return list of (show, episode, ep_dir). If allowed_shows provided, filter by show name."""
    eps: List[Tuple[str, str, str]] = []
    shows = [d for d in sorted(os.listdir(dataset_root)) if os.path.isdir(os.path.join(dataset_root, d))]
    for show in shows:
        if allowed_shows and show not in allowed_shows:
            continue
        show_dir = os.path.join(dataset_root, show)
        for ep in sorted(os.listdir(show_dir)):
            ep_dir = os.path.join(show_dir, ep)
            if os.path.isdir(ep_dir):
                eps.append((show, ep, ep_dir))
    return eps


def compute_K_from_tracks_identity(tracks_identity_path: str) -> Optional[int]:
    obj = load_pickle(tracks_identity_path)
    if not isinstance(obj, list) or not obj:
        return None
    ids = set()
    for t in obj:
        if not isinstance(t, dict):
            continue
        ident = t.get('identity')
        # Use only VID_* identities (from global constrained clustering)
        if isinstance(ident, str) and ident.startswith('VID_'):
            ids.add(ident)
    if not ids:
        return None
    return len(ids)


def derive_min_max(K: int) -> Tuple[int, int]:
    K = max(1, int(K))
    # Policy: min = 1, max = K
    return 1, K


def parse_log_for_k(log_path: str) -> Optional[Tuple[int, int, int]]:
    """Parse experiment_log.txt to extract (K, min, max) from lines like:
    'whisperx diarization with min/max speakers = 3/5 (K=4)'
    Returns tuple (K, min_spk, max_spk) from the last occurrence.
    """
    if not os.path.isfile(log_path):
        return None
    pat = re.compile(r"min/max speakers\s*=\s*(\d+)\/(\d+)\s*\(K=(\d+)\)")
    found = None
    with open(log_path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                min_s = int(m.group(1))
                max_s = int(m.group(2))
                K = int(m.group(3))
                found = (K, min_s, max_s)
    return found


def main():
    parser = argparse.ArgumentParser(description='Report visual K and min/max speakers for episodes.')
    parser.add_argument('--dataset_root', type=str,
                        default='/workspace/siyuan/siyuan/whisperv_proj/multi_human_talking_dataset')
    parser.add_argument('--shows', type=str, nargs='*', default=['fallowshow', 'latenightshow'],
                        help='Filter by show names (folder names)')
    parser.add_argument('--only_existing', action='store_true',
                        help='Only report episodes that already have result/tracks_identity.pckl; skip others')
    parser.add_argument('--recompute_identities', action='store_true',
                        help='Recompute visual identities (VID_*) via identity_cluster for each episode before reporting K')
    parser.add_argument('--write', action='store_true',
                        help='When --recompute_identities is set, write tracks_identity.pckl back to result/')
    parser.add_argument('--device', type=str, default='cuda', help='Device for embedding when recomputing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding when recomputing')
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not os.path.isdir(dataset_root):
        print(f'ERROR: dataset_root not found: {dataset_root}')
        sys.exit(1)

    episodes = find_episodes(dataset_root, allowed_shows=args.shows)
    if not episodes:
        print('No episodes found matching filters.')
        sys.exit(1)

    print('show,episode,K,min,max,status')
    per_show = {s: [] for s in args.shows}
    for show, ep, ep_dir in episodes:
        res_dir = os.path.join(ep_dir, 'result')
        ti = os.path.join(res_dir, 'tracks_identity.pckl')
        status = None
        try:
            if args.recompute_identities:
                # Normalize vidTracks then cluster identities freshly
                try:
                    try:
                        from .identity_cluster import cluster_visual_identities  # when run as package
                    except Exception:
                        from identity_cluster import cluster_visual_identities
                except Exception as e:
                    print(f'{show},{ep},NA,NA,NA,error:import_cluster:{e}')
                    continue

                # Build vidTracks with cropFile alignment
                tracks_p = os.path.join(res_dir, 'tracks.pckl')
                crop_dir = os.path.join(ep_dir, 'crop')
                if not os.path.exists(tracks_p):
                    print(f'{show},{ep},NA,NA,NA,missing-tracks.pckl')
                    continue
                raw_tracks = load_pickle(tracks_p)
                crop_avis = sorted(glob.glob(os.path.join(crop_dir, '*.avi')))
                if not crop_avis:
                    print(f'{show},{ep},NA,NA,NA,missing-crop')
                    continue
                vidTracks = []
                if isinstance(raw_tracks, list) and raw_tracks and isinstance(raw_tracks[0], dict) and 'cropFile' in raw_tracks[0]:
                    vidTracks = raw_tracks
                else:
                    if not isinstance(raw_tracks, list) or len(raw_tracks) != len(crop_avis):
                        print(f'{show},{ep},NA,NA,NA,mismatch-tracks-crop')
                        continue
                    for i, t in enumerate(raw_tracks):
                        if not isinstance(t, dict):
                            print(f'{show},{ep},NA,NA,NA,unsupported-track-entry')
                            vidTracks = []
                            break
                        if 'track' in t:
                            tr = t['track']
                        elif all(k in t for k in ('frame', 'bbox')):
                            tr = t
                        else:
                            print(f'{show},{ep},NA,NA,NA,missing-track-keys')
                            vidTracks = []
                            break
                        base = os.path.splitext(crop_avis[i])[0]
                        vidTracks.append({'track': tr, 'cropFile': base})
                if not vidTracks:
                    continue
                # If scores.pckl exists, load to enable ASD gating; else recompute without gating
                scores_p = os.path.join(res_dir, 'scores.pckl')
                scores = None
                if os.path.exists(scores_p):
                    try:
                        scores = load_pickle(scores_p)
                    except Exception as e:
                        scores = None
                annotated = cluster_visual_identities(vidTracks, device=args.device, batch_size=args.batch_size,
                                                      scores_list=scores)
                # Compute K_used by speaking-time coverage (>=90%)
                id_speaking = {}
                total = 0
                if scores is not None:
                    for i, t in enumerate(annotated):
                        ident = t.get('identity')
                        if not (isinstance(ident, str) and ident.startswith('VID_')):
                            continue
                        if i >= len(scores):
                            continue
                        sc = scores[i]
                        if not isinstance(sc, (list, tuple)) or len(sc) == 0:
                            continue
                        speak = sum(1 for v in sc if v > 0)
                        if speak <= 0:
                            continue
                        id_speaking[ident] = id_speaking.get(ident, 0) + speak
                        total += speak
                # visual count fallback if no ASD
                if total > 0 and id_speaking:
                    items = sorted(id_speaking.items(), key=lambda x: x[1], reverse=True)
                    cum = 0
                    K_cov = 0
                    for _, dur in items:
                        cum += dur
                        K_cov += 1
                        if cum >= 0.90 * total:
                            break
                    K = max(1, K_cov)
                else:
                    fresh_ids = {t.get('identity') for t in annotated if isinstance(t.get('identity'), str) and t.get('identity').startswith('VID_')}
                    K = max(1, len(fresh_ids))
                if args.write:
                    load_pickle  # satisfy linter
                    import pickle
                    with open(ti, 'wb') as f:
                        pickle.dump(annotated, f)
                min_spk, max_spk = derive_min_max(K)
                print(f'{show},{ep},{K},{min_spk},{max_spk},recomputed')
                if show in per_show:
                    per_show[show].append((K, min_spk, max_spk))
                continue

            if os.path.exists(ti):
                K = compute_K_from_tracks_identity(ti)
                if K is not None:
                    min_spk, max_spk = derive_min_max(K)
                    print(f'{show},{ep},{K},{min_spk},{max_spk},from-tracks')
                    if show in per_show:
                        per_show[show].append((K, min_spk, max_spk))
                    continue
                else:
                    status = 'empty-ids'
            # Fallback to log parsing if tracks_identity unavailable or empty
            log_path = os.path.join(res_dir, 'experiment_log.txt')
            parsed = parse_log_for_k(log_path)
            if parsed is not None:
                K, min_spk, max_spk = parsed
                print(f'{show},{ep},{K},{min_spk},{max_spk},from-log')
                if show in per_show:
                    per_show[show].append((K, min_spk, max_spk))
            else:
                if args.only_existing:
                    continue
                print(f'{show},{ep},NA,NA,NA,{status or "missing-tracks_identity"}')
        except Exception as e:
            print(f'{show},{ep},NA,NA,NA,error:{e}')

    # Summary per show
    for show in args.shows:
        vals = per_show.get(show, [])
        if not vals:
            continue
        Ks = [v[0] for v in vals]
        import numpy as np
        print(f'\nSummary[{show}] episodes={len(vals)}: K_mean={np.mean(Ks):.2f} K_median={np.median(Ks):.2f} K_min={min(Ks)} K_max={max(Ks)}')


if __name__ == '__main__':
    main()
