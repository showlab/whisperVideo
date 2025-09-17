import os
import sys
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch.multiprocessing as mp


FPS = 25.0


@dataclass
class Episode:
    root: str
    show: str
    episode: str

    @property
    def ep_dir(self) -> str:
        return os.path.join(self.root, self.show, self.episode)

    @property
    def result_dir(self) -> str:
        return os.path.join(self.ep_dir, 'result')

    @property
    def crop_dir(self) -> str:
        return os.path.join(self.ep_dir, 'crop')

    def assert_layout(self):
        if not os.path.isdir(self.result_dir):
            raise FileNotFoundError(f"Missing result dir: {self.result_dir}")
        if not os.path.isdir(self.crop_dir):
            raise FileNotFoundError(f"Missing crop dir: {self.crop_dir}")


def _find_episodes(dataset_root: str) -> List[Episode]:
    eps: List[Episode] = []
    entries = [os.path.join(dataset_root, d) for d in sorted(os.listdir(dataset_root)) if os.path.isdir(os.path.join(dataset_root, d))]
    is_episode = lambda p: os.path.isdir(os.path.join(p, 'result')) and os.path.isdir(os.path.join(p, 'crop'))
    episode_like = [p for p in entries if is_episode(p)]
    if episode_like:
        for ep in episode_like:
            parent, episode = os.path.split(ep.rstrip('/'))
            show = os.path.basename(parent)
            eps.append(Episode(root=os.path.dirname(parent), show=show, episode=episode))
        return eps
    for show_dir in entries:
        show = os.path.basename(show_dir)
        for e in sorted(os.listdir(show_dir)):
            ep_dir = os.path.join(show_dir, e)
            if os.path.isdir(ep_dir) and is_episode(ep_dir):
                eps.append(Episode(root=dataset_root, show=show, episode=e))
    if not eps:
        raise RuntimeError(f"No episodes found under {dataset_root}")
    return eps


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def _load_pickle(path: str):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_pickle(obj, path: str):
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _assign_to_tracks(vid_tracks: List[Dict], person_tracks: List[Dict], fps: float, tau: float) -> List[Dict]:
    pts: List[Tuple[str, Tuple[float, float]]] = []
    for pt in person_tracks:
        tid = pt.get('id')
        for s, e in pt.get('segments', []):
            s = float(s); e = float(e)
            if e > s:
                pts.append((tid, (s, e)))

    out = []
    for t in vid_tracks:
        if not isinstance(t, dict):
            raise RuntimeError('tracks.pckl entry is not a dict')
        if 'track' in t and isinstance(t['track'], dict):
            frames = t['track'].get('frame')
        elif all(k in t for k in ('frame', 'bbox')):
            frames = t.get('frame')
        else:
            raise RuntimeError('Unsupported track entry: missing frame list')

        if frames is None or (hasattr(frames, '__len__') and len(frames) == 0):
            t2 = dict(t); t2['identity'] = None; out.append(t2); continue
        s = min(frames) / fps
        e = (max(frames) + 1) / fps
        dur = max(1e-6, e - s)
        best_id, best_ov = None, 0.0
        for tid, seg in pts:
            ov = _overlap((s, e), seg)
            if ov > best_ov:
                best_ov, best_id = ov, tid
        t2 = dict(t)
        if best_id is not None and (best_ov / dur) >= tau:
            t2['identity'] = best_id
        else:
            t2['identity'] = None
        out.append(t2)
    return out


def _normalize_tracks(ep: Episode, tracks) -> List[Dict]:
    if isinstance(tracks, list) and tracks and isinstance(tracks[0], dict) and 'cropFile' in tracks[0]:
        return tracks
    crop_avis = sorted(glob.glob(os.path.join(ep.crop_dir, '*.avi')))
    if not crop_avis:
        raise FileNotFoundError(f"No crop clips found in {ep.crop_dir}")
    if not isinstance(tracks, list) or len(tracks) != len(crop_avis):
        raise RuntimeError('tracks.pckl count does not match crop/*.avi files and no cropFile present')
    normalized = []
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            raise RuntimeError('Unsupported tracks.pckl entry (expected dict)')
        if 'track' in t:
            tr = t['track']
        elif all(k in t for k in ('frame', 'bbox')):
            tr = t
        else:
            raise RuntimeError("tracks.pckl missing per-track 'track' or ('frame','bbox') keys")
        base = os.path.splitext(crop_avis[i])[0]
        normalized.append({'track': tr, 'cropFile': base})
    return normalized


def _worker(args):
    ep: Episode
    ep, tau, fps = args
    ep.assert_layout()
    gcp_json = os.path.join(ep.result_dir, 'gcp_visual_identity.json')
    tracks_p = os.path.join(ep.result_dir, 'tracks.pckl')
    out_p = os.path.join(ep.result_dir, 'tracks_identity_gcp.pckl')
    if not os.path.exists(gcp_json):
        raise FileNotFoundError(f"Missing {gcp_json}")
    if not os.path.exists(tracks_p):
        raise FileNotFoundError(f"Missing {tracks_p}")
    with open(gcp_json, 'r') as f:
        data = json.load(f)
    person_tracks = data.get('tracks', [])
    tracks = _load_pickle(tracks_p)
    norm = _normalize_tracks(ep, tracks)
    assigned = _assign_to_tracks(norm, person_tracks, fps=fps, tau=tau)
    _save_pickle(assigned, out_p)
    labeled = sum(1 for a in assigned if a.get('identity'))
    return (ep.ep_dir, len(assigned), labeled)


def main():
    parser = argparse.ArgumentParser(description='Offline: map GCP visual identities to face tracks for all episodes.')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root dir of dataset or a single episode dir')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Minimum overlap ratio of a track with a GCP person segment to assign identity')
    parser.add_argument('--fps', type=float, default=FPS)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    eps = _find_episodes(args.dataset_root)

    tasks = [(ep, float(args.tau), float(args.fps)) for ep in eps]

    mp.set_start_method('spawn', force=True)
    results: List[Tuple[str, int, int]] = []
    with mp.Pool(processes=max(1, int(args.num_workers))) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            results.append(res)
            ep_dir, n_total, n_labeled = res
            print(f"[GCP-MAP] {ep_dir}: labeled {n_labeled}/{n_total}")
            sys.stdout.flush()

    total_tracks = sum(n for _, n, _ in results)
    total_labeled = sum(l for _, _, l in results)
    print(f"All done. Tracks labeled: {total_labeled}/{total_tracks} ({(total_labeled/max(1,total_tracks))*100:.1f}%) across {len(results)} episodes.")


if __name__ == '__main__':
    main()
