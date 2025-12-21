#!/usr/bin/env python
"""
Convert SAM3 instance mask PNGs into faces.pckl for whisperv inference.

Input:
  --mask_dir: directory containing 6-digit frame PNGs (uint16 ids, 0=bg)
  --output: path to write faces.pckl (list of per-frame detections)

Output format matches inference_folder expectations:
  faces[p] = list of {'frame': int, 'bbox': [x1,y1,x2,y2], 'conf': 1.0}
"""
import argparse
import glob
import os
import pickle
from typing import List, Dict

import numpy as np
from PIL import Image


def masks_to_faces(mask_dir: str, fps_div: int = 1) -> List[List[Dict]]:
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")
    # infer frame indices from filenames
    indices = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            idx = int(stem)
        except ValueError:
            raise ValueError(f"Mask filename not numeric: {p}")
        indices.append(idx)
    max_idx = max(indices)
    max_orig = max_idx * fps_div
    faces = [[] for _ in range(max_orig + 1)]

    for p, idx in zip(paths, indices):
        arr = np.array(Image.open(p))
        ids = np.unique(arr)
        ids = ids[ids > 0]
        frame_faces = []
        for oid in ids:
            mask = arr == oid
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            # ensure non-zero area bbox
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            frame_faces.append({"frame": idx * fps_div, "bbox": [int(x1), int(y1), int(x2), int(y2)], "conf": 1.0})
        faces[idx * fps_div] = frame_faces
    return faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True, help="Directory of SAM3 mask PNGs")
    ap.add_argument("--output", required=True, help="Output faces.pckl path")
    ap.add_argument("--fps_div", type=int, default=1, help="Temporal downsample factor used when generating masks")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    faces = masks_to_faces(args.mask_dir, fps_div=max(1, int(args.fps_div)))
    with open(args.output, "wb") as f:
        pickle.dump(faces, f)
    print(f"Wrote faces to {args.output}, frames={len(faces)}")


if __name__ == "__main__":
    main()
