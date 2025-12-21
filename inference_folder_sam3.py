import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, json
import torch.nn.functional as F
from collections import defaultdict, Counter

# Ensure TF backend is disabled for transformers to avoid protobuf/TensorFlow issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import whisperx
import gc

import numpy as np
import pandas as pd
import pysrt 
from datetime import timedelta

from scipy import signal
import subprocess
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from talkNet import talkNet
from identity_verifier import IdentityVerifier
try:
    from identity_cluster import cluster_visual_identities
except Exception:
    # When run as module, fallback to relative import
    from .identity_cluster import cluster_visual_identities

warnings.filterwarnings("ignore")
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
try:
    # Silence OpenCV logs (including FFmpeg backend noise) where possible
    import cv2 as _cv_quiet
    if hasattr(_cv_quiet, 'utils') and hasattr(_cv_quiet.utils, 'logging'):
        _cv_quiet.utils.logging.setLogLevel(_cv_quiet.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# Central place to choose ffmpeg binary. Allows user to override from env to
# ensure the Python process uses the same ffmpeg that was tested in the shell.
# Default to system ffmpeg (/usr/bin/ffmpeg), which we verified has libx264.
_FFMPEG_BIN = os.environ.get('WHISPERV_FFMPEG_BIN', '/usr/bin/ffmpeg')
try:
    import av as _av_quiet  # type: ignore
    if hasattr(_av_quiet, 'logging'):
        _av_quiet.logging.set_level(_av_quiet.logging.ERROR)
except Exception:
    pass

def _ensure_cfr25(input_path: str, output_path: str, start: float = 0.0, duration: float = 0.0, threads: int = 4):
    """Top-level helper to re-encode input to 25fps CFR MP4 using libx264."""
    ss_to = ''
    if duration and duration > 0:
        ss_to = f" -ss {start:.3f} -to {start+duration:.3f}"
    cmd = (
        f"{_FFMPEG_BIN} -y -i {input_path}{ss_to} -r 25 -vsync cfr -pix_fmt yuv420p "
        f"-c:v libx264 -crf 18 -c:a aac -b:a 192k -threads {int(threads)} {output_path} -loglevel panic"
    )
    return subprocess.call(cmd, shell=True, stdout=None)

def _ffmpeg_dual_worker(task):
    """Top-level worker for ffmpeg tasks used with multiprocessing.

    task: tuple(kind, payload)
      - ('cfr', (in_path, out_path, start, duration, threads))
      - ('aud', (in_path, out_wav, start, duration, threads))
    Returns: (kind, rc)
    """
    kind, payload = task
    try:
        if kind == 'cfr':
            in_p, out_p, st, du, th = payload
            return ('cfr', _ensure_cfr25(in_p, out_p, float(st), float(du), int(th)))
        elif kind == 'aud':
            in_p, out_wav, st, du, th = payload
            ss_to = ''
            if du and float(du) > 0:
                ss_to = f" -ss {float(st):.3f} -to {float(st)+float(du):.3f}"
            cmd = (
                f"{_FFMPEG_BIN} -y -i {in_p}{ss_to} -c:a pcm_s16le -ac 1 -vn -threads {int(th)} -ar 16000 {out_wav} -loglevel panic"
            )
            rc = subprocess.call(cmd, shell=True, stdout=None)
            return ('aud', rc)
        else:
            return (kind, 1)
    except Exception:
        return (kind, 1)

parser = argparse.ArgumentParser(description = "Demo")

# parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="/workspace/siyuan/siyuan/whisperv_proj/data/video/Frasier",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=-1,   help='FFmpeg/IO threads (-1 = auto CPU count)')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')
parser.add_argument('--facedetBatch',         type=int,   default=-1,  help='(Unused in SAM3-only pipeline)')
parser.add_argument('--detBackend',           type=str,   default='sam3', choices=['sam3'], help='Face detector backend (SAM3 only in this script)')
parser.add_argument('--detInputW',            type=int,   default=288, help='(Unused in SAM3-only pipeline)')
parser.add_argument('--detInputH',            type=int,   default=160, help='(Unused in SAM3-only pipeline)')
parser.add_argument('--idBatch',              type=int,   default=-1,   help='Batch size for identity embedding (-1 = auto max)')
parser.add_argument('--asdBatch',             type=int,   default=-1,   help='Batch size for ASD window inference (-1 = auto max)')
parser.add_argument('--cropWorkers',          type=int,   default=8,    help='Parallel workers for audio cut + mux per track')
parser.add_argument('--sceneWorkers',         type=int,   default=6,    help='Parallel workers for in-memory ASD by scene')
parser.add_argument('--sceneMinSec',         type=float, default=1.0,  help='Minimum scene length in seconds (detector min_scene_len)')

# Panel rendering (Skia) options
parser.add_argument('--renderPanel',         action='store_true', default=True, help='Render a Skia side panel with identity chat-like messages (default on)')
parser.add_argument('--panelWidthRatio',     type=float, default=0.28, help='Right panel width ratio relative to video width (0.2–0.4 recommended)')
parser.add_argument('--panelMaxItems',       type=int,   default=6,    help='Max messages to show in panel')
parser.add_argument('--panelWorkers',        type=int,   default=1,    help='Parallel workers for panel rendering (CPU, torch.multiprocessing)')
parser.add_argument('--panelTheme',          type=str,   default='glass', choices=['glass','dark'], help='Panel theme style')
parser.add_argument('--panelCompose',        type=str,   default='raw', choices=['subtitles','raw'], help='Compose panel onto which base: subtitles or raw video')
parser.add_argument('--subtitle',            action='store_true', default=False, help='Also render panel messages as bottom subtitles on the left video')

# Diarization parallelization
parser.add_argument('--diarWorkers',         type=int,   default=1,    help='Parallel workers for diarization (GPU, torch.multiprocessing)')
parser.add_argument('--diarWindowSec',       type=float, default=60.0, help='Diarization window length in seconds')
parser.add_argument('--diarOverlapSec',      type=float, default=3.0,  help='Diarization window overlap in seconds')

# SAM3 parallel processing options
parser.add_argument('--sam3Parallel',        action='store_true', default=True, help='Enable parallel SAM3 processing with temporal chunking (default on)')
parser.add_argument('--sam3ChunkSec',        type=float, default=30.0, help='SAM3 chunk duration in seconds (default 30s)')
parser.add_argument('--sam3OverlapSec',      type=float, default=3.0,  help='SAM3 chunk overlap in seconds for track continuity (default 3s)')

args, unknown  = parser.parse_known_args()

def _auto_set_max_batches(a):
    """Set large batch sizes by default when user didn't specify (-1)."""
    try:
        ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        ng = 0
    if ng <= 0:
        if getattr(a, 'facedetBatch', 0) <= 0:
            a.facedetBatch = 128
        if getattr(a, 'idBatch', 0) <= 0:
            a.idBatch = 128
        if getattr(a, 'asdBatch', 0) <= 0:
            a.asdBatch = 128
        return a
    if getattr(a, 'facedetBatch', 0) <= 0:
        a.facedetBatch = int(min(2048, 256 * ng))
    if getattr(a, 'idBatch', 0) <= 0:
        a.idBatch = int(min(4096, 256 * ng))
    if getattr(a, 'asdBatch', 0) <= 0:
        a.asdBatch = int(min(4096, 256 * ng))
    return a

args = _auto_set_max_batches(args)

def _auto_ffmpeg_threads(a):
    try:
        th = int(getattr(a, 'nDataLoaderThread', -1))
    except Exception:
        th = -1
    if th is None or th <= 0:
        try:
            import os
            a.nDataLoaderThread = max(1, int(os.cpu_count() or 8))
        except Exception:
            a.nDataLoaderThread = 8
    return a

args = _auto_ffmpeg_threads(args)

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)


class _StageTimer:
    """Append-only JSONL timing logger.

    Usage:
        timer = _StageTimer(log_path, meta)
        with timer.timer('stage_name'):
            ...
        # or timer.record(name, start, end)
    """
    def __init__(self, log_path: str, meta: dict | None = None):
        self.log_path = os.path.abspath(log_path)
        self.meta = dict(meta or {})
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    class _Ctx:
        def __init__(self, parent, name: str, extra: dict | None = None):
            self.parent = parent
            self.name = name
            self.extra = dict(extra or {})
            self.t0 = None
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter()
            self.parent.record(self.name, self.t0, t1, self.extra)

    def timer(self, name: str, extra: dict | None = None):
        return _StageTimer._Ctx(self, name, extra)

    def record(self, name: str, start: float, end: float, extra: dict | None = None):
        rec = {
            'stage': str(name),
            'ts_start': float(start),
            'ts_end': float(end),
            'elapsed_sec': float(max(0.0, end - start)),
        }
        if isinstance(self.meta, dict):
            rec.update(self.meta)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in rec:
                    rec[k] = v
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Must not break pipeline
            pass


def _ensure_chinese_font():
    """Ensure a CJK font exists for ffmpeg/libass to render Chinese subtitles.

    Priority:
    1) Use env CHINESE_FONT_PATH if provided (requires a valid file). Optional CHINESE_FONT_NAME for force_style.
    2) Use bundled font at whisperv/fonts/NotoSansCJKsc-Regular.otf; download it if missing.
    Returns (fonts_dir_abs, font_name_or_None). Raises RuntimeError on failure.
    """
    # 1) User-specified font path
    env_font_path = os.environ.get('CHINESE_FONT_PATH', '').strip()
    env_font_name = os.environ.get('CHINESE_FONT_NAME', '').strip() or None
    if env_font_path:
        if not os.path.isfile(env_font_path):
            raise RuntimeError(f"CHINESE_FONT_PATH set but file not found: {env_font_path}")
        return os.path.abspath(os.path.dirname(env_font_path)), env_font_name

    # 2) Bundled Noto Sans CJK SC Regular
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(_THIS_DIR, 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    font_path = os.path.join(fonts_dir, 'NotoSansCJKsc-Regular.otf')
    if not os.path.isfile(font_path):
        # Download from official repo (large file). Fail loudly on error.
        url = 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'
        import urllib.request
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            # cleanup partial file
            try:
                if os.path.exists(font_path):
                    os.remove(font_path)
            except Exception:
                pass
            raise RuntimeError(f"Failed to download Chinese font: {e}")
        if not os.path.isfile(font_path) or os.path.getsize(font_path) < 1024 * 1024:
            raise RuntimeError("Downloaded font file seems invalid or too small.")
    # Internal name for this font
    return os.path.abspath(fonts_dir), 'Noto Sans CJK SC'

# ============================================================================
# TEMPORAL CHUNKING PARALLEL SAM3 PROCESSING
# ============================================================================

def _get_video_info(video_path):
    """Get video duration, fps, and frame count using ffprobe."""
    import subprocess
    import json as _json
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,nb_frames',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        data = _json.loads(out)

        # Get duration (prefer format duration, fallback to stream)
        duration = 0.0
        if 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
        elif 'streams' in data and len(data['streams']) > 0:
            if 'duration' in data['streams'][0]:
                duration = float(data['streams'][0]['duration'])

        # Get fps
        fps = 25.0
        if 'streams' in data and len(data['streams']) > 0:
            r_rate = data['streams'][0].get('r_frame_rate', '25/1')
            if '/' in r_rate:
                num, den = r_rate.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 25.0
            else:
                fps = float(r_rate)

        # Get frame count
        nb_frames = 0
        if 'streams' in data and len(data['streams']) > 0:
            nb_frames = int(data['streams'][0].get('nb_frames', 0))
        if nb_frames == 0 and duration > 0:
            nb_frames = int(duration * fps)

        return duration, fps, nb_frames
    except Exception:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = nb_frames / fps if fps > 0 else 0.0
        cap.release()
        return duration, fps, nb_frames


def _segment_video_for_parallel(video_path, output_dir, chunk_duration_sec=120.0, overlap_sec=5.0, fps=25.0, threads=4):
    """
    Segment video into temporal chunks for parallel processing.

    Args:
        video_path: Input video path
        output_dir: Directory to store chunk files
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between consecutive chunks in seconds
        fps: Video FPS
        threads: FFmpeg threads

    Returns:
        List of tuples: [(chunk_path, start_frame, end_frame, chunk_idx), ...]
    """
    duration, vid_fps, total_frames = _get_video_info(video_path)
    if duration <= 0:
        raise RuntimeError(f"Cannot determine video duration: {video_path}")

    fps = vid_fps if vid_fps > 0 else fps

    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    chunk_idx = 0
    start_sec = 0.0

    while start_sec < duration:
        end_sec = min(start_sec + chunk_duration_sec, duration)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.mp4")

        # Calculate frame indices for this chunk
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # FFmpeg segment extraction (copy codec for speed, re-encode only if needed)
        cmd = (
            f"{_FFMPEG_BIN} -y -ss {start_sec:.3f} -i {video_path} "
            f"-t {end_sec - start_sec:.3f} -c:v libx264 -crf 18 -preset ultrafast "
            f"-c:a aac -threads {threads} {chunk_path} -loglevel error"
        )
        rc = subprocess.call(cmd, shell=True)
        if rc != 0 or not os.path.exists(chunk_path):
            raise RuntimeError(f"Failed to create chunk {chunk_idx} at {chunk_path}")

        chunks.append((chunk_path, start_frame, end_frame, chunk_idx))

        # Move to next chunk with overlap
        start_sec = end_sec - overlap_sec
        if start_sec >= duration - overlap_sec:
            break
        chunk_idx += 1

    return chunks


def _sam3_chunk_worker(task):
    """
    Worker function to process a single video chunk with SAM3.
    Must be defined at top level for multiprocessing.

    Args:
        task: tuple (chunk_path, start_frame, end_frame, chunk_idx, gpu_id, sam3_root)

    Returns:
        dict with chunk_idx, frame_outputs (frame_idx -> detections), H, W
    """
    chunk_path, start_frame, end_frame, chunk_idx, gpu_id, sam3_root = task

    import os
    import sys
    import numpy as np
    import cv2
    import torch

    # Add SAM3 to path
    if sam3_root not in sys.path:
        sys.path.insert(0, sam3_root)

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except Exception as e:
        import traceback
        return {'error': f'Failed to import SAM3: {e}\n{traceback.format_exc()}', 'chunk_idx': chunk_idx}

    try:
        # Build predictor on the specific GPU (pass real GPU ID directly)
        predictor = build_sam3_video_predictor(
            gpus_to_use=[gpu_id],  # Use the real GPU ID directly
            async_loading_frames=True,
            video_loader_type="pyav",
            compile=False,
            image_size=1008,
            recondition_every_nth_frame=64,
        )

        # Start session for this chunk
        resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": chunk_path,
        })
        session_id = resp["session_id"]

        # Add face prompt
        predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": "face",
        })

        # Propagate and collect outputs
        frame_outputs = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for res in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": None,
            }):
                fidx = int(res.get("frame_index", -1))
                out = res.get("outputs", None)
                if out is not None and fidx >= 0:
                    # Convert to numpy for pickling
                    frame_outputs[fidx] = {
                        'out_obj_ids': np.asarray(out.get('out_obj_ids', [])),
                        'out_probs': np.asarray(out.get('out_probs', [])),
                        'out_boxes_xywh': np.asarray(out.get('out_boxes_xywh', [])),
                        'out_binary_masks': np.asarray(out.get('out_binary_masks', [])),
                    }

        # Close session
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        try:
            predictor.shutdown()
        except Exception:
            pass

        # Get resolution
        H, W = None, None
        for out in frame_outputs.values():
            masks = out.get('out_binary_masks', None)
            if masks is not None and masks.ndim == 3 and masks.shape[0] > 0:
                H, W = int(masks.shape[1]), int(masks.shape[2])
                break
        if H is None:
            cap = cv2.VideoCapture(chunk_path)
            if cap.isOpened():
                ret, fr = cap.read()
                if ret and fr is not None:
                    H, W = fr.shape[0], fr.shape[1]
            cap.release()

        return {
            'chunk_idx': chunk_idx,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_outputs': frame_outputs,
            'H': H,
            'W': W,
            'error': None,
        }

    except Exception as e:
        import traceback
        return {
            'chunk_idx': chunk_idx,
            'error': f'SAM3 processing failed: {e}\n{traceback.format_exc()}',
        }
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()


def _merge_chunk_results(chunk_results, overlap_frames, total_frames, H, W, iou_thresh=0.5):
    """
    Merge results from multiple chunks, handling overlapping regions.

    Args:
        chunk_results: List of chunk result dicts (sorted by chunk_idx)
        overlap_frames: Number of overlapping frames between chunks
        total_frames: Total number of frames in original video
        H, W: Video resolution
        iou_thresh: IOU threshold for matching objects in overlap region

    Returns:
        faces: List of detections per frame
        sam3_masks_all: List of masks per frame
    """
    faces = [[] for _ in range(total_frames)]
    sam3_masks_all = [[] for _ in range(total_frames)]

    # Global object ID counter for remapping
    global_obj_id_counter = 0
    # Mapping from (chunk_idx, local_obj_id) -> global_obj_id
    obj_id_mapping = {}

    # Sort chunks by chunk_idx
    sorted_results = sorted(chunk_results, key=lambda x: x['chunk_idx'])

    for result in sorted_results:
        if result.get('error'):
            sys.stderr.write(f"Warning: chunk {result['chunk_idx']} failed: {result['error']}\n")
            continue

        chunk_idx = result['chunk_idx']
        start_frame = result['start_frame']
        frame_outputs = result.get('frame_outputs', {})

        for local_fidx, out in frame_outputs.items():
            global_fidx = start_frame + local_fidx
            if global_fidx >= total_frames:
                continue

            obj_ids = out.get('out_obj_ids', np.array([]))
            probs = out.get('out_probs', np.array([]))
            boxes_xywh = out.get('out_boxes_xywh', np.array([]))
            masks = out.get('out_binary_masks', np.array([]))

            if len(obj_ids) == 0:
                continue

            # Convert boxes to pixel XYXY
            if boxes_xywh.ndim == 2 and boxes_xywh.shape[1] == 4:
                xs = boxes_xywh[:, 0] * float(W)
                ys = boxes_xywh[:, 1] * float(H)
                ws = boxes_xywh[:, 2] * float(W)
                hs = boxes_xywh[:, 3] * float(H)
                boxes_xyxy = np.stack([xs, ys, xs + ws, ys + hs], axis=-1)
            else:
                continue

            for idx in range(len(obj_ids)):
                local_obj_id = int(obj_ids[idx])
                score = float(probs[idx])
                bbox = boxes_xyxy[idx]

                # Get or assign global object ID
                key = (chunk_idx, local_obj_id)
                if key not in obj_id_mapping:
                    # Check if this object matches one from previous chunk in overlap region
                    matched_global_id = None
                    if chunk_idx > 0 and local_fidx < overlap_frames:
                        # Look for matching object in previous chunk's overlap region
                        prev_chunk_idx = chunk_idx - 1
                        for (prev_cidx, prev_local_id), prev_global_id in obj_id_mapping.items():
                            if prev_cidx == prev_chunk_idx:
                                # Check IOU with existing detections at this frame
                                for existing_det in faces[global_fidx]:
                                    if existing_det.get('global_obj_id') == prev_global_id:
                                        iou = _compute_iou(bbox, existing_det['bbox'])
                                        if iou > iou_thresh:
                                            matched_global_id = prev_global_id
                                            break
                            if matched_global_id:
                                break

                    if matched_global_id is not None:
                        obj_id_mapping[key] = matched_global_id
                    else:
                        obj_id_mapping[key] = global_obj_id_counter
                        global_obj_id_counter += 1

                global_obj_id = obj_id_mapping[key]

                # Check if we already have this object at this frame (from overlap)
                already_exists = False
                for existing_det in faces[global_fidx]:
                    if existing_det.get('global_obj_id') == global_obj_id:
                        already_exists = True
                        break

                if already_exists:
                    continue  # Skip duplicate from overlap

                # Convert bbox to int
                bb = [int(round(v)) for v in bbox.tolist()]
                x1_i, y1_i, x2_i, y2_i = bb
                x1_i = max(0, min(x1_i, W - 1))
                y1_i = max(0, min(y1_i, H - 1))
                x2_i = max(0, min(x2_i, W))
                y2_i = max(0, min(y2_i, H))

                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                faces[global_fidx].append({
                    "frame": global_fidx,
                    "bbox": [x1_i, y1_i, x2_i, y2_i],
                    "conf": score,
                    "global_obj_id": global_obj_id,
                })

                if masks.ndim == 3 and idx < masks.shape[0]:
                    mask_arr = masks[idx]
                    if mask_arr.shape == (H, W) and mask_arr.any():
                        sam3_masks_all[global_fidx].append({
                            "bbox": [x1_i, y1_i, x2_i, y2_i],
                            "score": score,
                            "mask": mask_arr,
                            "obj_id": global_obj_id,
                        })

    return faces, sam3_masks_all


def _compute_iou(box1, box2):
    """Compute IOU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def inference_video_parallel(args, chunk_duration_sec=120.0, overlap_sec=5.0):
    """
    Parallel SAM3 face detection using temporal chunking.

    Args:
        args: Argument namespace with videoFilePath, pyworkPath, etc.
        chunk_duration_sec: Duration of each chunk in seconds (default 120s = 2min)
        overlap_sec: Overlap between chunks in seconds (default 5s)

    Returns:
        faces: List of detections per frame (same format as inference_video)
    """
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        raise RuntimeError('SAM3 face detector backend requires a CUDA GPU')

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError('No CUDA GPUs available')

    # Get SAM3 root path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sam3_root = os.path.abspath(os.path.join(this_dir, 'sam3-main'))
    if not os.path.isdir(sam3_root):
        raise RuntimeError(f'SAM3 root not found: {sam3_root}')

    # Get video info
    video_path = args.videoFilePath
    duration, fps, total_frames = _get_video_info(video_path)
    sys.stderr.write(f"Video info: duration={duration:.2f}s, fps={fps:.2f}, frames={total_frames}\n")

    # Create chunks directory
    chunks_dir = os.path.join(args.pyworkPath, 'sam3_chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    # Segment video
    sys.stderr.write(f"Segmenting video into chunks of {chunk_duration_sec}s with {overlap_sec}s overlap...\n")
    threads = int(getattr(args, 'nDataLoaderThread', 4))
    chunks = _segment_video_for_parallel(
        video_path, chunks_dir,
        chunk_duration_sec=chunk_duration_sec,
        overlap_sec=overlap_sec,
        fps=fps,
        threads=threads
    )
    sys.stderr.write(f"Created {len(chunks)} chunks\n")

    # Prepare tasks with GPU assignment (round-robin across GPUs)
    tasks = []
    for i, (chunk_path, start_frame, end_frame, chunk_idx) in enumerate(chunks):
        gpu_id = i % n_gpus  # Real GPU ID, not CUDA_VISIBLE_DEVICES
        tasks.append((chunk_path, start_frame, end_frame, chunk_idx, gpu_id, sam3_root))

    # Process chunks in parallel using spawn
    # Each worker uses a different real GPU via gpus_to_use parameter
    mp_ctx = mp.get_context('spawn')
    n_workers = min(len(tasks), n_gpus)  # One worker per GPU max

    sys.stderr.write(f"Processing {len(tasks)} chunks with {n_workers} parallel workers on {n_gpus} GPUs...\n")

    with mp_ctx.Pool(processes=n_workers) as pool:
        results = list(tqdm.tqdm(
            pool.imap(_sam3_chunk_worker, tasks),
            total=len(tasks),
            desc="SAM3 parallel chunks"
        ))

    # Check for errors and debug output
    for r in results:
        chunk_idx = r.get('chunk_idx', '?')
        if r.get('error'):
            sys.stderr.write(f"Chunk {chunk_idx} ERROR: {r['error']}\n")
        else:
            frame_outputs = r.get('frame_outputs', {})
            total_dets = sum(len(out.get('out_obj_ids', [])) for out in frame_outputs.values())
            sys.stderr.write(f"Chunk {chunk_idx}: {len(frame_outputs)} frames, {total_dets} detections\n")

    # Get resolution from first successful result
    H, W = None, None
    for r in results:
        if r.get('H') and r.get('W'):
            H, W = r['H'], r['W']
            break
    if H is None:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, fr = cap.read()
            if ret and fr is not None:
                H, W = fr.shape[0], fr.shape[1]
        cap.release()
    if H is None:
        raise RuntimeError("Could not determine video resolution")

    # Merge results
    overlap_frames = int(overlap_sec * fps)
    sys.stderr.write(f"Merging results from {len(results)} chunks (overlap={overlap_frames} frames)...\n")

    faces, sam3_masks_all = _merge_chunk_results(
        results, overlap_frames, total_frames, H, W
    )

    # Save results
    savePath = os.path.join(args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(faces, fil)
    masksPath = os.path.join(args.pyworkPath, "sam3_masks.pckl")
    with open(masksPath, "wb") as fil:
        pickle.dump(sam3_masks_all, fil)

    # Cleanup chunk files
    try:
        import shutil
        shutil.rmtree(chunks_dir)
    except Exception:
        pass

    sys.stderr.write(f"Parallel SAM3 processing complete: {sum(len(f) for f in faces)} detections\n")
    return faces


# ============================================================================
# END TEMPORAL CHUNKING PARALLEL SAM3 PROCESSING
# ============================================================================

def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    # Use scenedetect's native min_scene_len to enforce minimum scene duration.
    try:
        fps_eff = float(getattr(args, 'videoFps', 25.0)) if getattr(args, 'videoFps', None) is not None else 25.0
    except Exception:
        fps_eff = 25.0
    min_sec = max(0.0, float(getattr(args, 'sceneMinSec', 1.0)))
    min_frames = max(1, int(round(min_sec * fps_eff)))
    sceneManager.add_detector(ContentDetector(min_scene_len=min_frames))
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d (min >= %ss)\n'%(args.videoFilePath, len(sceneList), min_sec))
    return sceneList

def inference_video(args):
    # GPU: Face detection from container video stream using SAM3 video segmentation only.
    backend = str(getattr(args, 'detBackend', 'sam3')).lower()
    if backend != 'sam3':
        raise RuntimeError(f"inference_folder_sam3.py is SAM3-only; unsupported detBackend={backend!r}")

    # SAM3 text-grounded video segmentation using the official multi-GPU
    # video predictor. This mirrors the sam3-main usage:
    #   - start_session(resource_path=video_path)
    #   - add_prompt(text="face", frame_index=0)
    #   - propagate_in_video(...) to get per-frame masks.
    if not torch.cuda.is_available():
        raise RuntimeError('SAM3 face detector backend requires a CUDA GPU but none is available')

    # Resolve and insert local sam3-main into sys.path for imports
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sam3_root = os.path.abspath(os.path.join(this_dir, 'sam3-main'))
    if not os.path.isdir(sam3_root):
        raise RuntimeError(f'SAM3 root directory not found at {sam3_root}; please ensure sam3-main is present')
    if sam3_root not in sys.path:
        sys.path.insert(0, sam3_root)

    try:
        from sam3.model_builder import build_sam3_video_predictor  # type: ignore
    except Exception as e:  # pragma: no cover - environment/config errors
        raise RuntimeError('Failed to import SAM3; ensure sam3-main is installable in this environment') from e

    # Build multi-GPU video predictor; enable async PyAV loader to avoid
    # loading the entire video tensor into GPU memory at once.
    # OPTIMIZED: Enable compilation, skip frames (3x), and use bfloat16 for speedup
    # Note: image_size must stay at 1008 (ViT backbone has fixed RoPE position encodings)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpus_to_use = list(range(n_gpus)) if n_gpus > 0 else None
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        async_loading_frames=True,
        video_loader_type="pyav",
        compile=False,  # Disabled: compile overhead too high with multi-GPU
        image_size=1008,  # Must stay at 1008 (ViT RoPE is pre-computed for this size)
        recondition_every_nth_frame=64,  # Run detector every 64 frames (4x less than default 16) - tracker handles in-between frames
    )

    # Start a new video session.
    resp = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": args.videoFilePath,
        }
    )
    session_id = resp["session_id"]

    # Add a global text prompt "face" anchored at frame 0.
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": "face",
        }
    )

    # Collect per-frame outputs from propagate_in_video.
    # Use bfloat16 autocast for faster inference (2x speedup on Ampere+ GPUs)
    frame_outputs = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for res in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": None,
            }
        ):
            fidx = int(res.get("frame_index", -1))
            out = res.get("outputs", None)
            if out is None or fidx < 0:
                continue
            frame_outputs[fidx] = out

    # Close session and shut down workers to free GPU memory.
    predictor.handle_request({"type": "close_session", "session_id": session_id})
    try:
        predictor.shutdown()
    except Exception:
        pass

    if not frame_outputs:
        raise RuntimeError("SAM3 propagate_in_video produced no outputs")

    # Determine total number of frames from outputs
    max_fidx = max(frame_outputs.keys())
    num_frames = max_fidx + 1

    # Prepare per-frame detection/mask containers
    faces = [[] for _ in range(num_frames)]
    sam3_masks_all = [[] for _ in range(num_frames)]

    # Determine video resolution from first non-empty mask or fallback to cv2
    H = W = None
    for out in frame_outputs.values():
        masks = out.get("out_binary_masks", None)
        if masks is None:
            continue
        masks_np = np.asarray(masks)
        if masks_np.ndim == 3 and masks_np.shape[0] > 0:
            H, W = int(masks_np.shape[1]), int(masks_np.shape[2])
            break
    if H is None or W is None:
        cap0 = cv2.VideoCapture(args.videoFilePath)
        if not cap0.isOpened():
            raise RuntimeError(f"Failed to open video to derive resolution: {args.videoFilePath}")
        ret0, fr0 = cap0.read()
        cap0.release()
        if not ret0 or fr0 is None:
            raise RuntimeError("Failed to decode any frame to derive resolution")
        H, W = fr0.shape[0], fr0.shape[1]

    # Convert SAM3 outputs into whisperv faces.pckl + sam3_masks.pckl
    for fidx in sorted(frame_outputs.keys()):
        out = frame_outputs[fidx]
        obj_ids = out.get("out_obj_ids", None)
        probs = out.get("out_probs", None)
        boxes_xywh = out.get("out_boxes_xywh", None)
        masks = out.get("out_binary_masks", None)
        if (
            obj_ids is None
            or probs is None
            or boxes_xywh is None
            or masks is None
        ):
            continue

        boxes_xywh_np = np.asarray(boxes_xywh, dtype=np.float32)
        probs_np = np.asarray(probs, dtype=np.float32)
        masks_np = np.asarray(masks, dtype=bool)
        obj_ids_np = np.asarray(obj_ids, dtype=int)

        if boxes_xywh_np.ndim != 2 or boxes_xywh_np.shape[1] != 4:
            raise RuntimeError(f"SAM3 video: unexpected boxes_xywh shape {boxes_xywh_np.shape} at frame {fidx}")
        if masks_np.shape[0] != boxes_xywh_np.shape[0]:
            raise RuntimeError(f"SAM3 video: mask count mismatch with boxes at frame {fidx}")
        if probs_np.shape[0] != boxes_xywh_np.shape[0]:
            raise RuntimeError(f"SAM3 video: prob count mismatch with boxes at frame {fidx}")

        # Convert normalized XYWH to pixel XYXY
        xs = boxes_xywh_np[:, 0] * float(W)
        ys = boxes_xywh_np[:, 1] * float(H)
        ws = boxes_xywh_np[:, 2] * float(W)
        hs = boxes_xywh_np[:, 3] * float(H)
        x1 = xs
        y1 = ys
        x2 = xs + ws
        y2 = ys + hs
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

        frame_dets = []
        frame_masks = []
        for idx_det in range(boxes_xyxy.shape[0]):
            b = boxes_xyxy[idx_det]
            s = float(probs_np[idx_det])
            bb = [int(round(v)) for v in b.tolist()]
            x1_i, y1_i, x2_i, y2_i = bb
            # Clamp to image bounds and ensure non-degenerate area
            x1_i = max(0, min(x1_i, W - 1))
            y1_i = max(0, min(y1_i, H - 1))
            x2_i = max(0, min(x2_i, W))
            y2_i = max(0, min(y2_i, H))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            frame_dets.append(
                {
                    "frame": int(fidx),
                    "bbox": [x1_i, y1_i, x2_i, y2_i],
                    "conf": s,
                }
            )
            mask_arr = masks_np[idx_det]
            if mask_arr.shape != (H, W):
                raise RuntimeError(
                    f"SAM3 video: mask shape {mask_arr.shape} does not match video frame {(H, W)}"
                )
            if mask_arr.any():
                frame_masks.append(
                    {
                        "bbox": [x1_i, y1_i, x2_i, y2_i],
                        "score": s,
                        "mask": mask_arr,
                        "obj_id": int(obj_ids_np[idx_det]),
                    }
                )

        faces[int(fidx)] = frame_dets
        sam3_masks_all[int(fidx)] = frame_masks

    savePath = os.path.join(args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(faces, fil)
    masksPath = os.path.join(args.pyworkPath, "sam3_masks.pckl")
    with open(masksPath, "wb") as fil:
        pickle.dump(sam3_masks_all, fil)
    return faces

def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum    = numpy.array([ f['frame'] for f in track ])
            bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
    flist.sort()
    if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot crop face clips with correct timing.")
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), float(args.videoFps), (224,224))# Write video
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
        dets['y'].append((det[1]+det[3])/2) # crop center x
        dets['x'].append((det[0]+det[2])/2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args.cropScale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / float(args.videoFps)
    audioEnd    = (track['frame'][-1]+1) / float(args.videoFps)
    vOut.release()
    command = ("%s -y -i %s -c:a pcm_s16le -ac 1 -vn -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
              (_FFMPEG_BIN, args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("%s -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (_FFMPEG_BIN, cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets, "cropFile":cropFile}

def build_proc_track(track, crop_scale: float):
    """Replicate crop_video's medfilt smoothing to produce proc_track (s/x/y arrays).

    Returns a dict {'x': list[float], 'y': list[float], 's': list[float]} aligned with track['frame'].
    """
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    # Smooth detections identically to crop_video
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13).tolist()
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13).tolist()
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13).tolist()
    return dets

def _probe_frame_pts_with_pyav(video_path: str):
    """Return list of per-frame timestamps (seconds) using PyAV (PTS * time_base).

    Uses the first video stream. Raises RuntimeError if PyAV is unavailable or
    timestamps cannot be determined reliably.
    """
    try:
        import av  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyAV is required for PTS-based audio alignment but is not available."
        ) from e

    try:
        container = av.open(video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open video with PyAV: {video_path}") from e

    # Select the first video stream
    vstreams = [s for s in container.streams if s.type == 'video']
    if not vstreams:
        container.close()
        raise RuntimeError("No video stream found for PTS probing")
    vstream = vstreams[0]
    time_base = float(vstream.time_base) if vstream.time_base is not None else None

    pts_list = []
    try:
        for frame in container.decode(video=vstream.index):
            if frame.pts is not None and time_base is not None:
                ts = float(frame.pts) * time_base
            elif getattr(frame, 'time', None) is not None:
                ts = float(frame.time)
            else:
                ts = None
            pts_list.append(ts)
    finally:
        container.close()

    # Basic validation
    valid = [t for t in pts_list if isinstance(t, float) and t >= 0]
    if len(valid) < max(2, len(pts_list) // 10):
        raise RuntimeError("Insufficient valid PTS timestamps from PyAV for alignment")
    return pts_list

def _probe_frame_pts(video_path: str):
    """Fast per-frame timestamp (seconds) probe.

    Tries ffprobe (no decode) first for speed; falls back to PyAV decode if unavailable.
    """
    # Try ffprobe best_effort_timestamp_time
    try:
        cmd = [
            'ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time',
            '-of','csv=p=0',
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip().splitlines()
        pts = []
        for ln in out:
            ln = ln.strip()
            if not ln or ln == 'N/A':
                pts.append(None)
                continue
            try:
                pts.append(float(ln))
            except Exception:
                pts.append(None)
        # Basic validation: at least 10% valid or >=2
        valid = [t for t in pts if isinstance(t, float) and t >= 0]
        if len(valid) >= max(2, len(pts)//10):
            return pts
    except Exception:
        pass
    # Fallback to PyAV
    return _probe_frame_pts_with_pyav(video_path)

def _sanitize_frame_times(frame_times_sec: list) -> list:
    """Return a per-frame timestamp list of floats by filling missing entries.

    - Accepts list possibly containing None for frames with unavailable PTS.
    - Computes median positive delta from valid neighboring floats.
    - Fills leading/trailing/isolated None by stepping with median_dt to keep a
      non-decreasing sequence. Raises if insufficient valid data.
    """
    import numpy as _np
    if not isinstance(frame_times_sec, list) or len(frame_times_sec) == 0:
        raise RuntimeError("Empty frame_times_sec for PTS sanitization")
    # Collect valid floats
    vals = [float(x) for x in frame_times_sec if isinstance(x, (int, float))]
    if len(vals) < max(2, len(frame_times_sec)//10):
        raise RuntimeError("Insufficient valid PTS timestamps for sanitization")
    # Estimate median dt from valid consecutive diffs
    diffs = []
    last = None
    for x in frame_times_sec:
        if isinstance(x, (int, float)):
            xf = float(x)
            if last is not None and xf > last:
                diffs.append(xf - last)
            last = xf
    if not diffs:
        raise RuntimeError("Cannot derive positive PTS delta for sanitization")
    median_dt = float(_np.median(_np.asarray(diffs, dtype=float)))
    # Forward pass: fill from left to right
    out = [None] * len(frame_times_sec)
    last_val = None
    # First known index and value
    first_idx = None
    for i, v in enumerate(frame_times_sec):
        if isinstance(v, (int, float)):
            out[i] = float(v)
            last_val = out[i]
            if first_idx is None:
                first_idx = i
        else:
            if last_val is not None:
                last_val = last_val + median_dt
                out[i] = last_val
    # Backward pass: fill leading None using first known anchor
    if first_idx is not None:
        base = out[first_idx]
        for i in range(first_idx-1, -1, -1):
            base = base - median_dt
            out[i] = max(0.0, base)
    # Final check and monotonic clamp
    prev = 0.0
    for i in range(len(out)):
        if not isinstance(out[i], float):
            # In rare case no anchor found (shouldn't happen due to earlier checks)
            raise RuntimeError("Failed to sanitize frame timestamps (residual None)")
        if out[i] < prev:
            out[i] = prev
        prev = out[i]
    return out

def _resample_tracks_to_scores(annotated_tracks, scores):
    """Return a new list of tracks resampled so that len(frames)==len(scores[i]).

    - Frames become 0..T-1 (25fps grid), where T=len(scores[i]).
    - proc_track fields x,y,s are linearly interpolated to T.
    - identity/cropFile preserved; original bbox per-frame arrays are not resampled (unused downstream).
    """
    out = []
    for i, tr in enumerate(annotated_tracks):
        T = len(scores[i]) if i < len(scores) else 0
        tr2 = dict(tr)
        if T <= 0:
            # keep as-is when no scores (shouldn't happen)
            out.append(tr2)
            continue
        # Build new frame index 0..T-1
        new_frames = list(range(T))
        # Resample proc_track fields if present
        proc = tr.get('proc_track', None)
        if isinstance(proc, dict):
            def _interp(arr):
                try:
                    import numpy as _np
                    arr_np = _np.asarray(arr, dtype=float).reshape(-1)
                    n0 = arr_np.shape[0]
                    if n0 <= 1:
                        return [float(arr_np[0]) for _ in range(T)]
                    x0 = _np.linspace(0.0, 1.0, num=n0)
                    x1 = _np.linspace(0.0, 1.0, num=T)
                    v1 = _np.interp(x1, x0, arr_np)
                    return [float(v) for v in v1]
                except Exception:
                    # Fallback to nearest repeat
                    return [float(arr[0]) for _ in range(T)] if arr else [0.0]*T
            new_proc = {}
            for k in ('x','y','s'):
                if k in proc:
                    new_proc[k] = _interp(proc[k])
            tr2['proc_track'] = new_proc
        # Replace track.frames with new frames; keep bbox untouched
        track_obj = tr.get('track', {})
        tr2['track'] = dict(track_obj)
        tr2['track']['frame'] = new_frames
        out.append(tr2)
    return out

@torch.no_grad()
def evaluate_network_in_memory(tracks, args, frame_start: int = None, frame_end: int = None):
    """Compute ASD scores entirely in-memory without writing crop clips.

    - Builds 25fps ROI sequences per track via GPU roi_align using per-frame PTS.
    - Extracts per-track audio segments from full audio via PTS -ss/-to equivalents.
    - Runs TalkNet with batched windows per track (same durations/averaging as file-based path).
    Returns: list of scores per track (same structure as evaluate_network).
    """
    # 1) Prepare 25fps time grid and per-track start/end seconds via PTS timeline
    frame_times_sec = _probe_frame_pts(args.videoFilePath)
    if not isinstance(frame_times_sec, list) or len(frame_times_sec) == 0:
        raise RuntimeError("Invalid or empty PTS timestamps for ASD alignment")
    frame_times_sec = _sanitize_frame_times(frame_times_sec)
    # Median frame duration for tail fill
    _fts = np.asarray(frame_times_sec, dtype=float)
    diffs = np.diff(_fts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise RuntimeError("Non-positive PTS deltas; cannot derive frame duration")
    median_dt = float(np.median(diffs))

    # Build mapping: frame index -> list of (track_idx, local_idx)
    frame_to_entries = defaultdict(list)
    proc_tracks = []
    track_ranges = []
    track_start_sec = {}
    track_end_sec = {}
    for tidx, tr in enumerate(tracks):
        frames = tr['frame'] if isinstance(tr, dict) else tr['track']['frame']
        bboxes = tr['bbox'] if isinstance(tr, dict) else tr['track']['bbox']
        track_norm = {'frame': frames, 'bbox': bboxes}
        dets = build_proc_track(track_norm, args.cropScale)
        proc_tracks.append(dets)
        frames_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
        if not frames_list:
            track_ranges.append((None, None))
            continue
        s_f = int(frames_list[0]); e_f = int(frames_list[-1])
        track_ranges.append((s_f, e_f))
        if s_f < 0 or e_f < s_f:
            raise RuntimeError(f"Invalid track frame indices: {s_f}-{e_f}")
        # Use PTS timeline derived from container for robust alignment
        if s_f >= len(frame_times_sec) or e_f >= len(frame_times_sec):
            raise RuntimeError("Track frame indices exceed timestamp map length")
        t_s = float(frame_times_sec[s_f])
        t_e = float(frame_times_sec[e_f])
        track_start_sec[tidx] = t_s
        track_end_sec[tidx] = t_e
        for lidx, f in enumerate(frames_list):
            frame_to_entries[int(f)].append((tidx, lidx))

    # 2) Stream decode frames; GPU roi_align batch-crop; emit 25fps ROI to per-track buffers
    cap = cv2.VideoCapture(args.videoFilePath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for ASD in-memory: {args.videoFilePath}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA GPU required for in-memory ASD acceleration')
    OUT_FPS = 25.0
    next_time = {}
    last_face = {}
    faces_mem = {i: [] for i in range(len(tracks))}

    # Initialize frame index and optionally seek
    if frame_start is not None and frame_start >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
        fidx = int(frame_start)
    else:
        fidx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if frame_end is not None and fidx > int(frame_end):
            break
        entries = frame_to_entries.get(fidx, [])
        if entries:
            cs = args.cropScale
            # Compute maximum pad required across entries
            bsi_list = []
            for (tidx, lidx) in entries:
                bs = float(proc_tracks[tidx]['s'][lidx])
                bsi_list.append(int(bs * (1 + 2 * cs)))
            pad_used = int(max(bsi_list)) if bsi_list else 0

            # Prepare image tensor
            img_t = torch.from_numpy(image).to(device=device, dtype=torch.float32)  # H,W,C (BGR)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0  # 1,C,H,W
            if pad_used > 0:
                img_t = F.pad(img_t, (pad_used, pad_used, pad_used, pad_used), mode='constant', value=110.0/255.0)
            _, C, Hp, Wp = img_t.shape

            # Build ROIs for torchvision.ops.roi_align: [batch_idx, x1, y1, x2, y2]
            try:
                from torchvision.ops import roi_align  # type: ignore
            except Exception as e:
                raise RuntimeError('torchvision.ops.roi_align is required for GPU cropping')
            rois = []
            tids = []
            lidxs = []
            for (tidx, lidx) in entries:
                bs = float(proc_tracks[tidx]['s'][lidx])
                my = float(proc_tracks[tidx]['y'][lidx]) + pad_used
                mx = float(proc_tracks[tidx]['x'][lidx]) + pad_used
                y1 = my - bs
                y2 = my + bs * (1 + 2 * cs)
                x1 = mx - bs * (1 + cs)
                x2 = mx + bs * (1 + cs)
                # clamp to image bounds
                x1 = max(0.0, min(x1, Wp - 1.0)); x2 = max(0.0, min(x2, Wp - 1.0))
                y1 = max(0.0, min(y1, Hp - 1.0)); y2 = max(0.0, min(y2, Hp - 1.0))
                if x2 <= x1 or y2 <= y1:
                    continue
                rois.append([0.0, x1, y1, x2, y2])
                tids.append(tidx)
                lidxs.append(lidx)
            if rois:
                rois_t = torch.tensor(rois, device=device, dtype=torch.float32)
                crops = roi_align(img_t, rois_t, output_size=(224,224), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                crops = (crops.clamp(0.0,1.0) * 255.0).to(torch.uint8).permute(0,2,3,1).contiguous().cpu().numpy()  # B,224,224,3
                t_src = float(frame_times_sec[fidx])
                for j in range(crops.shape[0]):
                    tidx = tids[j]
                    # cache latest face
                    last_face[tidx] = crops[j]
                    # initialize next_time if first time
                    if tidx not in next_time:
                        next_time[tidx] = track_start_sec.get(tidx, t_src)
                    # emit frames up to current time
                    end_time_allowed = track_end_sec.get(tidx, t_src)
                    while next_time[tidx] <= t_src + 1e-6 and next_time[tidx] <= end_time_allowed + median_dt + 1e-6:
                        faces_mem[tidx].append(last_face[tidx])
                        next_time[tidx] += (1.0 / OUT_FPS)
        fidx += 1
    cap.release()

    # Tail fill
    for tidx in range(len(tracks)):
        if tidx in next_time and tidx in track_end_sec and tidx in last_face:
            while next_time[tidx] <= track_end_sec[tidx] + median_dt + 1e-6:
                faces_mem[tidx].append(last_face[tidx])
                next_time[tidx] += (1.0 / OUT_FPS)

    # 3) TalkNet ASD with cross-track batching per duration
    s = talkNet(); s.loadParameters(args.pretrainModel); s.eval()
    # Multi-GPU: simple DataParallel across all visible devices (can be disabled via env in worker)
    use_dp = (torch.cuda.device_count() > 1) and (os.environ.get('ASD_FORCE_DP', '1') != '0')
    dp_model = torch.nn.DataParallel(s) if use_dp else None
    durationU = [1,2,3,4,5,6]
    # load full audio once
    _, full_audio = wavfile.read(os.path.join(args.pyaviPath, 'audio.wav'))
    sr = 16000

    # Precompute per-track audioFeature and v_arr once (reuse across durations)
    featsA = []  # list of np.ndarray [Ta,13]
    featsV = []  # list of np.ndarray [Tv,112,112]
    lens = []    # list of common length in seconds
    for tidx in range(len(tracks)):
        t_s = track_start_sec.get(tidx, 0.0)
        t_e = track_end_sec.get(tidx, t_s) + median_dt
        a0 = int(round(t_s * sr)); a1 = max(a0+1, int(round(t_e * sr)))
        a_seg = full_audio[a0:a1]
        v_seq = faces_mem[tidx]
        if not v_seq or a_seg.size == 0:
            featsA.append(None); featsV.append(None); lens.append(0.0); continue
        # visual: grayscale + center-crop 112x112
        v_arr = np.empty((len(v_seq), 112, 112), dtype=np.uint8)
        for i_f, f in enumerate(v_seq):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            v_arr[i_f] = g[56:168, 56:168]
        # audio mfcc 100Hz
        a_feat = python_speech_features.mfcc(a_seg, sr, numcep=13, winlen=0.025, winstep=0.010)
        length = min((a_feat.shape[0] - a_feat.shape[0] % 4) / 100.0, v_arr.shape[0] / 25.0)
        a_feat = a_feat[:int(round(length*100)), :]
        v_arr = v_arr[:int(round(length*25)), :, :]
        featsA.append(a_feat); featsV.append(v_arr); lens.append(length)

    # For each duration, cross-track batch windows
    perDurScores = {d: [None]*len(tracks) for d in durationU}
    Bglob = max(1, int(getattr(args,'asdBatch',64)))
    for duration in durationU:
        winA = int(duration*100); winV=int(duration*25)
        # per-track pointers
        n_full = []
        for i in range(len(tracks)):
            a_feat = featsA[i]; v_arr = featsV[i]
            if a_feat is None or v_arr is None:
                n_full.append(0); continue
            n_full.append(int(min(a_feat.shape[0] // winA, v_arr.shape[0] // winV)))
        # storage
        scores_by_track = [[] for _ in range(len(tracks))]
        pos = [0]*len(tracks)
        # iterate until all windows consumed
        remaining = sum(n_full)
        while remaining > 0:
            batchA = []
            batchV = []
            owners = []
            for i in range(len(tracks)):
                if pos[i] < n_full[i]:
                    # take as many as fit into batch
                    take = min(n_full[i]-pos[i], max(1, Bglob - len(batchA)))
                    a_feat = featsA[i]; v_arr = featsV[i]
                    for k in range(take):
                        idx = pos[i]+k
                        batchA.append(a_feat[idx*winA:(idx+1)*winA, :])
                        batchV.append(v_arr[idx*winV:(idx+1)*winV, :, :])
                        owners.append(i)
                    pos[i] += take
                    if len(batchA) >= Bglob:
                        break
            # forward if batch non-empty
            if batchA:
                if use_dp:
                    inputA = torch.from_numpy(np.stack(batchA, axis=0).astype(np.float32))
                    inputV = torch.from_numpy(np.stack(batchV, axis=0).astype(np.float32))
                    out = dp_model(inputA, inputV)
                else:
                    inputA = torch.FloatTensor(np.stack(batchA,axis=0)).cuda()
                    inputV = torch.FloatTensor(np.stack(batchV,axis=0)).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                scoreBatch = s.lossAV.forward(out, labels=None)
                scoreBatch = np.asarray(scoreBatch)
                # Split back per-window using uniform per-window length, then dispatch to owners
                nW = len(owners)
                if nW <= 0:
                    break
                per_len = int(scoreBatch.shape[0] // nW) if nW > 0 else 0
                # Safety: if per_len == 0, fall back to one scalar per window
                if per_len <= 0:
                    for j, iowner in enumerate(owners):
                        val = float(scoreBatch[j]) if j < scoreBatch.shape[0] else 0.0
                        scores_by_track[iowner].append(val)
                else:
                    for j, iowner in enumerate(owners):
                        start = j * per_len
                        end = start + per_len
                        vals = scoreBatch[start:end].tolist()
                        scores_by_track[iowner].extend(vals)
                remaining -= len(batchA)
            else:
                break
        # Handle tail windows (variable length) singly to mirror file-based path
        for i in range(len(tracks)):
            a_feat = featsA[i]; v_arr = featsV[i]
            if a_feat is None or v_arr is None:
                continue
            usedA = n_full[i] * winA
            usedV = n_full[i] * winV
            a_tail = a_feat[usedA: usedA + winA, :]
            v_tail = v_arr[usedV: usedV + winV, :, :]
            if a_tail.shape[0] > 0 and v_tail.shape[0] > 0:
                if use_dp:
                    inputA = torch.from_numpy(a_tail.astype(np.float32)).unsqueeze(0)
                    inputV = torch.from_numpy(v_tail.astype(np.float32)).unsqueeze(0)
                    out = dp_model(inputA, inputV)
                else:
                    inputA = torch.FloatTensor(a_tail).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(v_tail).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                score_tail = s.lossAV.forward(out, labels=None)
                try:
                    vals = np.asarray(score_tail).tolist()
                    scores_by_track[i].extend(vals)
                except Exception:
                    scores_by_track[i].append(float(score_tail))
        perDurScores[duration] = scores_by_track

    # Average across durations with original weighting {1,1,1,2,2,2,3,3,4,5,6}
    weights = {1:3, 2:3, 3:2, 4:1, 5:1, 6:1}
    allScores = []
    for i in range(len(tracks)):
        # gather scores per duration for this track
        seqs = [perDurScores[d][i] for d in durationU]
        # consider only non-empty sequences when computing the common length
        nonempty = [np.array(x, dtype=float) for x in seqs if isinstance(x, (list, tuple)) and len(x) > 0]
        if not nonempty:
            allScores.append([])
            continue
        Lmin = min(arr.shape[0] for arr in nonempty)
        if Lmin <= 0:
            allScores.append([])
            continue
        # build weighted stack using only durations that produced predictions
        stack_list = []
        for d, x in zip(durationU, seqs):
            if not (isinstance(x, (list, tuple)) and len(x) > 0):
                continue
            w = int(weights.get(d, 1))
            if w <= 0:
                continue
            arrd = np.array(x[:Lmin], dtype=float)
            for _ in range(w):
                stack_list.append(arrd)
        if not stack_list:
            allScores.append([])
            continue
        arr = np.stack(stack_list, axis=0)
        allScores.append(np.round(arr.mean(axis=0), 1).astype(float))
    return allScores

def stream_crop_tracks(args, tracks):
    """Stream video once and crop all face tracks without using pyframes on disk.

    For each track, writes a temporary '<cropFile>t.avi', then muxes cropped audio
    from args.audioFilePath using ffmpeg -ss/-to based on args.videoFps.

    Returns list of dicts [{'track': track, 'proc_track': dets, 'cropFile': cropFile}, ...]
    with the exact same schema as previous crop_video-based outputs.
    """
    if not hasattr(args, 'videoFps') or args.videoFps is None or float(args.videoFps) <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot stream-crop with correct timing.")

    # Precompute smoothed proc_track and per-frame index mapping
    proc_tracks = []
    frame_to_entries = defaultdict(list)  # fidx -> list of (track_idx, local_idx)
    for tidx, tr in enumerate(tracks):
        frames = tr['frame'] if isinstance(tr, dict) else tr['track']['frame']
        bboxes = tr['bbox'] if isinstance(tr, dict) else tr['track']['bbox']
        track_norm = {'frame': frames, 'bbox': bboxes}
        dets = build_proc_track(track_norm, args.cropScale)
        proc_tracks.append(dets)
        frames_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
        for local_idx, f in enumerate(frames_list):
            frame_to_entries[int(f)].append((tidx, local_idx))

    # Prepare per-track writers (opened lazily when first frame is reached)
    writers = {}
    crop_files = {}
    # Precompute (start_frame, end_frame) per track
    track_ranges = []
    # Also compute PTS-based start/end seconds per track for 25fps resampling grid
    track_start_sec = {}
    track_end_sec = {}
    # Probe per-frame timestamps (prefer ffprobe; fallback to PyAV) before mapping frames to seconds
    frame_times_sec = _probe_frame_pts(args.videoFilePath)
    frame_times_sec = _sanitize_frame_times(frame_times_sec)
    for tr in tracks:
        frames = tr['frame'] if isinstance(tr, dict) else tr['track']['frame']
        frames_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
        if not frames_list:
            track_ranges.append((None, None))
            continue
        else:
            s_f = int(frames_list[0]); e_f = int(frames_list[-1])
            track_ranges.append((s_f, e_f))
            if s_f < 0 or e_f >= len(frame_times_sec):
                raise RuntimeError(f"Track frames out of bounds for timestamp map: {s_f}-{e_f} vs {len(frame_times_sec)}")
            t_s = float(frame_times_sec[s_f]); t_e = float(frame_times_sec[e_f])
            track_start_sec[len(track_ranges)-1] = t_s
            track_end_sec[len(track_ranges)-1] = t_e

    # Compute median frame interval from PTS for end padding; fallback to 1/25 if diffs missing
    _pts_diffs = []
    for i in range(1, len(frame_times_sec)):
        t0 = frame_times_sec[i-1]
        t1 = frame_times_sec[i]
        if isinstance(t0, float) and isinstance(t1, float) and t1 > t0:
            _pts_diffs.append(t1 - t0)
    median_dt = float(np.median(np.array(_pts_diffs, dtype=float))) if _pts_diffs else (1.0 / 25.0)

    cap = cv2.VideoCapture(args.videoFilePath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for cropping: {args.videoFilePath}")

    # Determine frame size
    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to decode any frame from video for cropping")
    fh, fw = first.shape[0], first.shape[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Iterate frames and write crops with 25fps resampling on PTS grid
    fidx = 0
    OUT_FPS = 25.0
    # Next output write time per track (seconds)
    next_time = {}
    # Cache last cropped face per track for duplication when needed
    last_face = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA GPU is required for accelerated cropping but is not available')

    while True:
        ret, image = cap.read()
        if not ret:
            break

        entries = frame_to_entries.get(fidx, [])
        if entries:
            # Prepare per-entry params
            cs = args.cropScale
            # Compute bs and required padding per entry
            bs_list = []
            bsi_list = []
            for (tidx, lidx) in entries:
                dets = proc_tracks[tidx]
                bs = float(dets['s'][lidx])
                bs_list.append(bs)
                bsi_list.append(int(bs * (1 + 2 * cs)))
            pad_used = int(max(bsi_list)) if bsi_list else 0

            # Convert frame to GPU tensor and pad once
            img_t = torch.from_numpy(image).to(device=device, dtype=torch.float32)  # H,W,C (BGR)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0  # 1,C,H,W
            if pad_used > 0:
                img_t = F.pad(img_t, (pad_used, pad_used, pad_used, pad_used), mode='constant', value=110.0/255.0)
            _, C, Hp, Wp = img_t.shape

            # Build batch grids for all entries
            grids = []
            tids = []
            lidxs = []
            for idx, (tidx, lidx) in enumerate(entries):
                dets = proc_tracks[tidx]
                bs = float(dets['s'][lidx])
                my = float(dets['y'][lidx]) + pad_used
                mx = float(dets['x'][lidx]) + pad_used
                y1 = my - bs
                y2 = my + bs * (1 + 2 * cs)
                x1 = mx - bs * (1 + cs)
                x2 = mx + bs * (1 + cs)
                # Construct sampling grid for this ROI
                xs = torch.linspace(x1, x2, 224, device=device)
                ys = torch.linspace(y1, y2, 224, device=device)
                grid_x = xs.view(1, 1, 224).expand(1, 224, 224)
                grid_y = ys.view(1, 224, 1).expand(1, 224, 224)
                # Normalize to [-1,1] with align_corners=True convention
                gx = (2.0 * (grid_x / max(Wp - 1, 1.0))) - 1.0
                gy = (2.0 * (grid_y / max(Hp - 1, 1.0))) - 1.0
                grid = torch.stack([gx, gy], dim=-1)  # 1,224,224,2
                grids.append(grid)
                tids.append(tidx)
                lidxs.append(lidx)

            grid_b = torch.cat(grids, dim=0)  # B,224,224,2
            img_b = img_t.expand(grid_b.shape[0], -1, -1, -1).contiguous()  # B,C,H,W
            crops_b = F.grid_sample(img_b, grid_b, mode='bilinear', align_corners=True)
            crops_b = (crops_b.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # B,C,224,224
            crops_b = crops_b.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # B,224,224,C (BGR preserved)

            # Update writers and write frames according to 25fps grid
            t_src = float(frame_times_sec[fidx])
            for j in range(len(tids)):
                tidx = tids[j]
                lidx = lidxs[j]
                # Open writer lazily
                if tidx not in writers:
                    cropFile = os.path.join(args.pycropPath, '%05d' % tidx)
                    crop_files[tidx] = cropFile
                    writers[tidx] = cv2.VideoWriter(
                        cropFile + 't.avi',
                        cv2.VideoWriter_fourcc(*'XVID'),
                        OUT_FPS,
                        (224, 224),
                    )
                    next_time[tidx] = track_start_sec.get(tidx, t_src)
                # Cache latest face for this track
                last_face[tidx] = crops_b[j]
                # Emit duplicated frames up to current source time / end time
                end_time_allowed = track_end_sec.get(tidx, t_src)
                while tidx in next_time and next_time[tidx] <= t_src + 1e-6 and next_time[tidx] <= end_time_allowed + median_dt + 1e-6:
                    writers[tidx].write(last_face[tidx])
                    next_time[tidx] += (1.0 / OUT_FPS)

        fidx += 1

    cap.release()

    # Close writers before muxing; finalize tail writes up to end time
    for tidx in list(writers.keys()):
        # Fill tail if needed using last available face
        if tidx in next_time and tidx in track_end_sec and tidx in last_face:
            while next_time[tidx] <= track_end_sec[tidx] + median_dt + 1e-6:
                writers[tidx].write(last_face[tidx])
                next_time[tidx] += (1.0 / OUT_FPS)
    for tidx in list(writers.keys()):
        try:
            writers[tidx].release()
        except Exception:
            pass

    # median_dt computed earlier alongside frame_times_sec

    # Prepare parallel mux tasks
    tasks = []
    for tidx, tr in enumerate(tracks):
        cropFile = crop_files.get(tidx, os.path.join(args.pycropPath, '%05d' % tidx))
        start_f, end_f = track_ranges[tidx]
        if start_f is None or end_f is None:
            continue
        if start_f < 0 or end_f >= len(frame_times_sec):
            raise RuntimeError(f"Track frame indices out of bounds for timestamp map: {start_f}-{end_f} vs {len(frame_times_sec)}")
        t_start = frame_times_sec[start_f]
        t_end = frame_times_sec[end_f]
        if not (isinstance(t_start, float) and isinstance(t_end, float)):
            raise RuntimeError("Encountered invalid frame timestamps; cannot PTS-align audio")
        # End time: last frame timestamp plus median frame duration (approximate)
        audioStart = float(t_start)
        audioEnd = float(t_end + median_dt)
        tasks.append((tidx, cropFile, audioStart, audioEnd, args.audioFilePath, int(args.nDataLoaderThread)))

    # Run tasks in parallel using torch.multiprocessing
    import torch.multiprocessing as mp
    num_workers = max(1, int(getattr(args, 'cropWorkers', 8)))
    if len(tasks) <= 1 or num_workers == 1:
        results = list(map(_mux_worker, tasks))
    else:
        # Use spawn context to be safe
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_mux_worker, tasks)

    # Verify and assemble outputs
    failures = [(tidx, msg) for (tidx, ok, msg) in results if not ok]
    if failures:
        raise RuntimeError(f"Mux failures: {failures[:3]} (total {len(failures)})")

    vidTracks = []
    for tidx, _cropFile, *_ in tasks:
        cropFile = crop_files.get(tidx, os.path.join(args.pycropPath, '%05d' % tidx))
        tr = tracks[tidx]
        vidTracks.append({'track': {'frame': tr['frame'], 'bbox': tr['bbox']}, 'proc_track': proc_tracks[tidx], 'cropFile': cropFile})
    return vidTracks

def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)

def _mux_worker(t):
    """Top-level worker to cut audio and mux with cropped video.

    Args: t = (tidx, cropFile, aStart, aEnd, audioPath, nThreads)
    Returns: (tidx, ok: bool, msg: str)
    """
    import subprocess, os
    try:
        tidx, cropFile, aStart, aEnd, audioPath, nThreads = t
        audioTmp = cropFile + '.wav'
        # Limit threads per ffmpeg when running in parallel to avoid oversubscription
        ff_threads = max(1, min(2, int(nThreads)))
        cmd1 = (
            "%s -y -i %s -c:a pcm_s16le -ac 1 -vn -ar 16000 -threads %d -ss %.6f -to %.6f %s -loglevel panic"
            % (_FFMPEG_BIN, audioPath, ff_threads, float(aStart), float(aEnd), audioTmp)
        )
        r1 = subprocess.call(cmd1, shell=True, stdout=None)
        if r1 != 0:
            return (tidx, False, f"audio cut failed rc={r1}")
        cmd2 = (
            "%s -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
            % (_FFMPEG_BIN, cropFile, audioTmp, ff_threads, cropFile)
        )
        r2 = subprocess.call(cmd2, shell=True, stdout=None)
        if r2 != 0:
            return (tidx, False, f"mux failed rc={r2}")
        try:
            os.remove(cropFile + 't.avi')
        except Exception:
            pass
        return (tidx, True, '')
    except Exception as e:
        return (-1, False, f"exception: {e}")

def _asd_scene_worker(args_pack):
    """Top-level worker for in-memory ASD on a scene range.

    Args: (tasks_chunk, minimal_dict, gpu_id)
      tasks_chunk: list of (idxs, tr_sub, s_f, e_f)
    Returns: list of (idxs, scores_sublist)
    """
    from types import SimpleNamespace
    tasks_chunk, minimal, gpu_id = args_pack
    # Bind this worker to a single device and disable DP inside
    try:
        if torch.cuda.is_available() and gpu_id is not None and int(gpu_id) >= 0:
            torch.cuda.set_device(int(gpu_id))
    except Exception:
        pass
    os.environ['ASD_FORCE_DP'] = '0'
    a = SimpleNamespace(**minimal)
    out = []
    for (idxs, tr_sub, s_f, e_f) in tasks_chunk:
        out.append((idxs, evaluate_network_in_memory(tr_sub, a, frame_start=s_f, frame_end=e_f)))
    return out

def _diar_chunk_worker(args_pack):
    """Worker: run WhisperX diarization on an audio chunk and shift times.

    Args: (s0, s1, audio_path, min_speakers, max_speakers, hf_token, gpu_id)
    Returns: list[dict{start,end,speaker}]
    """
    s0, s1, audio_path, min_k, max_k, tok, gpu_id = args_pack
    try:
        if torch.cuda.is_available() and int(gpu_id) >= 0:
            torch.cuda.set_device(int(gpu_id))
    except Exception:
        pass
    pipe = whisperx.DiarizationPipeline(use_auth_token=tok, device='cuda' if torch.cuda.is_available() else 'cpu')
    # Load audio and slice locally to avoid large IPC payloads
    audio_full = whisperx.load_audio(audio_path)
    sr_local = 16000.0
    a0 = int(round(float(s0) * sr_local)); a1 = int(round(float(s1) * sr_local))
    a0 = max(0, a0); a1 = max(a0 + 1, a1)
    wav_chunk = audio_full[a0:a1]
    if min_k is not None or max_k is not None:
        segs = pipe(wav_chunk, min_speakers=min_k, max_speakers=max_k)
    else:
        segs = pipe(wav_chunk)
    df = _to_diarize_df(segs)
    rows = []
    for _, r in df.iterrows():
        st = float(r['start']) + float(s0)
        en = float(r['end']) + float(s0)
        if en > st:
            rows.append({'start': st, 'end': en, 'speaker': str(r.get('speaker', 'SPEAKER_XX'))})
    return rows

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet (batched windows)
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()
    # Multi-GPU: simple DataParallel across all visible devices (can be disabled via env in worker)
    use_dp = (torch.cuda.device_count() > 1) and (os.environ.get('ASD_FORCE_DP', '1') != '0')
    dp_model = torch.nn.DataParallel(s) if use_dp else None
    allScores = []
    # durationSet = {1,2,4,6}
    durationSet = {1,1,1,2,2,2,3,3,4,5,6}
    B = max(1, int(getattr(args, 'asdBatch', 64)))

    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0]
        # Audio features @100Hz
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        # Video features (center crop to 112x112)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)

        # Keep TalkNet's expected 4:1 audio:video ratio
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100.0, videoFeature.shape[0] / 25.0)
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]

        allScore = []
        for duration in durationSet:
            winA = int(duration * 100)
            winV = int(duration * 25)
            total_len_a = audioFeature.shape[0]
            total_len_v = videoFeature.shape[0]
            # Full windows count
            n_full = int(min(total_len_a // winA, total_len_v // winV))
            has_tail = (total_len_a % winA != 0) or (total_len_v % winV != 0)

            scores = []
            with torch.no_grad():
                # Process full windows in batches
                i = 0
                while i < n_full:
                    j = min(n_full, i + B)
                    batchA = [audioFeature[k * winA:(k + 1) * winA, :] for k in range(i, j)]
                    batchV = [videoFeature[k * winV:(k + 1) * winV, :, :] for k in range(i, j)]
                    if use_dp:
                        inputA = torch.from_numpy(numpy.stack(batchA, axis=0).astype(numpy.float32))
                        inputV = torch.from_numpy(numpy.stack(batchV, axis=0).astype(numpy.float32))
                        out = dp_model(inputA, inputV)
                    else:
                        inputA = torch.FloatTensor(numpy.stack(batchA, axis=0)).cuda()
                        inputV = torch.FloatTensor(numpy.stack(batchV, axis=0)).cuda()
                        embedA = s.model.forward_audio_frontend(inputA)
                        embedV = s.model.forward_visual_frontend(inputV)
                        embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                        out = s.model.forward_audio_visual_backend(embedA, embedV)
                    scoreBatch = s.lossAV.forward(out, labels=None)  # (B*T,)
                    scoreBatch = numpy.asarray(scoreBatch)
                    # Split back per window (equal length since full windows)
                    if (j - i) > 0:
                        # Estimate per-window length in predictions
                        per_len = int(scoreBatch.shape[0] // (j - i))
                        for b in range(j - i):
                            start = b * per_len
                            end = start + per_len
                            scores.extend(scoreBatch[start:end].tolist())
                    i = j

                # Tail window if exists (variable length) — process singly to keep logic identical
                if has_tail:
                    k = n_full
                    a_tail = audioFeature[k * winA: (k + 1) * winA, :]
                    v_tail = videoFeature[k * winV: (k + 1) * winV, :, :]
                    if a_tail.shape[0] > 0 and v_tail.shape[0] > 0:
                        if use_dp:
                            inputA = torch.from_numpy(a_tail.astype(numpy.float32)).unsqueeze(0)
                            inputV = torch.from_numpy(v_tail.astype(numpy.float32)).unsqueeze(0)
                            out = dp_model(inputA, inputV)
                        else:
                            inputA = torch.FloatTensor(a_tail).unsqueeze(0).cuda()
                            inputV = torch.FloatTensor(v_tail).unsqueeze(0).cuda()
                            embedA = s.model.forward_audio_frontend(inputA)
                            embedV = s.model.forward_visual_frontend(inputV)
                            embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                            out = s.model.forward_audio_visual_backend(embedA, embedV)
                        score_tail = s.lossAV.forward(out, labels=None)
                        # Append per-step tail predictions (1 window)
                        try:
                            scores.extend(np.asarray(score_tail).tolist())
                        except Exception:
                            # If score_tail is scalar-like
                            scores.append(float(score_tail))

            allScore.append(scores)

        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores

def visualization(tracks, scores, args):
    # CPU: visualize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        identity = track.get('identity', 'None')
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = numpy.mean(s)
            faces[frame].append({'track':tidx,
                                 'identity': identity,
                                 'score':float(s),
                                 's':track['proc_track']['s'][fidx],
                                 'x':track['proc_track']['x'][fidx],
                                 'y':track['proc_track']['y'][fidx]})
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot render visualization with correct timing.")
    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, 'video_only.avi'),
        cv2.VideoWriter_fourcc(*'XVID'),
        float(args.videoFps),
        (fw, fh)
    )
    # Build a stable color map per identity (BGR)
    def _id_color_map(tracks_list):
        ids = []
        for tr in tracks_list:
            ident = tr.get('identity', None)
            if ident is None or ident == 'None':
                continue
            ids.append(ident)
        uniq = sorted(set(ids))
        colors = {}
        import colorsys, hashlib
        base_h = {}
        for ident in uniq:
            hval = int(hashlib.md5(ident.encode('utf-8')).hexdigest()[:8], 16)
            base_h[ident] = float((hval % 360) / 360.0)
        used = []
        min_sep = 0.12
        for ident in uniq:
            h = base_h[ident]
            for _ in range(6):
                ok = True
                for hu in used:
                    d = abs(h - hu)
                    d = min(d, 1.0 - d)
                    if d < min_sep:
                        h = (h + 0.5) % 1.0
                        ok = False
                        break
                if ok:
                    break
            used.append(h)
            s = 0.65
            v = 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors[ident] = (int(b * 255), int(g * 255), int(r * 255))
        return colors
    ID_COLORS = _id_color_map(tracks)

    # Prepare diarization segments per identity for speech bubbles (right/left balloons)
    segs_by_ident = defaultdict(list)
    if isinstance(diarization_results, list):
        for seg in diarization_results:
            ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            s = float(seg.get('start', seg.get('start_time', 0.0)))
            e = float(seg.get('end', seg.get('end_time', s)))
            if e <= s:
                continue
            txt = str(seg.get('text', ''))
            segs_by_ident[ident].append((s, e, txt))
        for k in segs_by_ident:
            segs_by_ident[k].sort(key=lambda x: (x[0], x[1]))

    def _bubble_text_for(ident: str, t_sec: float) -> str:
        lst = segs_by_ident.get(ident, [])
        for (s, e, txt) in lst:
            if s <= t_sec <= e:
                return txt
        return ''

    def _wrap_lines(txt: str, max_cn: int = 16, max_lat: int = 24, max_lines: int = 3):
        wrapped = _wrap_text_for_ass(txt, max_cn, max_lat)
        lines = [ln for ln in wrapped.split('\\N') if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines and not lines[-1].endswith('…'):
                lines[-1] = lines[-1] + '…'
        return lines

    def _draw_rounded_rect(img, x, y, w, h, radius, fill_color, border_color, border_th=2, alpha=0.9):
        x = int(x); y = int(y); w = int(w); h = int(h); radius = int(max(2, radius))
        overlay = img.copy()
        cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), fill_color, thickness=-1)
        cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.rectangle(img, (x+radius, y), (x+w-radius, y+h), border_color, thickness=border_th)
        cv2.rectangle(img, (x, y+radius), (x+w, y+h-radius), border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+h-radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+h-radius), radius, border_color, thickness=border_th)

    def _draw_tail(img, base_x, base_y, to_left: bool, fill_color, border_color, tail_len=14, tail_half=8, alpha=0.9, border_th=2):
        base_x = int(base_x); base_y = int(base_y)
        if to_left:
            pts = np.array([[base_x, base_y], [base_x+tail_len, base_y-tail_half], [base_x+tail_len, base_y+tail_half]], dtype=np.int32)
        else:
            pts = np.array([[base_x, base_y], [base_x-tail_len, base_y-tail_half], [base_x-tail_len, base_y+tail_half]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=border_th)

    # Prepare diarization segments per identity for speech bubbles
    segs_by_ident = defaultdict(list)
    if isinstance(diarization_results, list):
        for seg in diarization_results:
            ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            s = float(seg.get('start', seg.get('start_time', 0.0)))
            e = float(seg.get('end', seg.get('end_time', s)))
            if e <= s:
                continue
            txt = str(seg.get('text', ''))
            segs_by_ident[ident].append((s, e, txt))
        for k in segs_by_ident:
            segs_by_ident[k].sort(key=lambda x: (x[0], x[1]))

    def _bubble_text_for(ident: str, t_sec: float) -> str:
        lst = segs_by_ident.get(ident, [])
        for (s, e, txt) in lst:
            if s <= t_sec <= e:
                return txt
        return ''

    def _wrap_lines(txt: str, max_cn: int = 16, max_lat: int = 24, max_lines: int = 3):
        wrapped = _wrap_text_for_ass(txt, max_cn, max_lat)
        lines = [ln for ln in wrapped.split('\\N') if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines and not lines[-1].endswith('…'):
                lines[-1] = lines[-1] + '…'
        return lines

    def _draw_rounded_rect(img, x, y, w, h, radius, fill_color, border_color, border_th=2, alpha=0.9):
        x = int(x); y = int(y); w = int(w); h = int(h); radius = int(max(2, radius))
        overlay = img.copy()
        cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), fill_color, thickness=-1)
        cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.rectangle(img, (x+radius, y), (x+w-radius, y+h), border_color, thickness=border_th)
        cv2.rectangle(img, (x, y+radius), (x+w, y+h-radius), border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+h-radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+h-radius), radius, border_color, thickness=border_th)

    def _draw_tail(img, base_x, base_y, to_left: bool, fill_color, border_color, tail_len=14, tail_half=8, alpha=0.9, border_th=2):
        base_x = int(base_x); base_y = int(base_y)
        if to_left:
            pts = np.array([[base_x, base_y], [base_x+tail_len, base_y-tail_half], [base_x+tail_len, base_y+tail_half]], dtype=np.int32)
        else:
            pts = np.array([[base_x, base_y], [base_x-tail_len, base_y-tail_half], [base_x-tail_len, base_y+tail_half]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=border_th)

    # Prepare diarization segments per identity for speech bubbles
    segs_by_ident = defaultdict(list)
    if isinstance(diarization_results, list):
        for seg in diarization_results:
            ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            s = float(seg.get('start', seg.get('start_time', 0.0)))
            e = float(seg.get('end', seg.get('end_time', s)))
            if e <= s:
                continue
            txt = str(seg.get('text', ''))
            segs_by_ident[ident].append((s, e, txt))
        for k in segs_by_ident:
            segs_by_ident[k].sort(key=lambda x: (x[0], x[1]))

    def _bubble_text_for(ident: str, t_sec: float) -> str:
        lst = segs_by_ident.get(ident, [])
        for (s, e, txt) in lst:
            if s <= t_sec <= e:
                return txt
        return ''

    def _wrap_lines(txt: str, max_cn: int = 16, max_lat: int = 24, max_lines: int = 3):
        wrapped = _wrap_text_for_ass(txt, max_cn, max_lat)
        lines = [ln for ln in wrapped.split('\\N') if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                if not lines[-1].endswith('…'):
                    lines[-1] = lines[-1] + '…'
        return lines

    def _draw_rounded_rect(img, x, y, w, h, radius, fill_color, border_color, border_th=2, alpha=0.9):
        x = int(x); y = int(y); w = int(w); h = int(h); radius = int(max(2, radius))
        overlay = img.copy()
        cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), fill_color, thickness=-1)
        cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.rectangle(img, (x+radius, y), (x+w-radius, y+h), border_color, thickness=border_th)
        cv2.rectangle(img, (x, y+radius), (x+w, y+h-radius), border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+h-radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+h-radius), radius, border_color, thickness=border_th)

    def _draw_tail(img, base_x, base_y, to_left: bool, fill_color, border_color, tail_len=14, tail_half=8, alpha=0.9, border_th=2):
        base_x = int(base_x); base_y = int(base_y)
        if to_left:
            pts = np.array([[base_x, base_y], [base_x+tail_len, base_y-tail_half], [base_x+tail_len, base_y+tail_half]], dtype=np.int32)
        else:
            pts = np.array([[base_x, base_y], [base_x-tail_len, base_y-tail_half], [base_x-tail_len, base_y+tail_half]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=border_th)

    # Build memory ROI map from tracks: per frame f -> list of (mem_frame, x, y, s)
    M = 6
    stride = 1
    mem_rois_by_frame = defaultdict(list)
    for tr in tracks:
        frames_arr = tr.get('track', {}).get('frame') if isinstance(tr, dict) else None
        proc = tr.get('proc_track', {}) if isinstance(tr, dict) else {}
        if frames_arr is None or not isinstance(proc, dict):
            continue
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        xs = proc.get('x', []); ys = proc.get('y', []); ss = proc.get('s', [])
        if not frames_list or not xs or not ys or not ss:
            continue
        for l, f in enumerate(frames_list):
            start = max(0, l - M * stride)
            idxs = list(range(start, l, stride))
            for ii in idxs:
                mf = int(frames_list[ii])
                x = float(xs[ii]); y = float(ys[ii]); s = float(ss[ii])
                mem_rois_by_frame[f].append((mf, x, y, s))

    # Thumbnail cache and layout (use flist images)
    thumb_cache = {}
    tile_w = max(1, min(160, fw // 8))
    tile_h = tile_w
    margin = 6
    label_height = 28

    def get_face_thumb_from_flist(frame_index: int, x: float, y: float, s: float):
        key = (frame_index, int(x), int(y), int(s))
        if key in thumb_cache:
            return thumb_cache[key]
        if frame_index < 0 or frame_index >= len(flist):
            return None
        img = cv2.imread(flist[frame_index])
        if img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, int(x - s)); y1 = max(0, int(y - s))
        x2 = min(w, int(x + s)); y2 = min(h, int(y + s))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        thumb = cv2.resize(roi, (tile_w, tile_h))
        thumb_cache[key] = thumb
        return thumb

    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            ident = face.get('identity', 'None')
            color = ID_COLORS.get(ident, (200, 200, 200))
            x1, y1 = int(face['x']-face['s']), int(face['y']-face['s'])
            x2, y2 = int(face['x']+face['s']), int(face['y']+face['s'])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
            # Always show label; append " (speaking)" when active
            if isinstance(ident, str) and ident != 'None':
                label = ident + (" (speaking)" if face['score'] > 0 else "")
                cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Memory bank overlay removed: now shown in side panel
        vOut.write(image)
    vOut.release()
    command = ("%s -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (_FFMPEG_BIN, os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)

def _to_diarize_df(diarize_segments):
    """Normalize diarization output to a pandas DataFrame with columns [start, end, speaker].
    Supports pyannote.core.Annotation, list[dict], or already-DataFrame inputs.
    """
    # Already a DataFrame
    if hasattr(diarize_segments, "__class__") and diarize_segments.__class__.__name__ == "DataFrame":
        return diarize_segments

    # pyannote Annotation -> rows
    try:
        itertracks = getattr(diarize_segments, "itertracks", None)
        if callable(itertracks):
            rows = []
            for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
                rows.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(speaker),
                })
            return pd.DataFrame(rows)
    except Exception:
        pass

    # list of dicts
    if isinstance(diarize_segments, (list, tuple)) and diarize_segments and isinstance(diarize_segments[0], dict):
        rows = []
        for d in diarize_segments:
            rows.append({
                "start": float(d.get("start", 0.0)),
                "end": float(d.get("end", 0.0)),
                "speaker": str(d.get("speaker", "SPEAKER_XX")),
            })
        return pd.DataFrame(rows)

    raise TypeError(f"Unsupported diarization type: {type(diarize_segments)}")


def _best_visible_cuda_index() -> int:
    try:
        if not torch.cuda.is_available():
            return -1
        best_i = 0
        best_free = -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best_free = free
                best_i = i
        return int(best_i)
    except Exception:
        return 0

def speech_diarization(min_speakers: int = None, max_speakers: int = None):
    # Choose the visible CUDA device with most free memory for ASR to avoid OOM
    dev_idx = _best_visible_cuda_index()
    device = "cuda" if dev_idx >= 0 else "cpu"
    audio_file = os.path.join(args.pyaviPath, "audio.wav")
    # Keep conservative batch by default; allow override via env WHISPERX_BATCH
    try:
        batch_size = int(os.environ.get('WHISPERX_BATCH', '8') or 8)
    except Exception:
        batch_size = 8
    compute_type = "int8"  # use int8 for lower memory with large model

    # 1. Transcribe
    # Choose between whisperx internal loader and transformers pipeline (e.g., BELLE-2)
    model_name = os.environ.get("WHISPERX_MODEL", "").strip() or "large-v3"
    use_tf = os.environ.get("USE_TRANSFORMERS_ASR", "").strip().lower() in ("1","true","yes") \
             or model_name.startswith("BELLE-2/")

    if use_tf:
        try:
            from transformers import pipeline as hf_pipeline
        except Exception as e:
            raise RuntimeError("Transformers not available but USE_TRANSFORMERS_ASR requested") from e
        dev_idx_tf = (dev_idx if dev_idx >= 0 else -1)
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model=(model_name if model_name else "BELLE-2/Belle-whisper-large-v3-zh"),
            device=dev_idx_tf,
            chunk_length_s=30,
            stride_length_s=6,
        )
        # Force Chinese transcribe mode as per BELLE docs
        try:
            asr.model.config.forced_decoder_ids = (
                asr.tokenizer.get_decoder_prompt_ids(language="zh", task="transcribe")
            )
        except Exception:
            pass
        # Run transcription with timestamps
        out = asr(audio_file, return_timestamps=True)
        # Expect 'chunks' with {'timestamp':(s,e), 'text':...}
        chunks = out.get("chunks", []) if isinstance(out, dict) else []
        if not chunks:
            raise RuntimeError("Transformers ASR returned no chunks with timestamps; cannot proceed")
        segments = []
        for ch in chunks:
            ts = ch.get("timestamp", None)
            txt = str(ch.get("text", "")).strip()
            if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                continue
            s, e = ts
            if s is None or e is None:
                continue
            s = float(s); e = float(e)
            if e <= s:
                continue
            if not txt:
                continue
            segments.append({"start": s, "end": e, "text": txt})
        if not segments:
            raise RuntimeError("No valid timestamped segments parsed from Transformers ASR output")
        result = {"segments": segments, "language": "zh"}
        audio = whisperx.load_audio(audio_file)  # used by alignment/diarization
        # 2) Optional: align words via whisperx to enrich segments with 'words'
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        except Exception as e:
            # If alignment model is unavailable, proceed without words (downstream handles gracefully)
            pass
        print(result.get("segments", []))
    else:
        # Faster-whisper direct multi-GPU with concurrency across fixed 30s windows
        try:
            from faster_whisper import WhisperModel as FWWhisperModel
        except Exception as e:
            raise RuntimeError("faster-whisper is required but not available in this environment") from e

        n_vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
        dev = "cuda" if n_vis > 0 else "cpu"
        dev_index = list(range(n_vis)) if n_vis > 0 else 0
        num_workers = max(1, n_vis)

        fw_model = FWWhisperModel(
            model_name,
            device=dev,
            device_index=dev_index,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        audio = whisperx.load_audio(audio_file)
        sr = 16000
        # Language detection on first 30s (info only)
        try:
            cut = int(min(len(audio), sr * 30))
            lang, lang_prob = fw_model.detect_language(audio[:cut])
            print(f"Detected language: {lang} ({lang_prob:.2f}) in first 30s of audio...")
        except Exception:
            lang = None

        # Fixed 30s windows; concurrency across GPUs
        total_sec = float(len(audio)) / float(sr)
        chunk_len = 30.0
        bounds = []
        t = 0.0
        while t < total_sec - 1e-6:
            t2 = min(total_sec, t + chunk_len)
            bounds.append((t, t2))
            t = t2

        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = max(1, n_vis)

        def _transcribe_window(s_sec: float, e_sec: float):
            a0 = int(round(s_sec * sr)); a1 = int(round(e_sec * sr))
            wav = audio[a0:a1]
            segs, info = fw_model.transcribe(
                wav,
                language=(lang if isinstance(lang, str) else None),
                beam_size=5,
                patience=1,
                length_penalty=1.0,
                without_timestamps=False,
                vad_filter=False,
                word_timestamps=False,
            )
            out = []
            for seg in segs:
                out.append({
                    "start": s_sec + float(seg.start),
                    "end":   s_sec + float(seg.end),
                    "text":  getattr(seg, 'text', ''),
                })
            # prefer detected language from info if available
            return out, getattr(info, 'language', None)

        merged_segments = []
        lang_final = lang
        if bounds:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_transcribe_window, s, e) for (s, e) in bounds]
                for fut in as_completed(futs):
                    out, l = fut.result()
                    if isinstance(l, str) and not lang_final:
                        lang_final = l
                    merged_segments.extend(out)
        merged_segments.sort(key=lambda d: (float(d.get('start', 0.0)), float(d.get('end', 0.0))))

        result = {"segments": merged_segments, "language": (lang_final or "en")}
        # 2) Align via WhisperX for word timing
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # segments after ASR (+optional alignment)

    # 3. Assign speaker labels
    # Read HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HuggingFace token: set HF_TOKEN or HUGGINGFACE_TOKEN in environment.")
    # Ensure huggingface_hub & pyannote kw compatibility (token vs use_auth_token)
    _patch_hf_hub_token_kw()
    _patch_pyannote_hf_token_kw()
    # Also expose token to HF via standard env so downstream libs can pick it up
    os.environ.setdefault('HUGGINGFACE_HUB_TOKEN', hf_token)

    try:
        # Parallel diarization by window if requested
        W = max(1, int(getattr(args, 'diarWorkers', 1)))
        if W > 1:
            import torch.multiprocessing as mp
            sr = 16000.0
            total_sec = float(len(audio)) / sr if isinstance(audio, (list, np.ndarray)) and len(audio) > 0 else 0.0
            if total_sec <= 0.0:
                # Probe duration via ffprobe as fallback
                try:
                    out_d = subprocess.check_output([
                        'ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1', audio_file
                    ], stderr=subprocess.STDOUT).decode('utf-8').strip()
                    total_sec = float(out_d)
                except Exception:
                    total_sec = 0.0
            if total_sec <= 0.0:
                raise RuntimeError('Unable to determine audio duration for diarization')
            win = float(getattr(args, 'diarWindowSec', 60.0))
            ov = float(getattr(args, 'diarOverlapSec', 3.0))
            # Pre-warm pipeline once to ensure models cached (avoid multi-proc downloads)
            try:
                tlog_local = _StageTimer(os.path.join(args.pyworkPath, 'time_log.jsonl'), meta={'pid': int(os.getpid()), 'video_name': str(args.videoName)})
            except Exception:
                tlog_local = None
            if tlog_local:
                with tlog_local.timer('diar_prepare'):
                    _ = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                    del _
            else:
                _ = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                del _

            # Build windows
            chunks = []
            t = 0.0
            while t < total_sec - 1e-6:
                t2 = min(total_sec, t + win)
                chunks.append((t, t2))
                if t2 >= total_sec:
                    break
                t = t + win - ov
            if not chunks:
                raise RuntimeError('No diarization chunks built')
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            packs = []
            for i, (s0, s1) in enumerate(chunks):
                packs.append((s0, s1, audio_file, min_speakers, max_speakers, hf_token, (i % n_gpus) if n_gpus > 0 else -1))
            ctx = mp.get_context('spawn')
            # Map with timing
            if tlog_local:
                with tlog_local.timer('diar_chunk_map', extra={'chunks': int(len(packs)), 'workers': int(min(W, len(packs)))}):
                    with ctx.Pool(processes=min(W, len(packs))) as pool:
                        out_rows = pool.map(_diar_chunk_worker, packs)
            else:
                with ctx.Pool(processes=min(W, len(packs))) as pool:
                    out_rows = pool.map(_diar_chunk_worker, packs)
            # Merge with timing
            if tlog_local:
                with tlog_local.timer('diar_merge'):
                    merged = [x for rows in out_rows for x in rows]
                    merged.sort(key=lambda d: (d['start'], d['end']))
                    # Drop exact duplicates within small tolerance
                    tol = 0.1
                    diarize_segments = []
                    for d in merged:
                        if diarize_segments:
                            p = diarize_segments[-1]
                            if abs(p['start'] - d['start']) <= tol and abs(p['end'] - d['end']) <= tol and str(p.get('speaker')) == str(d.get('speaker')):
                                continue
                        diarize_segments.append(d)
            else:
                merged = [x for rows in out_rows for x in rows]
                merged.sort(key=lambda d: (d['start'], d['end']))
                tol = 0.1
                diarize_segments = []
                for d in merged:
                    if diarize_segments:
                        p = diarize_segments[-1]
                        if abs(p['start'] - d['start']) <= tol and abs(p['end'] - d['end']) <= tol and str(p.get('speaker')) == str(d.get('speaker')):
                            continue
                    diarize_segments.append(d)
        else:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            try:
                if min_speakers is not None or max_speakers is not None:
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                else:
                    diarize_segments = diarize_model(audio)
            except Exception:
                if min_speakers is not None or max_speakers is not None:
                    diarize_segments = diarize_model(
                        audio_file,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                else:
                    diarize_segments = diarize_model(audio_file)
    except AttributeError:
        # Fallback for older WhisperX versions
        from pyannote.audio import Pipeline
        diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        if min_speakers is not None or max_speakers is not None:
            diarize_segments = diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            diarize_segments = diarize_model(audio_file)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # Assign speakers to words/segments via whisperx using a normalized DataFrame
    diarize_df = _to_diarize_df(diarize_segments)
    result = whisperx.assign_word_speakers(diarize_df, result)
    
    # print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    return result

def _flatten_aligned_words(aligned_segments):
    """Flatten aligned WhisperX segments into a list of (start, end, text) words.
    Expects each segment to have 'words' with entries having 'start','end','word'.
    Returns a list sorted by start time.
    """
    words = []
    for seg in aligned_segments:
        ws = seg.get('words', []) or []
        for w in ws:
            try:
                t0 = float(w.get('start', None))
                t1 = float(w.get('end', None))
            except Exception:
                continue
            if t0 is None or t1 is None:
                continue
            if t1 <= t0:
                continue
            txt = str(w.get('word', '')).strip()
            if not txt:
                continue
            words.append((t0, t1, txt))
    words.sort(key=lambda x: (x[0], x[1]))
    return words

def rebuild_segments_with_visual_asd(annotated_tracks, scores, aligned_segments, fps=25.0,
                                     tau=0.2, min_seg=0.15, merge_gap=0.10):
    """Rebuild diarization purely from visual identities + ASD.

    - aligned_segments: WhisperX aligned output with per-segment words; 'speaker' is ignored.
    - Build ASD-active intervals per identity; split time by these boundaries.
    - Assign identity to each resulting chunk by max-overlap ratio (>= tau).
    - Reconstruct text by concatenating overlapped aligned words within each chunk.
    Returns list of dicts: {'start','end','identity','text'}.
    """
    # 1) Split and assign using existing visual refinement logic
    # Use an absolute-overlap threshold tied to time resolution. When ASD sequences are short
    # (e.g., produced from few windows), requiring 0.15s can drop all assignments. One score
    # step is roughly 1/fps seconds after resampling, so accept that as minimal evidence.
    min_abs = max(1.0 / float(fps if fps and fps > 0 else 25.0), 0.02)
    # Prefer using diarization speaker when visual evidence is insufficient to avoid losing subtitles.
    refined = refine_diarization_with_visual(
        annotated_tracks, scores, aligned_segments, fps=fps, tau=tau,
        min_seg=min_seg, merge_gap=merge_gap, argmax_only=False, min_abs_overlap=min_abs
    )
    if not refined:
        raise RuntimeError("Visual-ASD rebuild produced no segments; cannot continue.")

    # 2) Rebuild text from aligned words per refined segment
    words = _flatten_aligned_words(aligned_segments)
    out = []
    for seg in refined:
        s = float(seg.get('start', 0.0))
        e = float(seg.get('end', s))
        if e <= s:
            continue
        ident = seg.get('speaker', None)
        # Select words overlapping this interval
        toks = []
        for (t0, t1, wtxt) in words:
            if t1 <= s:
                continue
            if t0 >= e:
                break
            toks.append(wtxt)
        # Join tokens; naive spacing for Latin, no extra spacing for CJK
        if toks:
            has_cjk = any('\u4e00' <= ch <= '\u9fff' for tk in toks for ch in tk)
            if has_cjk:
                text = ''.join(toks)
            else:
                text = ' '.join(toks)
        else:
            text = ''
        out.append({'start': s, 'end': e, 'identity': ident, 'text': text})
    # Keep only segments with a resolved identity and some text content
    out = [x for x in out if (isinstance(x.get('identity'), str) and x['identity'] not in (None, 'None') and x.get('text'))]
    if not any((isinstance(x.get('identity'), str) and x['identity'] not in (None, 'None')) for x in out):
        raise RuntimeError("Visual-ASD rebuild assigned no identities above threshold; aborting to avoid wrong subtitles.")
    return out

def match_speaker_identity(vidTracks, scores, diarization_result, fps=25):
    """Assign Person_* by maximizing ASD energy overlap on a unified time base (seconds).

    Sums ReLU(score) over ASD time points t within each diarization segment [s,e).
    Uses PTS-mapped frame times to compute t for each track frame.
    """
    matched_results = []
    frame_times_sec = _sanitize_frame_times(_probe_frame_pts(args.videoFilePath))
    for diar in diarization_result:
        if "speaker" not in diar:
            continue
        start_time = float(diar.get("start", diar.get("start_time", 0.0)))
        end_time = float(diar.get("end", diar.get("end_time", start_time)))
        if end_time <= start_time:
            continue
        speaker = diar["speaker"]
        best_match_identity = None
        best_energy = -1.0
        for i, tr in enumerate(vidTracks):
            identity = tr.get("identity", None)
            if not (isinstance(identity, str) and identity not in (None, 'None')):
                continue
            frames = tr["track"].get("frame") if isinstance(tr.get("track"), dict) else None
            if frames is None:
                continue
            sc = scores[i] if i < len(scores) else []
            if not isinstance(sc, (list, tuple)) or len(sc) == 0:
                continue
            fr_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            import numpy as _np
            sc_arr = _np.asarray(sc, dtype=float) if len(sc) > 0 else _np.array([], dtype=float)
            T = min(len(fr_list), sc_arr.shape[0])
            e = 0.0
            for j in range(T):
                fidx = int(fr_list[j])
                if fidx < 0 or fidx >= len(frame_times_sec):
                    continue
                t = float(frame_times_sec[fidx])
                if (start_time <= t) and (t < end_time):
                    v = float(sc_arr[j])
                    if v > 0.0:
                        e += v
            if e > best_energy:
                best_energy = e
                best_match_identity = identity

        matched_results.append({
            "speaker": speaker,
            "identity": best_match_identity,
            "text": diar.get("text", ""),
            "start_time": start_time,
            "end_time": end_time,
        })

    return matched_results




def autofill_and_correct_matches(matched_results):
    # Track the most common identity for each speaker
    speaker_identity_map = defaultdict(list)

    # First pass: build a map of speaker to identities (including "None" as a valid identity)
    for result in matched_results:
        speaker = result['speaker']
        identity = result['identity']
        speaker_identity_map[speaker].append(identity)

    # Determine the most frequent identity for each speaker (including "None")
    speaker_most_common_identity = {
        speaker: Counter(identities).most_common(1)[0][0]
        for speaker, identities in speaker_identity_map.items()
    }

    # Second pass: autofill and correct identities based on consistency
    for result in matched_results:
        speaker = result['speaker']
        if result['identity'] is None or result['identity'] != speaker_most_common_identity[speaker]:
            # Autofill or correct the identity with the most consistent one
            result['identity'] = speaker_most_common_identity[speaker]

    return matched_results

def _aggregate_overlap_counts_by_speaker(annotated_tracks, raw_segments, fps: float = 25.0):
    """Aggregate visual overlap frame counts per diarization speaker vs Person_* identity.

    Does NOT use ASD; counts number of track frames that fall inside the diarization
    speaker's time ranges. Returns dict: speaker -> {identity -> count}.
    """
    from collections import defaultdict
    import bisect
    counts = defaultdict(lambda: defaultdict(int))
    for diar in raw_segments:
        if 'speaker' not in diar:
            continue
        s = float(diar.get('start', diar.get('start_time', 0.0)))
        e = float(diar.get('end', diar.get('end_time', s)))
        if e <= s:
            continue
        ds = int(s * float(fps)); de = int(e * float(fps))
        spk = diar['speaker']
        for tr in annotated_tracks:
            ident = tr.get('identity', None)
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            frames = tr['track']['frame'] if 'track' in tr and 'frame' in tr['track'] else None
            if frames is None:
                continue
            fr_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            if not fr_list:
                continue
            t0 = int(fr_list[0]); t1 = int(fr_list[-1])
            a = max(t0, ds); b = min(t1, de)
            if a >= b:
                continue
            # count frames f in [a, b]
            lo = bisect.bisect_left(fr_list, a)
            hi = bisect.bisect_right(fr_list, b)
            c = max(0, hi - lo)
            if c > 0:
                counts[spk][ident] += int(c)
    return counts

def build_global_speaker_to_person_map(annotated_tracks, raw_segments, fps: float = 25.0):
    """Return a dict mapping each diarization speaker (e.g., SPEAKER_01) to a Person_* identity.

    - Uses aggregated visual overlap counts over all segments of the speaker (no ASD).
    - If a speaker has no positive energy for any Person_*, raises RuntimeError (no fallback).
    """
    counts = _aggregate_overlap_counts_by_speaker(annotated_tracks, raw_segments, fps=fps)
    mapping = {}
    for spk, c_map in counts.items():
        if not c_map:
            raise RuntimeError(f"No overlapping frames found for speaker {spk}; cannot map to Person_* without fallback.")
        # Choose identity with maximum overlap count
        ident, val = max(c_map.items(), key=lambda x: x[1])
        if int(val) <= 0:
            raise RuntimeError(f"Non-positive overlap for speaker {spk}; refusing to map with zero evidence.")
        mapping[spk] = ident
    # Ensure all speakers present in diarization are covered
    spk_all = [seg.get('speaker') for seg in raw_segments if 'speaker' in seg]
    for sp in set(spk_all):
        if sp not in mapping:
            raise RuntimeError(f"Missing mapping for diarization speaker {sp}; no ASD-supported Person_* found.")
    return mapping

def apply_global_mapping_to_segments(raw_segments, speaker_to_person_map):
    """Build per-segment Person_* assignments using a global speaker->Person_* map.

    Returns list of {'start','end','identity','text'}.
    """
    out = []
    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        spk = seg.get('speaker')
        if spk not in speaker_to_person_map:
            raise RuntimeError(f"No mapping for segment speaker {spk} in global map.")
        out.append({
            'start': s,
            'end': e,
            'identity': speaker_to_person_map[spk],
            'text': seg.get('text', ''),
        })
    return out

def _per_frame_identity_scores(annotated_tracks, scores):
    """Build per-frame identity score dict.

    Returns: dict[global_frame] -> dict[identity] = score (float)
    - For identities with multiple tracks on the same frame, keeps max score.
    """
    per_frame = {}
    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity', None)
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        frames = tr.get('track', {}).get('frame') if isinstance(tr, dict) else None
        if frames is None:
            continue
        sc = scores[i] if i < len(scores) else []
        try:
            import numpy as _np
            sc_arr = _np.asarray(sc, dtype=float)
        except Exception:
            continue
        fr_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
        T = min(len(fr_list), int(sc_arr.shape[0]))
        for j in range(T):
            f = int(fr_list[j])
            v = float(sc_arr[j])
            m = per_frame.get(f)
            if m is None:
                per_frame[f] = {ident: v}
            else:
                if ident not in m or v > m[ident]:
                    m[ident] = v
    return per_frame

def split_segments_by_frame_argmax(annotated_tracks, scores, raw_segments, fps: float = 25.0, min_run_frames: int = 6):
    """Split each diarization segment into subsegments by per-frame ASD argmax identity.

    - For each frame in a segment, pick identity with max score among identities present at that frame.
    - Frames with no identity present in that frame are filled with the sentence-level top identity
      (by presence count over the segment).
    - Consecutive frames with same identity are merged; very short runs (< min_run_frames) are
      absorbed into the longer adjacent neighbor to reduce fragmentation.
    - Returns list of dicts {'start','end','identity','text'}.
    """
    if min_run_frames is None:
        min_run_frames = 0
    per_frame = _per_frame_identity_scores(annotated_tracks, scores)
    out = []
    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        a = int(s * float(fps))
        b = int(e * float(fps))
        if b <= a:
            continue
        # Determine sentence-level top identity by presence within [a,b)
        from collections import defaultdict
        pres = defaultdict(int)
        for f in range(a, b):
            m = per_frame.get(f, None)
            if not m:
                continue
            # count presence for identities defined at this frame
            for ident in m.keys():
                pres[ident] += 1
        if not pres:
            # No identities present over this sentence; cannot label without fake data
            raise RuntimeError("No identity presence within diarization segment; cannot assign labels.")
        sent_top = max(pres.items(), key=lambda x: x[1])[0]

        # Build per-frame winners with fill for empty frames
        winners = []  # list of identity strings, len = b-a
        frames = list(range(a, b))
        for f in frames:
            m = per_frame.get(f, None)
            if m:
                # pick highest score identity among those defined at f
                ident = max(m.items(), key=lambda x: x[1])[0]
                winners.append(ident)
            else:
                # fill with sentence-level top identity to avoid gaps
                winners.append(sent_top)

        # Run-length compress winners into (start_f, end_f, ident)
        runs = []
        cur_ident = None
        cur_start = None
        for idx, ident in enumerate(winners):
            if ident != cur_ident:
                if cur_ident is not None:
                    runs.append((frames[cur_start], frames[idx], cur_ident))  # [start, end)
                cur_ident = ident
                cur_start = idx
        if cur_ident is not None:
            runs.append((frames[cur_start], frames[-1] + 1, cur_ident))

        # Absorb very short runs into longer neighbor to reduce flicker
        if min_run_frames > 0 and len(runs) >= 2:
            merged = []
            i = 0
            while i < len(runs):
                rs, re, rid = runs[i]
                length = re - rs
                if length >= min_run_frames or len(runs) == 1:
                    merged.append([rs, re, rid])
                    i += 1
                    continue
                # short run: merge into neighbor with larger duration
                if i == 0:
                    # merge right
                    nr_s, nr_e, nr_id = runs[i + 1]
                    merged.append([rs, nr_e, nr_id])
                    i += 2
                elif i == len(runs) - 1:
                    # merge left
                    ml_s, ml_e, ml_id = merged[-1]
                    merged[-1] = [ml_s, re, ml_id]
                    i += 1
                else:
                    # choose neighbor with longer duration
                    pl_s, pl_e, pl_id = merged[-1]
                    nr_s, nr_e, nr_id = runs[i + 1]
                    if (pl_e - pl_s) >= (nr_e - nr_s):
                        merged[-1] = [pl_s, re, pl_id]
                        i += 1
                    else:
                        merged.append([rs, nr_e, nr_id])
                        i += 2
            # coalesce adjacent same-identity after merges
            runs2 = []
            for rs, re, rid in merged:
                if runs2 and runs2[-1][2] == rid:
                    runs2[-1][1] = re
                else:
                    runs2.append([rs, re, rid])
            runs = [(rs, re, rid) for rs, re, rid in runs2]

        # Convert runs to segments
        for rs, re, rid in runs:
            st = max(s, rs / float(fps))
            et = min(e, re / float(fps))
            if et <= st:
                continue
            out.append({'start': st, 'end': et, 'identity': rid, 'text': seg.get('text', '')})

    return out

def split_segments_by_positive_fill(annotated_tracks, scores, raw_segments, fps: float = 25.0, min_run_frames: int = 6):
    """Split diarization segments into subsegments using ASD evidence on a unified time base (seconds).

    - Build per-identity positive-time lists from track frames mapped via PTS to seconds.
    - Build per-time (ms) map of identity->score at that instant.
    - For each diarization segment [s,e):
        * sentence-level winner = identity with max count of positive-time events in [s,e)
        * grid-sample at 25fps between [s,e) to assign per-step winner using per-time map; fill with sentence winner when ambiguous/missing
        * RLE compress and absorb short runs using time thresholds
    """
    # Map track frame indices to seconds via PTS (sanitized)
    frame_times_sec = _sanitize_frame_times(_probe_frame_pts(args.videoFilePath))

    from collections import defaultdict
    id_pos_times = defaultdict(list)           # ident -> list[float seconds]
    per_time_ms = defaultdict(dict)            # ms(int) -> {ident: score}

    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity', None)
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        frames = tr.get('track', {}).get('frame')
        if frames is None:
            continue
        fr_list = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
        sc = scores[i] if i < len(scores) else []
        try:
            import numpy as _np
            sc_arr = _np.asarray(sc, dtype=float)
        except Exception:
            sc_arr = []
        T = min(len(fr_list), int(getattr(sc_arr, 'shape', [0])[0] if hasattr(sc_arr, 'shape') else len(sc)))
        for j in range(T):
            f = int(fr_list[j])
            if f < 0 or f >= len(frame_times_sec):
                continue
            t = float(frame_times_sec[f])
            v = float(sc_arr[j])
            ms = int(round(t * 1000.0))
            # positive evidence by time
            if v > 0.0:
                id_pos_times[ident].append(t)
            # strongest score per time instant
            m = per_time_ms[ms]
            if (ident not in m) or (v > m[ident]):
                m[ident] = v

    # Sort positive time lists for efficient counting
    for ident in list(id_pos_times.keys()):
        id_pos_times[ident].sort()

    out = []
    dt = 1.0 / 25.0
    min_run_sec = float(min_run_frames) / float(fps if fps and fps > 0 else 25.0)

    import bisect
    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        # 1) sentence-level winner by counting positives in [s,e)
        pos_counts = {}
        for ident, tlist in id_pos_times.items():
            lo = bisect.bisect_left(tlist, s)
            hi = bisect.bisect_left(tlist, e)
            c = max(0, hi - lo)
            if c > 0:
                pos_counts[ident] = pos_counts.get(ident, 0) + c
        if not pos_counts:
            # No positive ASD evidence: emit as unassigned
            out.append({'start': s, 'end': e, 'identity': 'None', 'text': seg.get('text', '')})
            continue
        sent_top = max(pos_counts.items(), key=lambda x: x[1])[0]

        # 2) per-step assignment on 25fps grid within [s,e)
        times = []
        winners = []
        t_cur = s
        while t_cur < e - 1e-9:
            ms = int(round(t_cur * 1000.0))
            m = per_time_ms.get(ms, None)
            if not m:
                winners.append(sent_top)
            else:
                # pick if exactly one positive; else fill with sentence winner
                pos = [iden for iden, val in m.items() if float(val) > 0.0]
                if len(pos) == 1:
                    winners.append(pos[0])
                else:
                    winners.append(sent_top)
            times.append(t_cur)
            t_cur += dt
        # Ensure we cover the end boundary
        if not times or times[-1] < e:
            times.append(e)
            winners.append(winners[-1] if winners else sent_top)

        # 3) RLE over time grid
        runs = []
        cur_ident = None
        cur_start_t = None
        for idx, ident in enumerate(winners):
            if ident != cur_ident:
                if cur_ident is not None:
                    runs.append((cur_start_t, times[idx], cur_ident))
                cur_ident = ident
                cur_start_t = times[idx]
        if cur_ident is not None:
            runs.append((cur_start_t, times[-1], cur_ident))

        # absorb short runs (by seconds)
        if min_run_sec > 0 and len(runs) >= 2:
            merged = []
            i = 0
            while i < len(runs):
                rs, re, rid = runs[i]
                L = float(re - rs)
                if L >= min_run_sec or len(runs) == 1:
                    merged.append([rs, re, rid]); i += 1; continue
                if i == 0:
                    nrs, nre, nrid = runs[i + 1]
                    merged.append([rs, nre, nrid]); i += 2
                elif i == len(runs) - 1:
                    prs, pre, prid = merged[-1]
                    merged[-1] = [prs, re, prid]
                    i += 1
                else:
                    prs, pre, prid = merged[-1]
                    nrs, nre, nrid = runs[i + 1]
                    if (pre - prs) >= (nre - nrs):
                        merged[-1] = [prs, re, prid]
                        i += 1
                    else:
                        merged.append([rs, nre, nrid])
                        i += 2
            # coalesce
            runs2 = []
            for rs, re, rid in merged:
                if runs2 and runs2[-1][2] == rid:
                    runs2[-1][1] = re
                else:
                    runs2.append([rs, re, rid])
            runs = [(rs, re, rid) for rs, re, rid in runs2]

        # 4) emit segments clipped to [s,e)
        for rs, re, rid in runs:
            st = max(s, float(rs))
            et = min(e, float(re))
            if et <= st:
                continue
            out.append({'start': st, 'end': et, 'identity': rid, 'text': seg.get('text', '')})
    return out
def _global_top_identity_by_asd(annotated_tracks, scores):
    id_speaking = {}
    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity')
        if not (isinstance(ident, str) and ident not in (None, 'None')):
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
    if not id_speaking:
        return None
    return sorted(id_speaking.items(), key=lambda x: x[1], reverse=True)[0][0]

def map_segments_to_person(annotated_tracks, scores, raw_segments, fps=25):
    """Map diarization segments to Person_* identities using ASD overlap, with consistent per-speaker fill.

    Steps:
      1) match_speaker_identity: assigns identity by counting ASD-active frames overlapping segment
      2) autofill_and_correct_matches: enforces consistent identity per diarization speaker
      3) any remaining None identities are filled by the globally most-speaking Person_* (by ASD)
    Returns list of {'start','end','identity','text'} suitable for ASS rendering.
    """
    matched = match_speaker_identity(annotated_tracks, scores, raw_segments, fps=fps)
    matched = autofill_and_correct_matches(matched)
    # Fallback fill for any None using global ASD-dominant identity (data-driven)
    fallback_ident = _global_top_identity_by_asd(annotated_tracks, scores)
    out = []
    for m in matched:
        ident = m.get('identity')
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            ident = fallback_ident
            # If still None, fall back to visual presence coverage in this interval
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                s = float(m.get('start_time', m.get('start', 0.0)))
                e = float(m.get('end_time', m.get('end', s)))
                if e > s:
                    a_f = int(s * float(fps))
                    b_f = int(e * float(fps))
                    best_cov = -1
                    best_id = None
                    for tr in annotated_tracks:
                        tid = tr.get('identity')
                        if not (isinstance(tid, str) and tid not in (None, 'None')):
                            continue
                        frs = tr['track']['frame']
                        fr_list = frs.tolist() if hasattr(frs, 'tolist') else list(frs)
                        if not fr_list:
                            continue
                        t0 = int(fr_list[0]); t1 = int(fr_list[-1])
                        ov_a = max(t0, a_f); ov_b = min(t1, b_f)
                        cov = max(0, ov_b - ov_a + 1)
                        if cov > best_cov:
                            best_cov = cov
                            best_id = tid
                    if isinstance(best_id, str) and best_id not in (None, 'None') and best_cov > 0:
                        ident = best_id
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            # No valid identity available; skip segment to avoid fake labels
            continue
        s = float(m.get('start_time', m.get('start', 0.0)))
        e = float(m.get('end_time', m.get('end', s)))
        if e <= s:
            continue
        text = str(m.get('text', ''))
        out.append({'start': s, 'end': e, 'identity': ident, 'text': text})
    return out

def _smooth_person_segments(segments, max_flip_dur: float = 0.5):
    """Reduce identity flicker like A-B-A by absorbing short flips.

    - If a middle segment B is shorter than max_flip_dur and neighbors are both A,
      change B's identity to A and merge durations.
    - Returns a new list; input is not modified.
    """
    if not segments:
        return []
    segs = [dict(s) for s in segments]
    i = 1
    while i + 1 < len(segs):
        prev, cur, nxt = segs[i-1], segs[i], segs[i+1]
        a = prev.get('identity'); b = cur.get('identity'); c = nxt.get('identity')
        if isinstance(a, str) and isinstance(c, str) and a == c and b != a:
            s = float(cur.get('start', cur.get('start_time', 0.0)))
            e = float(cur.get('end', cur.get('end_time', s)))
            if (e - s) <= max_flip_dur:
                cur['identity'] = a
        i += 1
    # Merge adjacent segments with same identity
    merged = []
    for seg in segs:
        if merged and seg.get('identity') == merged[-1].get('identity'):
            merged[-1]['end'] = max(float(merged[-1].get('end', merged[-1].get('end_time', 0.0))), float(seg.get('end', seg.get('end_time', 0.0))))
            # concatenate text for readability
            t_prev = merged[-1].get('text','')
            t_cur = seg.get('text','')
            if t_cur:
                if t_prev and any('\\u4e00' <= ch <= '\\u9fff' for ch in (t_prev[-1],)):
                    merged[-1]['text'] = t_prev + t_cur
                else:
                    merged[-1]['text'] = (t_prev + ' ' + t_cur).strip()
        else:
            merged.append(seg)
    return merged

def _intervals_from_active_frames(frames, scores, fps=25):
    """Build active time intervals from per-track frames and ASD scores.
    frames: array-like of frame indices for this track (ascending)
    scores: list/array of ASD scores aligned to frames within the track
    Returns list of (start_time, end_time) intervals (seconds) where score > 0.
    """
    if scores is None or len(scores) == 0:
        return []
    base_f = int(frames[0])
    active = []
    start = None
    for j, val in enumerate(scores):
        on = (val > 0)
        if on and start is None:
            start = j
        if (not on) and start is not None:
            s_t = (base_f + start) / float(fps)
            e_t = (base_f + j) / float(fps)
            if e_t > s_t:
                active.append((s_t, e_t))
            start = None
    if start is not None:
        s_t = (base_f + start) / float(fps)
        e_t = (base_f + len(scores)) / float(fps)
        if e_t > s_t:
            active.append((s_t, e_t))
    # Merge adjacent/overlapping intervals
    if not active:
        return []
    active.sort()
    merged = [active[0]]
    for s, e in active[1:]:
        ms, me = merged[-1]
        if s <= me + 1e-6:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged

def _overlap_dur(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def refine_diarization_with_visual(annotated_tracks, scores, raw_segments, fps=25, tau=0.3, min_seg=0.08, merge_gap=0.2, argmax_only=False, min_abs_overlap=0.0):
    """Refine WhisperX diarization using visual active tracks + identities.
    - Split segments at visual activity change points
    - Assign identity per sub-segment if overlap_ratio >= tau
    - Merge adjacent segments with same identity and small gaps
    Returns list of dicts: {'start':, 'end':, 'speaker':, 'text':}
    """
    # Build active intervals per track identity
    track_intervals = []  # list of (identity, intervals)
    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity', None)
        frames = tr['track']['frame']
        sc = scores[i] if i < len(scores) else []
        intervals = _intervals_from_active_frames(frames, sc, fps=fps)
        if intervals:
            track_intervals.append((ident, intervals))

    def boundaries_in(s, e):
        b = {s, e}
        for _, ivs in track_intervals:
            for a, bnd in ivs:
                if s < a < e:
                    b.add(a)
                if s < bnd < e:
                    b.add(bnd)
        return sorted(b)

    refined = []

    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        text = seg.get('text', '')
        orig_spk = seg.get('speaker', None)
        # Split at visual boundaries
        cuts = boundaries_in(s, e)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if (b - a) < min_seg:
                continue
            # Assign identity by max overlap
            best_id = None
            best_ov = (-1.0 if argmax_only else 0.0)
            for ident, ivs in track_intervals:
                ov = 0.0
                for iv in ivs:
                    ov += _overlap_dur((a, b), iv)
                if ov > best_ov:
                    best_ov = ov
                    best_id = ident
            ratio = best_ov / max(1e-6, (b - a))
            if argmax_only:
                # In argmax mode, enforce an absolute overlap threshold; otherwise drop identity (None)
                label = best_id if (best_id is not None and best_ov >= float(min_abs_overlap)) else None
            else:
                label = best_id if (best_id is not None and ratio >= tau) else orig_spk
            refined.append({'start': a, 'end': b, 'speaker': label, 'text': text})

    # Merge adjacent same-speaker segments with small gaps
    if not refined:
        return []
    refined.sort(key=lambda x: (x['start'], x['end']))
    merged = [refined[0].copy()]
    for cur in refined[1:]:
        prev = merged[-1]
        if cur['speaker'] == prev['speaker'] and (cur['start'] - prev['end']) <= merge_gap:
            prev['end'] = max(prev['end'], cur['end'])
            # keep first text
        else:
            merged.append(cur.copy())
    return merged

def refine_diarization_boundaries(raw_segments, pad=0.05, gap_split=0.25, min_seg=0.15, merge_gap=0.10, close_gap=0.12):
    """Refine WhisperX diarization boundaries using aligned words.
    - Snap segment to min/max word times with small padding
    - Split segments at long internal silences (> gap_split)
    - Drop/absorb very short slivers (< min_seg)
    - Merge adjacent same-speaker segments with small gaps (< merge_gap)
    Returns list of dicts with keys: start, end, speaker, text
    """
    # Enforce a harder minimum to avoid over-fragmentation after snapping/splitting
    # Use at least twice min_seg and not smaller than ~0.5s (close to common collars)
    min_seg_hard = max(min_seg * 2.0, 0.5)
    refined = []

    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        spk = seg.get('speaker', None)
        text = seg.get('text', '')
        words = seg.get('words', []) or []
        # Extract valid words
        ws = []
        for w in words:
            try:
                ws.append((float(w.get('start', 0.0)), float(w.get('end', 0.0))))
            except Exception:
                continue
        ws = [(ws_, we_) for ws_, we_ in ws if we_ > ws_]
        ws.sort()
        if not ws:
            # No words: keep original segment
            refined.append({'start': s, 'end': e, 'speaker': spk, 'text': text})
            continue

        # Snap to word bounds with padding
        ws0 = min(t0 for t0, _ in ws)
        weN = max(t1 for _, t1 in ws)
        snapped_start = max(s, ws0 - pad)
        snapped_end = min(e, weN + pad)
        if snapped_end <= snapped_start:
            continue

        # Split on long internal silences between words
        splits = [snapped_start]
        prev_end = ws[0][1]
        for t0, t1 in ws[1:]:
            gap = t0 - prev_end
            if gap > gap_split:
                # split at midpoint of gap
                cut = prev_end + gap / 2.0
                # only split if both sides remain reasonably long
                if (cut - splits[-1]) >= min_seg_hard and (snapped_end - cut) >= min_seg_hard:
                    splits.append(cut)
            prev_end = t1
        splits.append(snapped_end)

        # Create sub-segments per split range
        prev_a = splits[0]
        groups = []
        for b in splits[1:]:
            a = prev_a
            prev_a = b
            if (b - a) < min_seg_hard:
                # too short; accumulate by merging later
                groups.append((a, b))
            else:
                groups.append((a, b))

        # Merge tiny slivers into neighbors
        merged_groups = []
        for g in groups:
            if not merged_groups:
                merged_groups.append(g)
                continue
            a, b = g
            if (b - a) < min_seg_hard:
                # absorb into previous
                pa, pb = merged_groups[-1]
                merged_groups[-1] = (pa, max(pb, b))
            else:
                merged_groups.append((a, b))

        for a, b in merged_groups:
            if b - a >= min_seg_hard:
                refined.append({'start': a, 'end': b, 'speaker': spk, 'text': text})

    # Merge adjacent same-speaker segments with small gaps
    if not refined:
        return []
    refined.sort(key=lambda x: (x['start'], x['end']))
    out = [refined[0].copy()]
    for cur in refined[1:]:
        prev = out[-1]
        if cur['speaker'] == prev['speaker'] and (cur['start'] - prev['end']) <= merge_gap:
            prev['end'] = max(prev['end'], cur['end'])
        else:
            out.append(cur.copy())

    # Overlap trimming and gap snapping across different-speaker boundaries
    if len(out) <= 1:
        return out
    trimmed = [out[0].copy()]
    for i in range(1, len(out)):
        prev = trimmed[-1]
        cur = out[i].copy()
        if prev['speaker'] != cur['speaker']:
            # Overlap case
            if prev['end'] > cur['start']:
                cut = 0.5 * (prev['end'] + cur['start'])
                # Ensure resulting segments are not too short
                # Adjust cut if needed to respect min_seg from segment endpoints
                lo = max(prev['start'] + min_seg_hard * 0.5, cur['start'])
                hi = min(prev['end'], cur['end'] - min_seg_hard * 0.5)
                cut = min(max(cut, lo), hi)
                # Apply cut
                prev_end_new = max(prev['start'], cut)
                cur_start_new = min(cur['end'], cut)
                # Only keep if durations are valid
                if (prev_end_new - prev['start']) >= min_seg_hard:
                    prev['end'] = prev_end_new
                # else: keep prev as-is (will likely be < min_seg; handled by next check)
                if (cur['end'] - cur_start_new) >= min_seg_hard:
                    cur['start'] = cur_start_new
            else:
                # Tiny gap snapping
                gap = cur['start'] - prev['end']
                if gap <= close_gap:
                    mid = 0.5 * (prev['end'] + cur['start'])
                    prev['end'] = mid
                    cur['start'] = mid

            # Drop segments that became too short
            if (prev['end'] - prev['start']) < min_seg_hard:
                # Remove prev by merging its time into current start if overlapping
                if prev['end'] > cur['start']:
                    cur['start'] = min(cur['start'], prev['end'])
                trimmed[-1] = cur
                continue
            if (cur['end'] - cur['start']) < min_seg_hard:
                # Skip current segment
                trimmed[-1] = prev
                continue
            trimmed[-1] = prev
            trimmed.append(cur)
        else:
            # Same speaker adjacency: merge if close
            if cur['start'] - prev['end'] <= merge_gap:
                prev['end'] = max(prev['end'], cur['end'])
                trimmed[-1] = prev
            else:
                trimmed.append(cur)

    return trimmed

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def _wrap_text_for_ass(text: str, max_chars_cn: int = 16, max_chars_lat: int = 24) -> str:
    # Basic punctuation-aware wrapping: prefer breaking at sentence-ending punctuation, then spaces, otherwise hard-wrap.
    if not text:
        return ""
    # Normalize braces to avoid colliding with ASS override tags
    t = str(text).replace('{', '(').replace('}', ')')
    # Detect CJK presence
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in t)
    maxw = max_chars_cn if has_cjk else max_chars_lat
    # Split by sentence punctuation first
    seps = ['。','！','？','；','!','?',';']
    parts = []
    buf = ''
    for ch in t:
        buf += ch
        if ch in seps:
            parts.append(buf.strip())
            buf = ''
    if buf.strip():
        parts.append(buf.strip())
    # Wrap each part to max width
    lines = []
    for p in parts:
        if len(p) <= maxw:
            lines.append(p)
            continue
        if has_cjk:
            # Hard-wrap every maxw characters
            for i in range(0, len(p), maxw):
                lines.append(p[i:i+maxw])
        else:
            # Word-aware wrap for latin text
            cur = []
            cur_len = 0
            for w in p.split():
                if (cur_len + (1 if cur else 0) + len(w)) > maxw:
                    lines.append(' '.join(cur))
                    cur = [w]
                    cur_len = len(w)
                else:
                    cur.append(w)
                    cur_len += (1 if cur_len>0 else 0) + len(w)
            if cur:
                lines.append(' '.join(cur))
    return '\\N'.join([ln for ln in (ln.strip() for ln in lines) if ln])

def _bgr_to_ass_hex(color_bgr):
    # ASS expects &HBBGGRR (no alpha here; alpha handled separately in styles)
    b, g, r = color_bgr
    return f"&H{b:02X}{g:02X}{r:02X}"

def _normalize_identity_prefix(ident: str) -> str:
    if isinstance(ident, str) and ident.startswith('VID_'):
        return 'Person_' + ident.split('_', 1)[1]
    return ident

def _collapse_to_single_line(segments):
    # Ensure at most one subtitle is active at any time by trimming previous end to next start
    # Input: list of dicts with 'start'/'start_time' and 'end'/'end_time'
    # Output: new list with no overlaps
    canon = []
    for seg in segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e > s:
            canon.append({'start': s, 'end': e, 'identity': seg.get('identity'), 'text': seg.get('text', '')})
    if not canon:
        return []
    canon.sort(key=lambda x: (x['start'], x['end']))
    out = []
    for seg in canon:
        if not out:
            out.append(seg)
            continue
        prev = out[-1]
        if seg['start'] < prev['end']:
            # trim previous to avoid overlap; drop if becomes invalid
            new_end = max(prev['start'], min(prev['end'], seg['start']))
            prev['end'] = new_end
            if prev['end'] <= prev['start']:
                out.pop()
        out.append(seg)
    # Final cleanup: remove any non-positive durations
    out2 = [s for s in out if (s['end'] - s['start']) > 1e-3]
    return out2

# Compatibility shim: huggingface_hub >= 0.20 removed 'use_auth_token' kw in favor of 'token'.
# Some dependencies (pyannote.audio, whisperx) still pass 'use_auth_token'.
def _patch_hf_hub_token_kw():
    try:
        import huggingface_hub as _h
    except Exception:
        return
    # Patch hf_hub_download
    try:
        _orig = _h.hf_hub_download
        def _wrap_hf_hub_download(*args, **kwargs):
            if 'use_auth_token' in kwargs and 'token' not in kwargs:
                tok = kwargs.pop('use_auth_token')
                if isinstance(tok, (str, bytes)):
                    kwargs['token'] = tok
                else:
                    # Ignore boolean/None legacy; rely on env/cached token
                    pass
            return _orig(*args, **kwargs)
        _h.hf_hub_download = _wrap_hf_hub_download  # type: ignore
    except Exception:
        pass
    # Patch snapshot_download as well
    try:
        _orig_s = _h.snapshot_download
        def _wrap_snapshot_download(*args, **kwargs):
            if 'use_auth_token' in kwargs and 'token' not in kwargs:
                tok = kwargs.pop('use_auth_token')
                if isinstance(tok, (str, bytes)):
                    kwargs['token'] = tok
            return _orig_s(*args, **kwargs)
        _h.snapshot_download = _wrap_snapshot_download  # type: ignore
    except Exception:
        pass

def _patch_pyannote_hf_token_kw():
    """Patch pyannote modules that captured hf_hub_download/snapshot_download
    to translate 'use_auth_token' -> 'token'.
    """
    try:
        import importlib
        import huggingface_hub as _h
    except Exception:
        return

    def _compat(func):
        def _wrap(*args, **kwargs):
            if 'use_auth_token' in kwargs and 'token' not in kwargs:
                tok = kwargs.pop('use_auth_token')
                if isinstance(tok, (str, bytes)):
                    kwargs['token'] = tok
            return func(*args, **kwargs)
        return _wrap

    modules = [
        'pyannote.audio.core.pipeline',
        'pyannote.audio.core.model',
        'pyannote.audio.core.inference',
        'pyannote.audio.pipelines.utils.getter',
        'pyannote.audio.pipelines.speaker_diarization',
    ]
    for name in modules:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        try:
            if hasattr(m, 'hf_hub_download'):
                setattr(m, 'hf_hub_download', _compat(_h.hf_hub_download))
        except Exception:
            pass
        try:
            if hasattr(m, 'snapshot_download'):
                setattr(m, 'snapshot_download', _compat(_h.snapshot_download))
        except Exception:
            pass

# ===== Modern Skia Panel Rendering =====
def _id_color_map_global(tracks_list):
    ids = []
    for tr in tracks_list:
        ident = tr.get('identity', None)
        # Treat only real Person_* identities; leave tracks with identity None
        # out of the global color map so they won't appear in memory or panel.
        if isinstance(ident, str) and ident not in (None, 'None'):
            ids.append(_normalize_identity_prefix(ident))
    uniq = sorted(set(ids))
    colors = {}
    # Use a fixed palette of moderately saturated, perceptually distinct RGB
    # colors (Tableau-style), cycling when there are more identities than
    # palette entries. This avoids both hue clustering和过高饱和度.
    base_palette = [
        (78, 121, 167),   # blue
        (242, 142, 43),   # orange
        (225, 87, 89),    # red
        (118, 183, 178),  # teal
        (89, 161, 79),    # green
        (237, 201, 72),   # yellow
        (176, 122, 161),  # purple
        (255, 157, 167),  # pink
    ]
    n_palette = len(base_palette) or 1
    for idx, ident in enumerate(uniq):
        colors[ident] = base_palette[idx % n_palette]
    # RGB tuples for Skia/panel
    return colors

def _ffprobe_video_props(path):
    import json, subprocess
    cmd = [
        'ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,duration',
        '-of','json', path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    js = json.loads(out)
    st = (js.get('streams') or [{}])[0]
    w = int(st.get('width', 0))
    h = int(st.get('height', 0))
    fr = st.get('avg_frame_rate', '0/1')
    try:
        if '/' in fr:
            a,b = fr.split('/')
            fps = float(a)/float(b) if float(b) != 0 else 0.0
        else:
            fps = float(fr)
    except Exception:
        fps = 0.0
    try:
        dur = float(st.get('duration', 0.0))
    except Exception:
        # duration can be at format level; fallback probe
        out2 = subprocess.check_output([
            'ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1', path
        ], stderr=subprocess.STDOUT).decode('utf-8').strip()
        try:
            dur = float(out2)
        except Exception:
            dur = 0.0
    if w <= 0 or h <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video props for {path}: {w}x{h} @ {fps}")
    return w, h, fps, max(0.0, dur)

def _collect_identity_avatar_targets(tracks_list):
    """Pick one representative (frame, x, y, s) per identity from tracks.
    Returns dict ident -> (frame_index, x, y, s).
    """
    targets = {}
    for tr in tracks_list:
        ident = _normalize_identity_prefix(tr.get('identity'))
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        if ident in targets:
            continue
        frames_arr = tr.get('track', {}).get('frame')
        proc = tr.get('proc_track', {})
        if frames_arr is None or not isinstance(proc, dict):
            continue
        frames = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        xs = proc.get('x', []); ys = proc.get('y', []); ss = proc.get('s', [])
        if not frames or not xs or not ys or not ss:
            continue
        mid = len(frames)//2
        try:
            targets[ident] = (int(frames[mid]), float(xs[mid]), float(ys[mid]), float(ss[mid]))
        except Exception:
            continue
    return targets

def _build_identity_avatars_pyav(video_path, tracks_list, max_edge=96):
    """Decode required frames via PyAV and crop avatar ROIs as numpy RGB arrays.
    Returns dict ident -> np.ndarray(H,W,3) RGB.
    """
    try:
        import av  # PyAV
    except Exception as e:
        raise RuntimeError('PyAV is required to decode frames for panel avatars') from e

    targets = _collect_identity_avatar_targets(tracks_list)
    if not targets:
        return {}
    # Build reverse map: frame_index -> list[(ident, x,y,s)]
    by_frame = {}
    for ident, (fi, x, y, s) in targets.items():
        by_frame.setdefault(int(fi), []).append((ident, float(x), float(y), float(s)))
    needed = set(by_frame.keys())

    try:
        container = av.open(video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open video for avatars: {video_path}") from e

    thumbs = {}
    fidx = -1
    try:
        for frame in container.decode(video=0):
            fidx += 1
            if fidx not in needed:
                continue
            img = frame.to_ndarray(format='rgb24')  # H,W,3 RGB
            H, W = img.shape[:2]
            for (ident, x, y, s) in by_frame[fidx]:
                x1 = max(0, int(x - s)); y1 = max(0, int(y - s))
                x2 = min(W, int(x + s)); y2 = min(H, int(y + s))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = img[y1:y2, x1:x2].copy()
                # Optional: downscale if very large to save memory; final scale at draw time
                mh, mw = roi.shape[:2]
                if max(mh, mw) > 512:
                    # simple stride-based reduce to avoid heavy deps
                    step = int(max(2, round(max(mh, mw)/256)))
                    roi = roi[::step, ::step]
                thumbs[ident] = roi
            if len(thumbs) == len(targets):
                break
    finally:
        container.close()
    return thumbs

def _wrap_for_panel(text: str, text_w_px: int, font_size_px: float, max_lines: int = 4) -> list:
    # Estimate per-char width; CJK ~ 1.0*font_size, Latin ~0.58*font_size
    t = (text or '').strip()
    if not t:
        return []
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in t)
    if has_cjk:
        max_chars = max(1, int(text_w_px / max(1.0, font_size_px * 1.02)))
    else:
        max_chars = max(1, int(text_w_px / max(1.0, font_size_px * 0.58)))
    wrapped = _wrap_text_for_ass(t, max_chars_cn=max_chars, max_chars_lat=max_chars)
    lines = [ln for ln in wrapped.split('\\N') if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith('…'):
            lines[-1] += '…'
    return lines

def _render_panel_chunk_worker(args_pack):
    """Worker: render panel frames [start_idx, end_idx) to a chunk mp4.

    args_pack = (
        start_idx, end_idx, panel_w, panel_h, fps_eff,
        font_path, msgs, id_colors, mem_idents,
        avatars_cache_path, max_items, chunk_out,
    )
    """
    import os, math, pickle, subprocess
    import numpy as _np
    try:
        import skia  # type: ignore
    except Exception as e:
        raise RuntimeError('skia-python is required for panel rendering (worker)') from e

    (start_idx, end_idx, panel_w, panel_h, fps_eff,
     font_path, msgs, id_colors, mem_idents,
     avatars_cache_path, max_items, chunk_out) = args_pack

    typeface = skia.Typeface.MakeFromFile(font_path)
    if typeface is None:
        raise RuntimeError(f'Failed to load typeface from {font_path}')
    font_title = skia.Font(typeface, 20)
    font_text = skia.Font(typeface, 18)

    # Load avatars cache
    with open(avatars_cache_path, 'rb') as f:
        avatars = pickle.load(f)

    # Colors and layout
    title_color = skia.ColorSetARGB(255, 230, 230, 235)
    text_color = skia.ColorSetARGB(255, 220, 220, 220)
    time_color = skia.ColorSetARGB(255, 170, 170, 180)
    card_bg = skia.ColorSetARGB(180, 28, 28, 32)
    card_active = skia.ColorSetARGB(210, 40, 40, 50)
    border_color = skia.ColorSetARGB(200, 70, 70, 80)
    glow_color = skia.ColorSetARGB(160, 50, 120, 255)
    bg_grad_c1 = skia.ColorSetARGB(255, 16, 16, 20)
    bg_grad_c2 = skia.ColorSetARGB(255, 6, 6, 8)

    pad = 16
    card_gap = 10
    avatar_r = 26
    avatar_d = avatar_r * 2
    ring_th = 3
    text_gap = 10

    info = skia.ImageInfo.Make(panel_w, panel_h, skia.ColorType.kRGBA_8888_ColorType, skia.AlphaType.kUnpremul_AlphaType)
    buf = _np.empty((panel_h, panel_w, 4), dtype=_np.uint8)

    # ffmpeg pipe for chunk encoding (keep params identical across chunks)
    cmd = [
        _FFMPEG_BIN,'-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s', f'{panel_w}x{panel_h}','-r', f'{fps_eff}',
        '-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',
        '-crf','20', chunk_out,
        '-loglevel','error'
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def ease_out_cubic(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 1.0 - (1.0 - x) ** 3

    def draw_frame(canvas: 'skia.Canvas', t: float):
        # Background gradient
        shader = skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, panel_h)],
            [bg_grad_c1, bg_grad_c2],
            [0.0, 1.0],
            skia.TileMode.kClamp
        )
        p_bg = skia.Paint(AntiAlias=True)
        p_bg.setShader(shader)
        canvas.drawRect(skia.Rect.MakeWH(panel_w, panel_h), p_bg)

        # Memory bank (top area)
        mem_pad = 12
        avatar_sz = 42
        cols = max(1, min(6, panel_w // (avatar_sz + mem_pad)))
        rows = (len(mem_idents) + cols - 1) // cols if mem_idents else 0
        mem_h = 0
        p_title = skia.Paint(AntiAlias=True, Color=title_color)
        if rows > 0:
            title_h = int(font_title.getSize() * 1.2)
            grid_h = rows * avatar_sz + (rows - 1) * mem_pad
            mem_h = mem_pad + title_h + 6 + grid_h + mem_pad
            canvas.drawString('Memory', mem_pad, mem_pad + font_title.getSize(), font_title, p_title)
            start_y = mem_pad + title_h + 6
            for idx, ident in enumerate(mem_idents):
                r = idx // cols
                c = idx % cols
                cx = int(mem_pad + c * (avatar_sz + mem_pad) + avatar_sz / 2)
                cy = int(start_y + r * (avatar_sz + mem_pad) + avatar_sz / 2)
                # ring
                ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
                rgb = id_colors.get(ident, (200, 200, 200))
                ring.setColor(skia.ColorSetRGB(*rgb))
                ring.setStrokeWidth(3)
                canvas.drawCircle(cx, cy, avatar_sz / 2 + 2, ring)
                # avatar
                ava = avatars.get(ident)
                if isinstance(ava, _np.ndarray) and ava.size > 0:
                    h0, w0 = ava.shape[:2]
                    arr_rgba = _np.concatenate([ava, _np.full((h0, w0, 1), 255, dtype=_np.uint8)], axis=-1)
                    sk_img = skia.Image.fromarray(arr_rgba)
                    if sk_img is not None:
                        canvas.save()
                        path = skia.Path(); path.addCircle(cx, cy, avatar_sz/2)
                        canvas.clipPath(path, doAntiAlias=True)
                        dst = skia.Rect.MakeXYWH(cx - avatar_sz/2, cy - avatar_sz/2, avatar_sz, avatar_sz)
                        canvas.drawImageRect(sk_img, dst, skia.SamplingOptions(skia.FilterMode.kLinear))
                        canvas.restore()

        # visible messages up to t
        vis = [m for m in msgs if m['start'] <= t]
        if not vis:
            return
        vis = vis[-max_items:]

        # Layout metrics
        line_h = int(font_text.getSize() * 1.3)
        title_h = int(font_title.getSize() * 1.2)
        card_pad_v = 12
        card_pad_h = 12
        heights = []
        for m in vis:
            text_h = title_h + (len(m['lines']) * line_h)
            h = max(avatar_d, text_h) + card_pad_v*2
            heights.append(h)
        total_h = sum(heights) + card_gap * (len(heights) - 1)
        y_top = mem_h + pad
        chat_rect = skia.Rect.MakeLTRB(0, y_top, panel_w, panel_h - pad)
        y_base = panel_h - pad - total_h
        if y_base < y_top:
            y_base = y_top
        # Slide-up animation for newest
        slide_dur = 0.25
        s = 0.0
        if len(vis) >= 1:
            newest = vis[-1]
            h_new = heights[-1]
            p = ease_out_cubic((t - float(newest['start'])) / slide_dur) if t >= float(newest['start']) else 0.0
            slack = max(0.0, (panel_h - pad) - (y_base + total_h))
            want = float(h_new + (card_gap if len(vis) > 1 else 0))
            s = (1.0 - p) * min(want, slack)
        canvas.save()
        canvas.clipRect(chat_rect, doAntiAlias=True)
        y = y_base - s

        p_fill = skia.Paint(AntiAlias=True)
        p_stroke = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        p_text = skia.Paint(AntiAlias=True, Color=text_color)
        p_title2 = skia.Paint(AntiAlias=True, Color=title_color)

        for i, m in enumerate(vis):
            h = heights[i]
            appear = ease_out_cubic((t - float(m['start'])) / 0.18)
            x_offset = int((1.0 - appear) * 30)
            alpha_scale = appear
            active = (m['start'] <= t <= m['end'])
            bg_col = card_active if active else card_bg
            rect = skia.Rect.MakeXYWH(pad + x_offset, y, panel_w - pad*2, h)
            rrect = skia.RRect.MakeRectXY(rect, 14, 14)
            # Shadow
            shadow = skia.Paint(AntiAlias=True)
            shadow.setColor(skia.ColorSetARGB(int(70*alpha_scale), 0, 0, 0))
            try:
                drop = skia.ImageFilters.DropShadow(0, 3, 10, 10, skia.ColorSetARGB(int(140*alpha_scale), 0, 0, 0))
                shadow.setImageFilter(drop)
            except Exception:
                pass
            canvas.drawRRect(rrect, shadow)
            # Fill
            p_fill.setColor(bg_col)
            p_fill.setAlphaf(float(alpha_scale))
            canvas.drawRRect(rrect, p_fill)
            # Border
            p_stroke.setColor(border_color)
            p_stroke.setStrokeWidth(1.5)
            p_stroke.setAlphaf(0.8 * float(alpha_scale))
            canvas.drawRRect(rrect, p_stroke)
            # Glow
            if active:
                pulse = 0.5 + 0.5 * math.sin((t - m['start']) * 6.28 * 0.8)
                gpaint = skia.Paint(AntiAlias=True)
                gpaint.setColor(glow_color)
                gpaint.setAlphaf(0.25 + 0.35 * pulse)
                canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(rect.left(), rect.top(), 6, rect.height()), 3, 3), gpaint)
            # Avatar ring
            cx = rect.left() + card_pad_h + avatar_r
            cy = rect.top() + card_pad_v + avatar_r
            rgb = id_colors.get(m['identity'], (200, 200, 200))
            ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
            ring.setColor(skia.ColorSetRGB(*rgb))
            ring.setStrokeWidth(ring_th)
            ring.setAlphaf(float(alpha_scale))
            canvas.drawCircle(cx, cy, avatar_r + ring_th*0.5, ring)
            # Avatar image
            ava_img = avatars.get(m['identity'])
            if isinstance(ava_img, _np.ndarray) and ava_img.size > 0:
                h0, w0 = ava_img.shape[:2]
                arr_rgba = _np.concatenate([ava_img, _np.full((h0, w0, 1), 255, dtype=_np.uint8)], axis=-1)
                sk_img = skia.Image.fromarray(arr_rgba)
                if sk_img is not None:
                    canvas.save()
                    path = skia.Path(); path.addCircle(cx, cy, avatar_r)
                    canvas.clipPath(path, doAntiAlias=True)
                    dst = skia.Rect.MakeXYWH(cx - avatar_r, cy - avatar_r, avatar_d, avatar_d)
                    canvas.drawImageRect(sk_img, dst, skia.SamplingOptions(skia.FilterMode.kLinear))
                    canvas.restore()
            # Title
            ts_min = int(m['start'] // 60)
            ts_sec = int(m['start'] % 60)
            title = f"{m['identity']}  {ts_min:02d}:{ts_sec:02d}"
            tx = rect.left() + card_pad_h + avatar_d + text_gap
            ty = rect.top() + card_pad_v + font_title.getSize()
            p_title2.setAlphaf(float(alpha_scale))
            canvas.drawString(title, tx, ty, font_title, p_title2)
            # Text lines
            p_text.setAlphaf(float(alpha_scale))
            for li, line in enumerate(m['lines']):
                ly = ty + 6 + (li + 1) * line_h
                canvas.drawString(line, tx, ly, font_text, p_text)
            y += h + card_gap
        canvas.restore()

    try:
        for i in range(int(start_idx), int(end_idx)):
            t = i / float(fps_eff)
            surface = skia.Surface(panel_w, panel_h)
            canvas = surface.getCanvas()
            draw_frame(canvas, t)
            row_bytes = int(buf.strides[0])
            ok = surface.readPixels(info, buf, row_bytes)
            if not ok:
                raise RuntimeError('Failed to read Skia surface pixels')
            frame_rgb = buf[:, :, :3]
            proc.stdin.write(frame_rgb.tobytes())
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    return chunk_out

def _render_side_panel_skia_parallel(base_video_path: str,
                                     diarization_results: list,
                                     tracks: list,
                                     output_panel_path: str,
                                     width_ratio: float,
                                     theme: str,
                                     max_items: int,
                                     W: int, H: int, fps_eff: float, dur: float):
    import os, math, subprocess, pickle
    import torch.multiprocessing as mp
    # Dimensions
    panel_w = max(64, int(round(float(W) * float(width_ratio))))
    if panel_w % 2 != 0:
        panel_w += 1
    panel_h = int(H)
    total_frames = int(math.floor(dur * fps_eff)) if dur and dur > 0 else None
    if total_frames is None:
        raise RuntimeError('Cannot determine total frames for panel rendering')

    # Fonts
    fonts_dir_abs, _ = _ensure_chinese_font()
    font_path = os.path.join(fonts_dir_abs, 'NotoSansCJKsc-Regular.otf')
    if not os.path.isfile(font_path):
        cands = [f for f in os.listdir(fonts_dir_abs) if f.lower().endswith(('.otf', '.ttf'))]
        if not cands:
            raise RuntimeError('No suitable font file found for Skia panel')
        font_path = os.path.join(fonts_dir_abs, cands[0])

    # Colors and memory identities
    id_colors = _id_color_map_global(tracks)
    # Only keep identities that have actually spoken in diarization_results
    spoken_idents = []
    for seg in (diarization_results or []):
        ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        spoken_idents.append(ident)
    spoken_set = set(spoken_idents)

    mem_idents = []
    seen = set()
    for tr in tracks:
        ident = _normalize_identity_prefix(tr.get('identity'))
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        if ident not in spoken_set:
            continue
        if ident in seen:
            continue
        seen.add(ident)
        mem_idents.append(ident)
    mem_idents.sort()

    # Avatars cache (strict)
    avatars_cache_path = os.path.join(args.pyworkPath, 'identity_avatars_magface.pckl')
    if not os.path.isfile(avatars_cache_path):
        raise RuntimeError(f"Missing identity avatars cache: {avatars_cache_path}. Run clustering to generate avatars.")
    import pickle as _pickle
    try:
        with open(avatars_cache_path, 'rb') as _f:
            _avatars_map = _pickle.load(_f)
    except Exception as _e:
        raise RuntimeError(f"Failed to load avatars cache: {avatars_cache_path}") from _e
    missing = [ident for ident in set(mem_idents) if ident not in _avatars_map]
    if missing:
        raise RuntimeError(f"Avatars missing for identities: {missing}. Re-run clustering to regenerate avatars cache.")

    # Messages (pre-wrap lines)
    pad = 16
    avatar_r = 26
    avatar_d = avatar_r * 2
    text_gap = 10
    text_w = panel_w - pad*2 - avatar_d - text_gap
    msgs = []
    for seg in (diarization_results or []):
        ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        txt = str(seg.get('text', '')).strip()
        msgs.append({'identity': ident, 'start': s, 'end': e, 'text': txt})
    msgs.sort(key=lambda m: (m['start'], m['end']))
    # font_text size is 18 in worker; use same numeric size here for wrapping
    for m in msgs:
        m['lines'] = _wrap_for_panel(m['text'], text_w, 18, max_lines=4)

    # Split into chunks
    n_workers = max(1, int(getattr(args, 'panelWorkers', 1)))
    n_workers = min(n_workers, total_frames) if total_frames > 0 else 1
    chunk = (total_frames + n_workers - 1) // n_workers
    ranges = []
    for w in range(n_workers):
        s = w * chunk
        e = min(total_frames, (w + 1) * chunk)
        if e > s:
            ranges.append((s, e))
    if not ranges:
        raise RuntimeError('No frame ranges computed for panel rendering')

    # Prepare chunk outputs
    out_dir = os.path.dirname(os.path.abspath(output_panel_path))
    os.makedirs(out_dir, exist_ok=True)
    packs = []
    chunk_paths = []
    for idx, (s, e) in enumerate(ranges):
        chunk_out = f"{output_panel_path}.part{idx:03d}.mp4"
        try:
            os.remove(chunk_out)
        except Exception:
            pass
        pack = (s, e, panel_w, panel_h, fps_eff, font_path, msgs, id_colors, mem_idents, avatars_cache_path, max_items, chunk_out)
        packs.append(pack)
        chunk_paths.append(chunk_out)

    # Run workers (spawn)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(packs)) as pool:
        res = pool.map(_render_panel_chunk_worker, packs)
    # Verify
    for p in chunk_paths:
        if (p not in res) or (not os.path.isfile(p)) or (os.path.getsize(p) <= 0):
            raise RuntimeError(f'Panel chunk missing or empty: {p}')

    # Concat chunks
    list_file = f"{output_panel_path}.concat.txt"
    with open(list_file, 'w') as f:
        for p in chunk_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd_concat = [
        _FFMPEG_BIN,'-y','-f','concat','-safe','0','-i', list_file,
        '-c','copy', output_panel_path,
        '-loglevel','error'
    ]
    rc = subprocess.call(cmd_concat)
    if rc != 0:
        raise RuntimeError('Failed to concatenate panel chunks')
    # Cleanup
    try:
        os.remove(list_file)
        for p in chunk_paths:
            os.remove(p)
    except Exception:
        pass
    return output_panel_path

def render_side_panel_skia(base_video_path: str,
                           diarization_results: list,
                           tracks: list,
                           output_panel_path: str,
                           width_ratio: float = 0.28,
                           theme: str = 'glass',
                           max_items: int = 6):
    """Render a right-side animated chat-like panel using Skia and compose with ffmpeg.
    Produces a standalone panel video at output_panel_path (no audio).
    """
    import os, subprocess, math
    import numpy as _np
    try:
        import skia  # type: ignore
    except Exception as e:
        raise RuntimeError('skia-python is required for panel rendering') from e

    # Probe base video
    W, H, fps_eff, dur = _ffprobe_video_props(base_video_path)
    # Parallel path: split frames across workers and concat
    try:
        n_workers = int(getattr(args, 'panelWorkers', 1))
    except Exception:
        n_workers = 1
    if n_workers and n_workers > 1:
        _render_side_panel_skia_parallel(
            base_video_path,
            diarization_results,
            tracks,
            output_panel_path,
            width_ratio,
            theme,
            max_items,
            W, H, fps_eff, dur,
        )
        return
    panel_w = max(64, int(round(float(W) * float(width_ratio))))
    if panel_w % 2 != 0:
        panel_w += 1
    panel_h = H
    total_frames = int(math.floor(dur * fps_eff)) if dur > 0 else None

    # Fonts
    fonts_dir_abs, font_name = _ensure_chinese_font()
    # Try to load bundled Noto font file
    font_path = os.path.join(fonts_dir_abs, 'NotoSansCJKsc-Regular.otf')
    if not os.path.isfile(font_path):
        # Fallback to any OTF in fonts_dir_abs
        cands = [f for f in os.listdir(fonts_dir_abs) if f.lower().endswith(('.otf', '.ttf'))]
        if not cands:
            raise RuntimeError('No suitable font file found for Skia panel')
        font_path = os.path.join(fonts_dir_abs, cands[0])
    typeface = skia.Typeface.MakeFromFile(font_path)
    if typeface is None:
        raise RuntimeError(f'Failed to load typeface from {font_path}')
    font_title = skia.Font(typeface, 20)
    font_text = skia.Font(typeface, 18)

    # Identity colors and avatars (quality-aware via MagFace)
    id_colors = _id_color_map_global(tracks)
    def _build_identity_avatars_magface(video_path: str, tracks_list: list, max_per_ident: int = 16):
        try:
            # Deferred import to avoid heavy init when not needed
            from .embedders.magface_embedder import MagFaceEmbedder
        except Exception:
            from embedders.magface_embedder import MagFaceEmbedder
        import av
        import numpy as _np
        import cv2 as _cv

        # Build candidates: ident -> list[(frame_idx, x, y, s)] sampled across its tracks
        cand = {}
        for tr in tracks_list:
            ident = _normalize_identity_prefix(tr.get('identity'))
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            proc = tr.get('proc_track', {})
            frames = tr.get('track', {}).get('frame')
            if frames is None or not isinstance(proc, dict):
                continue
            xs = proc.get('x', []); ys = proc.get('y', []); ss = proc.get('s', [])
            fl = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            if not fl or not xs or not ys or not ss:
                continue
            n = len(fl)
            step = max(1, n // max_per_ident)
            for k in range(0, n, step):
                if len(cand.setdefault(ident, [])) >= max_per_ident:
                    break
                f = int(fl[k])
                cand[ident].append((f, float(xs[k]), float(ys[k]), float(ss[k])))

        if not cand:
            return {}

        # Decode needed frames using PyAV as BGR
        needed = {}
        for ident, lst in cand.items():
            for (fi, x, y, s) in lst:
                needed.setdefault(int(fi), []).append((ident, x, y, s))

        try:
            container = av.open(video_path)
        except Exception as e:
            raise RuntimeError(f'Failed to open video for MagFace avatars: {video_path}') from e

        # Helper: crop ROI around (x,y,s) with same logic as visualization
        def _crop_bgr(img_bgr: _np.ndarray, x: float, y: float, s: float, cs: float) -> _np.ndarray:
            H, W = img_bgr.shape[:2]
            bsi = int(s * (1 + 2 * cs))
            pad = 110
            fr = _np.pad(img_bgr, ((bsi,bsi),(bsi,bsi),(0,0)), mode='constant', constant_values=pad)
            my = y + bsi; mx = x + bsi
            y1 = int(my - s); y2 = int(my + s * (1 + 2 * cs))
            x1 = int(mx - s * (1 + cs)); x2 = int(mx + s * (1 + cs))
            if y2 <= y1 or x2 <= x1:
                return None
            face = fr[y1:y2, x1:x2]
            if face.size == 0:
                return None
            return _cv.resize(face, (224,224))

        # Collect crops per identity
        crops = {k: [] for k in cand.keys()}
        try:
            fidx = -1
            for frame in container.decode(video=0):
                fidx += 1
                lst = needed.get(fidx)
                if not lst:
                    continue
                bgr = frame.to_ndarray(format='bgr24')
                for (ident, x, y, s) in lst:
                    roi = _crop_bgr(bgr, x, y, s, float(getattr(args, 'cropScale', 0.40)))
                    if roi is not None:
                        crops[ident].append(roi)
        finally:
            container.close()

        # Build aligned faces and select best by MagFace magnitude
        # Use the same batch sizing policy as identity embedding elsewhere
        try:
            _idb = int(getattr(args, 'idBatch', 64))
        except Exception:
            _idb = 64
        embedder = MagFaceEmbedder(device='cuda', batch_size=_idb, backbone=os.environ.get('MAGFACE_BACKBONE', 'iresnet100'))
        out = {}
        for ident, imgs in crops.items():
            if not imgs:
                continue
            tensors = []
            aligned_store = []
            for img in imgs:
                t = embedder._align_and_preprocess(img)
                if t is not None:
                    tensors.append(t)
                    aligned_store.append(img)  # keep original in case alignment fails for all
            if not tensors:
                continue
            batch = torch.stack(tensors, dim=0).to(embedder.device)
            with torch.no_grad():
                raw = embedder.model(batch)  # (N,512)
                mags = torch.norm(raw, p=2, dim=1)  # (N,)
            idx = int(torch.argmax(mags).item())
            # Reconstruct aligned RGB from preprocessed tensor
            best = tensors[idx].cpu().numpy()  # CHW float [0,1], RGB
            rgb = (best.transpose(1,2,0) * 255.0).clip(0,255).astype(_np.uint8)
            out[ident] = rgb
        return out

    # Memory identities: only identities that have actually spoken (i.e., appear
    # in diarization_results with a valid identity). This avoids showing silent
    # faces in the memory grid and reduces color collisions.
    spoken_idents = []
    for seg in (diarization_results or []):
        ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        spoken_idents.append(ident)
    spoken_set = set(spoken_idents)

    mem_idents = []
    seen = set()
    for tr in tracks:
        ident = _normalize_identity_prefix(tr.get('identity'))
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        if ident not in spoken_set:
            continue
        if ident in seen:
            continue
        seen.add(ident)
        mem_idents.append(ident)
    mem_idents.sort()

    # Strictly reuse cached high-quality avatars produced during clustering
    avatars_cache_path = os.path.join(args.pyworkPath, 'identity_avatars_magface.pckl')
    if not os.path.isfile(avatars_cache_path):
        raise RuntimeError(f"Missing identity avatars cache: {avatars_cache_path}. Run clustering to generate avatars.")
    try:
        import pickle
        with open(avatars_cache_path, 'rb') as f:
            avatars = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load avatars cache: {avatars_cache_path}") from e
    # Validate all required identities exist
    missing = [ident for ident in set(mem_idents) if ident not in avatars]
    if missing:
        raise RuntimeError(f"Avatars missing for identities: {missing}. Re-run clustering to regenerate avatars cache.")

    # Build messages timeline
    msgs = []
    for seg in (diarization_results or []):
        ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
        if not (isinstance(ident, str) and ident not in (None, 'None')):
            continue
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        txt = str(seg.get('text', '')).strip()
        msgs.append({'identity': ident, 'start': s, 'end': e, 'text': txt})
    msgs.sort(key=lambda m: (m['start'], m['end']))

    # ffmpeg pipe for panel encoding
    # Use a conservative H.264 encoding command that avoids newer options.
    cmd = [
        _FFMPEG_BIN, '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{panel_w}x{panel_h}',
        '-r', f'{fps_eff}',
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        output_panel_path,
        '-loglevel', 'error',
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Drawing helpers
    def ease_out_cubic(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 1.0 - (1.0 - x) ** 3

    # Layout constants
    pad = 16
    card_gap = 10
    avatar_r = 26
    avatar_d = avatar_r * 2
    ring_th = 3
    text_gap = 10
    title_color = skia.ColorSetARGB(255, 230, 230, 235)
    text_color = skia.ColorSetARGB(255, 220, 220, 220)
    time_color = skia.ColorSetARGB(255, 170, 170, 180)
    card_bg = skia.ColorSetARGB(180, 28, 28, 32)
    card_active = skia.ColorSetARGB(210, 40, 40, 50)
    border_color = skia.ColorSetARGB(200, 70, 70, 80)
    glow_color = skia.ColorSetARGB(160, 50, 120, 255)
    # Background
    bg_grad_c1 = skia.ColorSetARGB(255, 16, 16, 20)
    bg_grad_c2 = skia.ColorSetARGB(255, 6, 6, 8)

    # Precompute text area width for wrapping
    text_w = panel_w - pad*2 - avatar_d - text_gap

    # Pre-wrap message text to lines according to panel width
    for m in msgs:
        m['lines'] = _wrap_for_panel(m['text'], text_w, font_text.getSize(), max_lines=4)
        # Precompute title (identity + timestamp placeholder computed per-frame)

    # Frame loop
    info = skia.ImageInfo.Make(panel_w, panel_h, skia.ColorType.kRGBA_8888_ColorType, skia.AlphaType.kUnpremul_AlphaType)
    buf = _np.empty((panel_h, panel_w, 4), dtype=_np.uint8)

    # Pre-allocate paints
    p_fill = skia.Paint(AntiAlias=True)
    p_stroke = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
    p_text = skia.Paint(AntiAlias=True, Color=text_color)
    p_title = skia.Paint(AntiAlias=True, Color=title_color)
    p_time = skia.Paint(AntiAlias=True, Color=time_color)

    def draw_frame(canvas: 'skia.Canvas', t: float):
        # Background gradient
        shader = skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, panel_h)],
            [bg_grad_c1, bg_grad_c2],
            [0.0, 1.0],
            skia.TileMode.kClamp
        )
        p_bg = skia.Paint(AntiAlias=True)
        p_bg.setShader(shader)
        canvas.drawRect(skia.Rect.MakeWH(panel_w, panel_h), p_bg)
        # Memory bank (top area)
        mem_pad = 12
        avatar_sz = 42
        cols = max(1, min(6, panel_w // (avatar_sz + mem_pad)))
        rows = (len(mem_idents) + cols - 1) // cols if mem_idents else 0
        mem_h = 0
        if rows > 0:
            title_h = int(font_title.getSize() * 1.2)
            grid_h = rows * avatar_sz + (rows - 1) * mem_pad
            mem_h = mem_pad + title_h + 6 + grid_h + mem_pad
            # Title
            canvas.drawString('Memory', mem_pad, mem_pad + font_title.getSize(), font_title, p_title)
            # Grid
            start_y = mem_pad + title_h + 6
            for idx, ident in enumerate(mem_idents):
                r = idx // cols
                c = idx % cols
                cx = int(mem_pad + c * (avatar_sz + mem_pad) + avatar_sz / 2)
                cy = int(start_y + r * (avatar_sz + mem_pad) + avatar_sz / 2)
                # ring
                rgb = id_colors.get(ident, (200, 200, 200))
                ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
                ring.setColor(skia.ColorSetRGB(*rgb))
                ring.setStrokeWidth(3)
                canvas.drawCircle(cx, cy, avatar_sz / 2 + 2, ring)
                # avatar
                ava = avatars.get(ident)
                if isinstance(ava, _np.ndarray) and ava.size > 0:
                    h0, w0 = ava.shape[:2]
                    arr_rgba = _np.concatenate([ava, _np.full((h0, w0, 1), 255, dtype=_np.uint8)], axis=-1)
                    sk_img = skia.Image.fromarray(arr_rgba)
                    if sk_img is not None:
                        canvas.save()
                        path = skia.Path(); path.addCircle(cx, cy, avatar_sz/2)
                        canvas.clipPath(path, doAntiAlias=True)
                        dst = skia.Rect.MakeXYWH(cx - avatar_sz/2, cy - avatar_sz/2, avatar_sz, avatar_sz)
                        canvas.drawImageRect(sk_img, dst, skia.SamplingOptions(skia.FilterMode.kLinear))
                        canvas.restore()

        # Select visible messages up to time t (keep last max_items)
        vis = [m for m in msgs if m['start'] <= t]
        if not vis:
            return
        vis = vis[-max_items:]

        # Compute total height to bottom-stack cards
        # Each card height: max(avatar_d, title+lines*line_h) + padding*2
        line_h = int(font_text.getSize() * 1.3)
        title_h = int(font_title.getSize() * 1.2)
        card_pad_v = 12
        card_pad_h = 12
        card_rects = []
        heights = []
        for m in vis:
            text_h = title_h + (len(m['lines']) * line_h)
            h = max(avatar_d, text_h) + card_pad_v*2
            heights.append(h)
        total_h = sum(heights) + card_gap * (len(heights) - 1)
        # Chat viewport (clip) below memory block
        y_top = mem_h + pad
        chat_rect = skia.Rect.MakeLTRB(0, y_top, panel_w, panel_h - pad)
        # Base top position that bottom-aligns the stack within viewport (fully visible)
        y_base = panel_h - pad - total_h
        if y_base < y_top:
            y_base = y_top
        # Slide-up animation: move entire stack upward (never downward), so newest始终完整可见
        slide_dur = 0.25
        s = 0.0
        if len(vis) >= 1:
            newest = vis[-1]
            h_new = heights[-1]
            p = ease_out_cubic((t - float(newest['start'])) / slide_dur) if t >= float(newest['start']) else 0.0
            # available slack so upward shift won't clip within viewport
            slack = max(0.0, (panel_h - pad) - (y_base + total_h))
            want = float(h_new + (card_gap if len(vis) > 1 else 0))
            s = (1.0 - p) * min(want, slack)
        canvas.save()
        canvas.clipRect(chat_rect, doAntiAlias=True)
        y = y_base - s
        # Draw each card
        for i, m in enumerate(vis):
            h = heights[i]
            # Appear animation (x-slide + fade)
            appear = ease_out_cubic((t - float(m['start'])) / 0.18)
            x_offset = int((1.0 - appear) * 30)
            alpha_scale = appear
            # Active highlight
            active = (m['start'] <= t <= m['end'])
            bg_col = card_active if active else card_bg
            # Card bg with rounded rect
            rect = skia.Rect.MakeXYWH(pad + x_offset, y, panel_w - pad*2, h)
            rrect = skia.RRect.MakeRectXY(rect, 14, 14)
            # Shadow (use ImageFilters.DropShadow for compatibility)
            shadow = skia.Paint(AntiAlias=True)
            shadow.setColor(skia.ColorSetARGB(int(70*alpha_scale), 0, 0, 0))
            try:
                drop = skia.ImageFilters.DropShadow(0, 3, 10, 10, skia.ColorSetARGB(int(140*alpha_scale), 0, 0, 0))
                shadow.setImageFilter(drop)
            except Exception:
                # If ImageFilters unavailable, skip filter but keep solid shadow color
                pass
            canvas.drawRRect(rrect, shadow)
            # Fill
            p_fill.setColor(bg_col)
            p_fill.setAlphaf(float(alpha_scale))
            canvas.drawRRect(rrect, p_fill)
            # Border
            p_stroke.setColor(border_color)
            p_stroke.setStrokeWidth(1.5)
            p_stroke.setAlphaf(0.8 * float(alpha_scale))
            canvas.drawRRect(rrect, p_stroke)
            # Active glow accent bar
            if active:
                pulse = 0.5 + 0.5 * math.sin((t - m['start']) * 6.28 * 0.8)
                gpaint = skia.Paint(AntiAlias=True)
                gpaint.setColor(glow_color)
                gpaint.setAlphaf(0.25 + 0.35 * pulse)
                canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(rect.left(), rect.top(), 6, rect.height()), 3, 3), gpaint)

            # Avatar circle
            cx = rect.left() + card_pad_h + avatar_r
            cy = rect.top() + card_pad_v + avatar_r
            # Accent ring
            rgb = id_colors.get(m['identity'], (200, 200, 200))
            ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
            ring.setColor(skia.ColorSetRGB(*rgb))
            ring.setStrokeWidth(ring_th)
            ring.setAlphaf(float(alpha_scale))
            canvas.drawCircle(cx, cy, avatar_r + ring_th*0.5, ring)
            # Clip circle and draw avatar image
            ava_img = avatars.get(m['identity'])
            if isinstance(ava_img, _np.ndarray) and ava_img.size > 0:
                # Convert numpy RGB to Skia Image
                h0, w0 = ava_img.shape[:2]
                # Skia expects RGBA for fromarray; add alpha channel
                arr_rgba = _np.concatenate([ava_img, _np.full((h0, w0, 1), 255, dtype=_np.uint8)], axis=-1)
                sk_img = skia.Image.fromarray(arr_rgba)
                if sk_img is not None:
                    canvas.save()
                    path = skia.Path()
                    path.addCircle(cx, cy, avatar_r)
                    canvas.clipPath(path, doAntiAlias=True)
                    dst = skia.Rect.MakeXYWH(cx - avatar_r, cy - avatar_r, avatar_d, avatar_d)
                    canvas.drawImageRect(sk_img, dst, skia.SamplingOptions(skia.FilterMode.kLinear))
                    canvas.restore()

            # Title: identity + time
            ts_min = int(m['start'] // 60)
            ts_sec = int(m['start'] % 60)
            title = f"{m['identity']}  {ts_min:02d}:{ts_sec:02d}"
            tx = rect.left() + card_pad_h + avatar_d + text_gap
            ty = rect.top() + card_pad_v + font_title.getSize()
            p_title.setAlphaf(float(alpha_scale))
            canvas.drawString(title, tx, ty, font_title, p_title)

            # Text lines
            p_text.setAlphaf(float(alpha_scale))
            for li, line in enumerate(m['lines']):
                ly = ty + 6 + (li + 1) * line_h
                canvas.drawString(line, tx, ly, font_text, p_text)

            # next y
            y += h + card_gap
        canvas.restore()

    # Render frames
    try:
        i = 0
        while True:
            if total_frames is not None and i >= total_frames:
                break
            t = i / fps_eff
            surface = skia.Surface(panel_w, panel_h)
            canvas = surface.getCanvas()
            draw_frame(canvas, t)
            # Read pixels to numpy
            # Read pixels into numpy buffer (use buffer protocol, rowBytes = first stride)
            row_bytes = int(buf.strides[0])
            ok = surface.readPixels(info, buf, row_bytes)
            if not ok:
                raise RuntimeError('Failed to read Skia surface pixels')
            frame_rgb = buf[:, :, :3]
            proc.stdin.write(frame_rgb.tobytes())
            i += 1
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()


def generate_ass(diarization_results, output_ass_path, id_colors_map, font_name_override=None):
    # Build ASS header + events with inline color for Person_[ID]
    font_name = font_name_override or 'Noto Sans CJK SC'
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},42,&H00FFFFFF,&H000000FF,&H00202020,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,30,30,30,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = []
    # Collapse to one-line-at-a-time display to avoid multiple lines showing together
    collapsed = _collapse_to_single_line(diarization_results)
    for seg in collapsed:
        ident = _normalize_identity_prefix(seg.get('identity'))
        if not ident or ident == 'None':
            continue
        st = float(seg.get('start', 0.0))
        et = float(seg.get('end', st))
        if et <= st:
            continue
        raw_text = str(seg.get('text', ''))
        text_wrapped = _wrap_text_for_ass(raw_text)
        # Colorize only the Person_[ID] prefix to match box color; keep rest as default (white)
        color_rgb = id_colors_map.get(ident, id_colors_map.get(_normalize_identity_prefix(ident), (255,255,255)))
        # ASS expects BGR in &HBBGGRR; panel/memory map stores RGB
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        ass_hex = _bgr_to_ass_hex(color_bgr)
        # Build dialogue text: colored ident + reset to default for the rest
        # Use \N for line breaks from wrapper
        # Note: reset color with {\c&HFFFFFF&}
        prefix = f"{{\\c{ass_hex}}}{ident}{{\\c&HFFFFFF&}}: "
        ass_text = prefix + text_wrapped
        # Time in h:mm:ss.cs (ASS uses centiseconds)
        def _ass_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            cs = int(round((t - int(t)) * 100))
            return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
        lines.append(f"Dialogue: 0,{_ass_time(st)},{_ass_time(et)},Default,,0,0,0,,{ass_text}")
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(header + lines))

def generate_ass_seq(diarization_results, output_ass_path, id_colors_map, font_name_override=None):
    # Build ASS with single-line sequential display per segment (no simultaneous multi-line)
    font_name = font_name_override or 'Noto Sans CJK SC'
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},42,&H00FFFFFF,&H000000FF,&H00202020,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,30,30,30,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = []
    collapsed = _collapse_to_single_line(diarization_results)
    def _ass_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int(round((t - int(t)) * 100))
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
    for seg in collapsed:
        ident = _normalize_identity_prefix(seg.get('identity'))
        if not ident or ident == 'None':
            continue
        st = float(seg.get('start', 0.0))
        et = float(seg.get('end', st))
        if et <= st:
            continue
        raw_text = str(seg.get('text', ''))
        wrapped = _wrap_text_for_ass(raw_text)
        parts = [ln for ln in wrapped.split('\\N') if ln.strip()]
        if not parts:
            continue
        dur = max(0.0, et - st)
        if dur <= 0.0:
            continue
        step = dur / len(parts)
        color_rgb = id_colors_map.get(ident, id_colors_map.get(_normalize_identity_prefix(ident), (255,255,255)))
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        ass_hex = _bgr_to_ass_hex(color_bgr)
        for i, ln in enumerate(parts):
            a = st + i * step
            b = st + (i + 1) * step if i < len(parts) - 1 else et
            if b <= a:
                continue
            prefix = f"{{\\c{ass_hex}}}{ident}{{\\c&HFFFFFF&}}: "
            lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{prefix}{ln}")
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(header + lines))

def generate_ass_seq_wordtimed(diarization_results, output_ass_path, id_colors_map, font_name_override=None, words_list=None):
    # Build ASS using word-aligned timings per displayed line when words_list provided.
    # Render at right side; do not show Person_* prefix; colorize text by identity.
    font_name = font_name_override or 'Noto Sans CJK SC'
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},40,&H00FFFFFF,&H000000FF,&H00202020,&H00000000,0,0,0,0,100,100,0,0,1,2,0,6,20,60,30,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = []
    collapsed = _collapse_to_single_line(diarization_results)
    def _ass_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int(round((t - int(t)) * 100))
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
    for seg in collapsed:
        ident = _normalize_identity_prefix(seg.get('identity'))
        if not ident or ident == 'None':
            continue
        st = float(seg.get('start', 0.0))
        et = float(seg.get('end', st))
        if et <= st:
            continue
        raw_text = str(seg.get('text', ''))
        color = id_colors_map.get(ident, id_colors_map.get(_normalize_identity_prefix(ident), (255,255,255)))
        ass_hex = _bgr_to_ass_hex(color)
        # No Person_* prefix
        if isinstance(words_list, list) and words_list:
            # collect words overlapping this segment
            toks = []
            for (t0, t1, wtxt) in words_list:
                t0f = float(t0); t1f = float(t1)
                if t1f <= st:
                    continue
                if t0f >= et:
                    break
                a = max(st, t0f)
                b = min(et, t1f)
                if b > a and str(wtxt).strip():
                    toks.append((a, b, str(wtxt)))
            if not toks:
                wrapped = _wrap_text_for_ass(raw_text)
                if wrapped.strip():
                    lines.append(f"Dialogue: 0,{_ass_time(st)},{_ass_time(et)},Default,,0,0,0,,{{\\c{ass_hex}}}{wrapped}")
                continue
            # detect CJK
            has_cjk = any('\u4e00' <= ch <= '\u9fff' for _,_,w in toks for ch in w)
            maxw = 18 if has_cjk else 28
            cur = []
            cur_len = 0
            cur_t0 = None
            cur_t1 = None
            def flush_line():
                nonlocal cur, cur_len, cur_t0, cur_t1
                if not cur:
                    return
                a = cur_t0; b = cur_t1
                if a is None or b is None or b <= a:
                    cur = []; cur_len = 0; cur_t0 = None; cur_t1 = None; return
                if has_cjk:
                    text_line = ''.join(w for _,_,w in cur)
                else:
                    text_line = ' '.join(w for _,_,w in cur)
                text_line = _wrap_text_for_ass(text_line)
                if text_line.strip():
                    lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{{\\c{ass_hex}}}{text_line}")
                cur = []; cur_len = 0; cur_t0 = None; cur_t1 = None
            for (a, b, w) in toks:
                wlen = len(w)
                if cur_len > 0 and (cur_len + (0 if has_cjk else 1) + wlen) > maxw:
                    flush_line()
                if not cur:
                    cur_t0 = a
                cur_t1 = b
                cur.append((a, b, w))
                cur_len += (wlen if has_cjk else (wlen if cur_len == 0 else (1 + wlen)))
            flush_line()
        else:
            # fallback: equal slicing
            wrapped = _wrap_text_for_ass(raw_text)
            parts = [ln for ln in wrapped.split('\\N') if ln.strip()]
            if not parts:
                continue
            dur = max(0.0, et - st)
            if dur <= 0.0:
                continue
            step = dur / len(parts)
            for i, ln in enumerate(parts):
                a = st + i * step
                b = st + (i + 1) * step if i < len(parts) - 1 else et
                if b <= a:
                    continue
                lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{{\\c{ass_hex}}}{ln}")
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(header + lines))

def visualization(tracks, scores, diarization_results, args, words_list=None):
    # Build per-frame overlays without using pyframes. This visualization is
    # tied to the track/proc_track pipeline; when detBackend == 'sam3', it
    # additionally uses SAM3 segmentation masks for overlays.
    backend = str(getattr(args, 'detBackend', 'sam3')).lower()
    use_sam3_masks = backend == 'sam3'
    sam3_masks = None
    if use_sam3_masks:
        masks_path = os.path.join(args.pyworkPath, 'sam3_masks.pckl')
        if not os.path.exists(masks_path):
            raise RuntimeError(f"SAM3 masks file not found at {masks_path} while detBackend=='sam3'")
        with open(masks_path, 'rb') as f:
            sam3_masks = pickle.load(f)

    faces_by_frame = defaultdict(list)
    for tidx, track in enumerate(tracks):
        if tidx >= len(scores):
            continue
        identity = track.get('identity', 'None')
        score = scores[tidx]
        frames_arr = track['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for lidx, f in enumerate(frames_list):
            s_window = score[max(lidx - 2, 0): min(lidx + 3, len(score) - 1)]
            s_val = float(np.mean(s_window)) if len(s_window) > 0 else 0.0
            faces_by_frame[int(f)].append({
                'track': tidx,
                'score': s_val,
                'identity': identity,
                's': track['proc_track']['s'][lidx],
                'x': track['proc_track']['x'][lidx],
                'y': track['proc_track']['y'][lidx],
            })

    # Determine frame size from video
    cap0 = cv2.VideoCapture(args.videoFilePath)
    if not cap0.isOpened():
        raise RuntimeError(f"Failed to open video for visualization: {args.videoFilePath}")
    ret, first = cap0.read()
    if not ret:
        cap0.release()
        raise RuntimeError("Failed to decode any frame for visualization")
    fw, fh = first.shape[1], first.shape[0]
    cap0.release()

    if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot render visualization with correct timing.")

    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, 'video_only.avi'),
        cv2.VideoWriter_fourcc(*'XVID'),
        float(args.videoFps),
        (fw, fh),
        True,
    )

    # Stable color map (per-identity) used for rectangles/masks in visualization.
    # Use a fixed high-contrast palette and only assign colors to identities
    # that have actually spoken, so speakers stay visually distinct.
    def _id_color_map(tracks_list, spoken_set=None):
        ids = []
        for tr in tracks_list:
            ident = tr.get('identity', None)
            if ident is None or ident == 'None':
                continue
            if spoken_set is not None and ident not in spoken_set:
                continue
            ids.append(ident)
        uniq = sorted(set(ids))
        colors = {}
        # Tableau-like muted palette in BGR (to avoid overly saturated colors)
        base_palette_bgr = [
            (167, 121, 78),    # blue (BGR of 78,121,167)
            (43, 142, 242),    # orange
            (89, 87, 225),     # red
            (178, 183, 118),   # teal
            (79, 161, 89),     # green
            (72, 201, 237),    # yellow
            (161, 122, 176),   # purple
            (167, 157, 255),   # pink
        ]
        n_palette = len(base_palette_bgr) or 1
        for idx, ident in enumerate(uniq):
            colors[ident] = base_palette_bgr[idx % n_palette]
        return colors

    # Speech bubble helpers (ensure defined in this visualization scope)
    segs_by_ident = defaultdict(list)
    if isinstance(diarization_results, list):
        for seg in diarization_results:
            ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            s = float(seg.get('start', seg.get('start_time', 0.0)))
            e = float(seg.get('end', seg.get('end_time', s)))
            if e <= s:
                continue
            txt = str(seg.get('text', ''))
            segs_by_ident[ident].append((s, e, txt))
        for k in segs_by_ident:
            segs_by_ident[k].sort(key=lambda x: (x[0], x[1]))

    def _bubble_text_for(ident: str, t_sec: float) -> str:
        lst = segs_by_ident.get(ident, [])
        for (s, e, txt) in lst:
            if s <= t_sec <= e:
                return txt
        return ''

    def _wrap_lines(txt: str, max_cn: int = 16, max_lat: int = 24, max_lines: int = 3):
        wrapped = _wrap_text_for_ass(txt, max_cn, max_lat)
        lines = [ln for ln in wrapped.split('\\N') if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines and not lines[-1].endswith('…'):
                lines[-1] = lines[-1] + '…'
        return lines

    def _draw_rounded_rect(img, x, y, w, h, radius, fill_color, border_color, border_th=2, alpha=0.9):
        x = int(x); y = int(y); w = int(w); h = int(h); radius = int(max(2, radius))
        overlay = img.copy()
        cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), fill_color, thickness=-1)
        cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.circle(overlay, (x+w-radius, y+h-radius), radius, fill_color, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.rectangle(img, (x+radius, y), (x+w-radius, y+h), border_color, thickness=border_th)
        cv2.rectangle(img, (x, y+radius), (x+w, y+h-radius), border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+radius, y+h-radius), radius, border_color, thickness=border_th)
        cv2.circle(img, (x+w-radius, y+h-radius), radius, border_color, thickness=border_th)

    def _draw_tail(img, base_x, base_y, to_left: bool, fill_color, border_color, tail_len=14, tail_half=8, alpha=0.9, border_th=2):
        base_x = int(base_x); base_y = int(base_y)
        if to_left:
            pts = np.array([[base_x, base_y], [base_x+tail_len, base_y-tail_half], [base_x+tail_len, base_y+tail_half]], dtype=np.int32)
        else:
            pts = np.array([[base_x, base_y], [base_x-tail_len, base_y-tail_half], [base_x-tail_len, base_y+tail_half]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=border_th)

    # Build global identity thumbnails: one per Person_* for the whole video (no duplicates)
    tile_w = max(1, min(100, fw // 12))
    tile_h = tile_w
    margin = 6
    label_height = 28

    def _build_identity_thumbs(video_path, tracks_list, scores_list):
        # Choose a representative frame per identity: prefer max ASD score; otherwise use center frame
        id_to_repr = {}
        for i, tr in enumerate(tracks_list):
            ident = tr.get('identity', None)
            if not (isinstance(ident, str) and ident not in (None, 'None')):
                continue
            frames = tr.get('track', {}).get('frame')
            proc = tr.get('proc_track', {})
            if frames is None or not isinstance(proc, dict):
                continue
            fr = frames.tolist() if hasattr(frames, 'tolist') else list(frames)
            if not fr:
                continue
            xs = proc.get('x', []); ys = proc.get('y', []); ss = proc.get('s', [])
            if not xs or not ys or not ss:
                continue
            sc = scores_list[i] if i < len(scores_list) else []
            best = None  # (score, global_frame, local_idx)
            try:
                import numpy as _np
                sc_arr = _np.asarray(sc, dtype=float)
                T = min(len(fr), int(sc_arr.shape[0]))
                if T > 0:
                    j = int(sc_arr[:T].argmax())
                    best = (float(sc_arr[j]), int(fr[j]), j)
            except Exception:
                T = 0
            if best is None:
                j = int(len(fr) // 2)
                best = (float('-inf'), int(fr[j]), j)
            # Keep the highest-score representative per identity
            cur = id_to_repr.get(ident)
            if cur is None or best[0] > cur[0]:
                # Clamp idx to proc lengths
                jj = best[2]
                jj = min(jj, len(xs) - 1, len(ys) - 1, len(ss) - 1)
                id_to_repr[ident] = (best[0], best[1], float(xs[jj]), float(ys[jj]), float(ss[jj]))

        # Decode representative frames and crop thumbs
        thumbs = {}
        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened():
            raise RuntimeError(f"Failed to open video for identity thumbnails: {video_path}")
        for ident, (_score, gf, x, y, s) in id_to_repr.items():
            if gf < 0:
                continue
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(gf))
            ret, img = cap2.read()
            if not ret or img is None:
                continue
            h, w = img.shape[:2]
            x1 = max(0, int(x - s)); y1 = max(0, int(y - s))
            x2 = min(w, int(x + s)); y2 = min(h, int(y + s))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            thumb = cv2.resize(roi, (tile_w, tile_h))
            thumbs[ident] = thumb
        cap2.release()
        return thumbs

    # Restrict to identities that have spoken anywhere in the video
    spoken_identities = set()
    for seg in (diarization_results or []):
        ident = _normalize_identity_prefix(seg.get('identity')) if isinstance(seg, dict) else None
        if isinstance(ident, str) and ident not in (None, 'None'):
            spoken_identities.add(ident)
    # Color map: reuse global identity color map used by panel/memory to keep
    # colors consistent across left video and right panel. Convert RGB->BGR for OpenCV.
    id_colors_rgb = _id_color_map_global(tracks)
    ID_COLORS = {}
    for ident_norm, rgb in id_colors_rgb.items():
        if spoken_identities and ident_norm not in spoken_identities:
            continue
        r, g, b = rgb
        ID_COLORS[ident_norm] = (b, g, r)  # BGR
    # Build thumbnails only for spoken identities
    ID_THUMBS_ALL = _build_identity_thumbs(args.videoFilePath, tracks, scores)
    ID_THUMBS = {k: v for k, v in ID_THUMBS_ALL.items() if k in spoken_identities}

    cap = cv2.VideoCapture(args.videoFilePath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for visualization pass: {args.videoFilePath}")
    fidx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        t_sec = float(fidx) / float(args.videoFps)
        faces_frame = []
        for face in faces_by_frame.get(fidx, []):
            ident_raw = face.get('identity', None)
            if not isinstance(ident_raw, str) or ident_raw in (None, 'None'):
                # Tracks without a resolved identity get no color/overlay
                continue
            ident = _normalize_identity_prefix(ident_raw)
            # Only visualize faces for identities that have spoken
            if ident not in spoken_identities:
                continue
            faces_frame.append((ident, face))

        # Draw face overlays: for SAM3 backend use segmentation mask *borders*; otherwise fall back to rectangles.
        if faces_frame:
            alpha = 0.45
            if use_sam3_masks:
                if sam3_masks is None or fidx >= len(sam3_masks):
                    raise RuntimeError(f"SAM3 masks list shorter than video frames (frame {fidx})")
                frame_masks = sam3_masks[fidx] or []
                for ident, face in faces_frame:
                    if ident not in ID_COLORS:
                        raise RuntimeError(f"Missing color for identity {ident} in visualization map")
                    color = ID_COLORS[ident]
                    # Use SAM3 instance masks directly: choose the mask that
                    # contains the track center (x, y). Among all masks that
                    # include this center, prefer the one with highest score.
                    cx = int(round(face['x']))
                    cy = int(round(face['y']))
                    if cx < 0 or cx >= fw or cy < 0 or cy >= fh:
                        continue
                    best = None
                    best_score = float('-inf')
                    for m in frame_masks:
                        mask_arr = m.get('mask', None)
                        if mask_arr is None:
                            continue
                        mask_bool = np.asarray(mask_arr, dtype=bool)
                        if mask_bool.shape[0] != fh or mask_bool.shape[1] != fw:
                            raise RuntimeError(
                                f"SAM3 mask resolution {mask_bool.shape} does not match video frame {(fh, fw)}"
                            )
                        if not mask_bool[cy, cx]:
                            continue
                        s_val = float(m.get('score', 0.0))
                        if s_val > best_score:
                            best_score = s_val
                            best = mask_bool
                    if best is None:
                        continue  # no mask covers the track center; skip to avoid fabricating shape
                    # Only draw the *border* of the mask to avoid covering
                    # interior pixels. Smooth edges by a light blur on the mask
                    # before contour extraction.
                    mask_u8 = (best.astype(np.uint8) * 255)
                    mask_blur = cv2.GaussianBlur(mask_u8, (5, 5), 0)
                    _, mask_bin = cv2.threshold(mask_blur, 128, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cnt is None or len(cnt) < 3:
                            continue
                        cv2.drawContours(image, [cnt], -1, color, thickness=2)
            else:
                overlay = image.copy()
                for ident, face in faces_frame:
                    if ident not in ID_COLORS:
                        raise RuntimeError(f"Missing color for identity {ident} in visualization map")
                    color = ID_COLORS[ident]
                    x1 = int(face['x'] - face['s'])
                    y1 = int(face['y'] - face['s'])
                    x2 = int(face['x'] + face['s'])
                    y2 = int(face['y'] + face['s'])
                    x1 = max(0, min(x1, fw - 1))
                    y1 = max(0, min(y1, fh - 1))
                    x2 = max(x1 + 1, min(x2, fw))
                    y2 = max(y1 + 1, min(y2, fh))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

        vOut.write(image)
        fidx += 1
    cap.release()
    vOut.release()

    # Merge audio without generating subtitles (panel will carry text)
    video_with_audio_path = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
    command = ("%s -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
              (_FFMPEG_BIN,
               os.path.join(args.pyaviPath, 'video_only.avi'),
               os.path.join(args.pyaviPath, 'audio.wav'),
               args.nDataLoaderThread, video_with_audio_path))
    subprocess.call(command, shell=True)

def process_folder():
    videoNames = [os.path.splitext(os.path.basename(v))[0] for v in glob.glob(os.path.join(args.videoFolder, '*.*'))]
    print(videoNames)
    for videoName in videoNames:
        args.videoName = videoName
        process_video()

# Main function
def process_video():
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...    
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    # if os.path.exists(args.savePath):
    #     rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok = True) # Retained for compatibility; no frames will be saved
    os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Timing logger
    time_log_path = os.path.join(args.pyworkPath, 'time_log.jsonl')
    tlog = _StageTimer(time_log_path, meta={'pid': int(os.getpid()), 'video_name': str(args.videoName)})

    # Extract video/audio — skip CFR re-encode; operate on original container with PTS
    args.videoFilePath = args.videoPath
    force_rebuild = False
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    with tlog.timer('audio_extract'):
        if os.path.exists(args.audioFilePath):
            sys.stderr.write("Audio already exists, skipping extraction: %s \r\n" % args.audioFilePath)
        else:
            ss_to = ''
            if args.duration and float(args.duration) > 0:
                ss_to = f" -ss {float(args.start):.3f} -to {float(args.start)+float(args.duration):.3f}"
            cmd = (
                f"{_FFMPEG_BIN} -y -i {args.videoPath}{ss_to} -c:a pcm_s16le -ac 1 -vn -threads {int(args.nDataLoaderThread)} -ar 16000 {args.audioFilePath} -loglevel panic"
            )
            subprocess.call(cmd, shell=True, stdout=None)
            if not os.path.exists(args.audioFilePath):
                raise RuntimeError("Audio extraction failed: output file missing")

    # Derive an effective FPS without relying on extracted frames
    def _compute_effective_fps_from_container(audio_wav_path: str, video_path: str) -> float:
        """Derive FPS robustly without extracted frames.

        Priority:
        1) ffprobe r_frame_rate (preferred true stream rate, e.g., 24000/1001)
        2) ffprobe avg_frame_rate if r_frame_rate missing
        3) OpenCV CAP_PROP_FPS
        4) frame_count / audio_duration
        """
        def _parse_frac(s: str) -> float:
            try:
                if '/' in s:
                    a, b = s.split('/')
                    a = float(a.strip()); b = float(b.strip())
                    return float(a / b) if b != 0 else 0.0
                return float(s)
            except Exception:
                return 0.0

        # Try ffprobe for precise rates
        r_fps = 0.0
        avg_fps = 0.0
        try:
            cmd = [
                'ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=r_frame_rate,avg_frame_rate',
                '-of','default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip().splitlines()
            if len(out) >= 2:
                r_fps = _parse_frac(out[0].strip())
                avg_fps = _parse_frac(out[1].strip())
            elif len(out) == 1:
                # Some builds only output one entry
                r_fps = _parse_frac(out[0].strip())
        except Exception:
            pass

        # Heuristic: prefer r_frame_rate within sane bounds
        if r_fps and r_fps > 0:
            return float(r_fps)
        if avg_fps and avg_fps > 0:
            return float(avg_fps)

        # Fallbacks via OpenCV
        cap_local = cv2.VideoCapture(video_path)
        if not cap_local.isOpened():
            raise RuntimeError(f"Failed to open video to derive FPS: {video_path}")
        fps_v = float(cap_local.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap_local.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap_local.release()
        if fps_v and fps_v > 0:
            return fps_v
        if frame_count and frame_count > 0:
            sr_local, audio_local = wavfile.read(audio_wav_path)
            if sr_local <= 0 or audio_local is None or len(audio_local) <= 0:
                raise RuntimeError("Invalid audio for FPS derivation")
            dur_local = float(len(audio_local)) / float(sr_local)
            if dur_local <= 0:
                raise RuntimeError("Non-positive audio duration for FPS derivation")
            fps_eff_local = frame_count / dur_local
            if fps_eff_local <= 0:
                raise RuntimeError("Computed non-positive effective FPS")
            return float(fps_eff_local)
        raise RuntimeError("Unable to derive FPS from container metadata or audio duration")

    args.videoFps = _compute_effective_fps_from_container(args.audioFilePath, args.videoFilePath)
    sys.stderr.write(f"Using effective FPS from container: {args.videoFps:.6f}\n")

    # Scene detection for the video frames
    scene_path = os.path.join(args.pyworkPath, 'scene.pckl')
    if (not os.path.exists(scene_path)) or force_rebuild:
        with tlog.timer('scene_detect'):
            scene = scene_detect(args)
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))    
    else:
        with tlog.timer('scene_load'):
            sys.stderr.write("Loading existing scene detection from %s \r\n" % scene_path)
            with open(scene_path, 'rb') as fil:
                scene = pickle.load(fil)

    # Face detection for the video frames
    faces_path = os.path.join(args.pyworkPath, 'faces.pckl')
    if (not os.path.exists(faces_path)) or force_rebuild:
        with tlog.timer('face_detect'):
            # Use parallel SAM3 processing if enabled (default)
            if getattr(args, 'sam3Parallel', True):
                chunk_sec = float(getattr(args, 'sam3ChunkSec', 120.0))
                overlap_sec = float(getattr(args, 'sam3OverlapSec', 5.0))
                sys.stderr.write(f"Using parallel SAM3 with chunk={chunk_sec}s, overlap={overlap_sec}s\n")
                faces = inference_video_parallel(args, chunk_duration_sec=chunk_sec, overlap_sec=overlap_sec)
            else:
                faces = inference_video(args)
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))
    else:
        with tlog.timer('face_load'):
            sys.stderr.write("Loading existing face detection from %s \r\n" % faces_path)
            with open(faces_path, 'rb') as fil:
                faces = pickle.load(fil)

    # Face tracking
    tracks_path = os.path.join(args.pyworkPath, 'tracks.pckl')
    if (not os.path.exists(tracks_path)) or force_rebuild:
        with tlog.timer('track_build'):
            allTracks, vidTracks = [], []
            for shot in scene:
                if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
                    allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))

            # Build in-memory tracks and also write pycrop clips for robust ASD
            base_tracks = []
            for t in allTracks:
                tr_norm = {'frame': t['frame'], 'bbox': t['bbox']}
                base_tracks.append(tr_norm)
            # Always build in-memory tracks; do not write pycrop clips
            for tr_norm in base_tracks:
                vidTracks.append({
                    'track': tr_norm,
                    'proc_track': build_proc_track(tr_norm, args.cropScale),
                    'video_path': args.videoFilePath,
                    'cropScale': float(args.cropScale),
                })
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Tracks prepared (in-memory only).\r\n")
            with open(tracks_path, 'wb') as fil:
                pickle.dump(vidTracks, fil)

            # Build scene frame ranges for parallel ASD (used if in-memory ASD is needed)
            scene_ranges = []  # list of (start_frame, end_frame)
            for shot in scene:
                s_f = int(shot[0].frame_num)
                e_f = int(shot[1].frame_num) - 1
                if e_f >= s_f:
                    scene_ranges.append((s_f, e_f))
    else:
        with tlog.timer('track_load'):
            sys.stderr.write("Loading existing face tracks from %s \r\n" % tracks_path)
            with open(tracks_path, 'rb') as fil:
                vidTracks = pickle.load(fil)
            # Ensure in-memory identity clustering has needed context
            for tr in vidTracks:
                if isinstance(tr, dict):
                    tr['video_path'] = args.videoFilePath
                    tr['cropScale'] = float(args.cropScale)
            # Do not generate pycrop clips in the reload path either (always in-memory)

    # Active Speaker Detection by TalkNet — always compute in-memory (no pycrop clips)
    scores_path = os.path.join(args.pyworkPath, 'scores.pckl')
    if 'vidTracks' not in locals():
        with open(tracks_path, 'rb') as fil:
            vidTracks = pickle.load(fil)
    base_tracks = [vt['track'] for vt in vidTracks]
    # Build scene ranges from 'scene' list
    scene_ranges = []
    for shot in scene:
        s_f = int(shot[0].frame_num); e_f = int(shot[1].frame_num) - 1
        if e_f >= s_f:
            scene_ranges.append((s_f, e_f))
    # Assign tracks to scenes
    scene_tasks = []  # list of (indices, tracks_sub, start, end)
    for (s_f, e_f) in scene_ranges:
        idxs = []
        tr_sub = []
        for i, tr in enumerate(base_tracks):
            frs = tr['frame']
            if len(frs) == 0:
                continue
            t0 = int(frs[0]); t1 = int(frs[-1])
            if t0 >= s_f and t1 <= e_f:
                idxs.append(i); tr_sub.append(tr)
        if tr_sub:
            scene_tasks.append((idxs, tr_sub, s_f, e_f))
    # Run workers (multi-GPU: split scenes across processes, one GPU per worker)
    import torch.multiprocessing as mp
    minimal = dict(videoFilePath=args.videoFilePath, pyaviPath=args.pyaviPath, pretrainModel=args.pretrainModel, cropScale=args.cropScale, asdBatch=int(getattr(args,'asdBatch',64)))
    results = []
    with tlog.timer('asd_compute'):
        if scene_tasks:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            workers = max(1, min(len(scene_tasks), n_gpus if n_gpus > 0 else 1))
            if workers > 1:
                # Round-robin assign tasks to workers with fixed gpu_id per worker
                chunks = [[] for _ in range(workers)]
                for idx, task in enumerate(scene_tasks):
                    chunks[idx % workers].append(task)
                args_list = [(chunks[i], minimal, i if n_gpus > 0 else -1) for i in range(workers)]
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=workers) as pool:
                    mapped = pool.map(_asd_scene_worker, args_list)
                # Flatten
                for lst in mapped:
                    for pair in lst:
                        results.append(pair)
            else:
                from types import SimpleNamespace
                # Single process path
                for (idxs, tr_sub, s_f, e_f) in scene_tasks:
                    results.append((idxs, evaluate_network_in_memory(tr_sub, SimpleNamespace(**minimal), frame_start=s_f, frame_end=e_f)))
    # Merge scores back in order; initialize empty lists
    scores = [None] * len(base_tracks)
    for idxs, sc_scores in results:
        for k, i in enumerate(idxs):
            scores[i] = sc_scores[k]
    for i in range(len(scores)):
        if scores[i] is None:
            scores[i] = []
    with open(scores_path, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted (in-memory) and saved in %s \r\n" %args.pyworkPath)

    # Identity assignment via episode-level visual clustering (stable VID_* labels)
    identity_tracks_path = os.path.join(args.pyworkPath, 'tracks_identity.pckl')
    if os.path.exists(identity_tracks_path):
        with tlog.timer('identity_load'):
            sys.stderr.write("Loading existing identity tracks from %s \r\n" % identity_tracks_path)
            with open(identity_tracks_path, 'rb') as fil:
                annotated_tracks = pickle.load(fil)
        # Normalize legacy VID_* -> Person_* for display consistency
        changed = False
        for tr in annotated_tracks:
            ident = tr.get('identity')
            if isinstance(ident, str) and ident.startswith('VID_'):
                tr['identity'] = 'Person_' + ident.split('_', 1)[1]
                changed = True
        if changed:
            try:
                with open(identity_tracks_path, 'wb') as fil:
                    pickle.dump(annotated_tracks, fil)
            except Exception:
                pass
    else:
        with tlog.timer('identity_cluster'):
            sys.stderr.write("Clustering visual identities with constraints (in-memory embeddings if needed)...\r\n")
            # Pass ASD scores to enable active-frame gating inside clustering/embedding
            annotated_tracks = cluster_visual_identities(
                vidTracks,
                scores_list=scores if 'scores' in locals() else None,
                batch_size=int(getattr(args, 'idBatch', 64)),
                face_sim_thresh=0.50,
                save_avatars_path=os.path.join(args.pyworkPath, 'identity_avatars_magface.pckl'),
            )
            with open(identity_tracks_path, 'wb') as fil:
                pickle.dump(annotated_tracks, fil)
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Tracks with clustered identities saved in %s \r\n" % args.pyworkPath)

    # Ensure ASD scores align with current tracks; if not, recompute scores in-memory
    if not isinstance(scores, list) or len(scores) != len(annotated_tracks):
        sys.stderr.write("Recomputing ASD scores in-memory to align with current tracks...\n")
        from types import SimpleNamespace
        base_tracks = [vt['track'] if 'track' in vt else vt for vt in vidTracks]
        minimal = dict(videoFilePath=args.videoFilePath, pyaviPath=args.pyaviPath, pretrainModel=args.pretrainModel, cropScale=args.cropScale, asdBatch=int(getattr(args,'asdBatch',64)), videoFps=float(args.videoFps))
        scores = evaluate_network_in_memory(base_tracks, SimpleNamespace(**minimal))
        with open(scores_path, 'wb') as fil:
            pickle.dump(scores, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores re-extracted (in-memory) and saved in %s \r\n" %args.pyworkPath)

    # If frame lengths mismatch scores lengths (due to 25fps resampling), we used to resample tracks.
    # Current diarization mapping uses original annotated_tracks with absolute frame indices and
    # robustly aligns via min(len(frames), len(scores)), so resampling is no longer needed and
    # only adds heavy compute on long videos. Skip it to avoid stalls.
    mism = [i for i in range(min(len(annotated_tracks), len(scores))) if len(scores[i]) != (len(annotated_tracks[i]['track']['frame']) if 'track' in annotated_tracks[i] and 'frame' in annotated_tracks[i]['track'] else 0)]
    if mism:
        sys.stderr.write(f"Skipping resample-to-25fps for diarization (mismatch count={len(mism)})\n")

    # Run WhisperX diarization without constraining K (use only for words + timings)
    raw_diarization_path = os.path.join(args.pyworkPath, 'raw_diriazation.pckl')
    if os.path.exists(raw_diarization_path):
        with tlog.timer('diarization_load'):
            sys.stderr.write("Loading existing raw diarization from %s \r\n" % raw_diarization_path)
            with open(raw_diarization_path, 'rb') as fil:
                raw_segments = pickle.load(fil)
            raw_results = {"segments": raw_segments}
    else:
        with tlog.timer('diarization_compute'):
            raw_results = speech_diarization()
            with open(raw_diarization_path, 'wb') as fil:
                pickle.dump(raw_results["segments"], fil)
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Raw diarization extracted and saved in %s \r\n" %args.pyworkPath)

    # Per-segment framewise argmax mapping with minimal run-length smoothing
    matched_diarization_path = os.path.join(args.pyworkPath, 'matched_diriazation.pckl')
    # Important: use original annotated_tracks with absolute frame indices for mapping.
    # Resampled tracks (0..T-1) lose absolute timing and break alignment with diarization.
    with tlog.timer('diarization_match'):
        diar_for_subs = split_segments_by_positive_fill(
            annotated_tracks,
            scores,
            raw_results["segments"],
            fps=25.0,
            min_run_frames=6,
        )
    if not diar_for_subs:
        raise RuntimeError("Framewise argmax mapping produced no segments; aborting to avoid empty subtitles.")
    # Persist for inspection
    with open(matched_diarization_path, 'wb') as fil:
        pickle.dump(diar_for_subs, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Positive-fill diarization saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video    
    # Build word list for word-timed subtitles
    flat_words = _flatten_aligned_words(raw_results["segments"]) if isinstance(raw_results, dict) and 'segments' in raw_results else _flatten_aligned_words(raw_results)
    with tlog.timer('visualization'):
        visualization(annotated_tracks, scores, diar_for_subs, args, words_list=flat_words)

    # Optional: Skia side panel; optionally burn subtitles onto the left video before hstack
    if bool(getattr(args, 'renderPanel', False)):
        base_in = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
        if not os.path.isfile(base_in):
            raise RuntimeError(f"Base video for panel composition not found: {base_in}")
        # Burn ASS subtitles (panel messages) onto base if requested
        if bool(getattr(args, 'subtitle', False)):
            fonts_dir_abs, _ = _ensure_chinese_font()
            ass_path = os.path.join(args.pyworkPath, 'panel_subtitles.ass')
            try:
                id_colors_map = _id_color_map_global(annotated_tracks)
            except Exception:
                id_colors_map = {}
            generate_ass(diar_for_subs, ass_path, id_colors_map)
            base_sub = os.path.join(args.pyaviPath, 'video_out_with_audio_subs.mp4')
            with tlog.timer('subtitle_burn'):
                # Escape paths for ffmpeg filter
                ass_esc = ass_path.replace('\\', '\\\\')
                fdir_esc = fonts_dir_abs.replace('\\', '\\\\')
                cmd_sub = (
                    f"{_FFMPEG_BIN} -y -i {base_in} -vf \"subtitles='{ass_esc}':fontsdir='{fdir_esc}'\" "
                    f"-c:v libx264 -crf 18 -threads {int(args.nDataLoaderThread)} "
                    f"-c:a aac -b:a 192k -ar 16000 {base_sub} -loglevel error"
                )
                rc0 = subprocess.call(cmd_sub, shell=True)
                if rc0 != 0 or (not os.path.isfile(base_sub)):
                    raise RuntimeError('Failed to burn subtitles onto base video')
            base_in = base_sub
        # Render panel
        panel_out = os.path.join(args.pyaviPath, 'video_panel.mp4')
        sys.stderr.write(f"Rendering Skia side panel to {panel_out}...\n")
        with tlog.timer('panel_render'):
            render_side_panel_skia(
                base_in,
                diar_for_subs,
                annotated_tracks,
                panel_out,
                width_ratio=float(getattr(args, 'panelWidthRatio', 0.28)),
                theme=str(getattr(args, 'panelTheme', 'glass')).lower(),
                max_items=int(getattr(args, 'panelMaxItems', 6)),
            )
        # hstack left(base) + right(panel)
        combined = os.path.join(args.pyaviPath, 'video_with_panel.mp4')
        cmd_combine = (
            f"{_FFMPEG_BIN} -y -i {base_in} -i {panel_out} -filter_complex \"[0:v][1:v]hstack=inputs=2[v]\" "
            f"-map \"[v]\" -map 0:a? -c:v libx264 -crf 18 -threads {int(args.nDataLoaderThread)} "
            f"-c:a aac -b:a 192k -ar 16000 {combined} -loglevel error"
        )
        with tlog.timer('panel_compose'):
            rc = subprocess.call(cmd_combine, shell=True)
        if rc != 0:
            raise RuntimeError("Failed to compose base video with side panel via ffmpeg hstack")
        sys.stderr.write(f"Composed video saved: {combined}\n")

if __name__ == '__main__':
    process_folder()
