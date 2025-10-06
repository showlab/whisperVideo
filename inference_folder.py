import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
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

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from identity_verifier import IdentityVerifier
try:
    from identity_cluster import cluster_visual_identities
except Exception:
    # When run as module, fallback to relative import
    from .identity_cluster import cluster_visual_identities

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Demo")

# parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="/workspace/siyuan/siyuan/whisperv_proj/data/video/Frasier",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=4,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')
parser.add_argument('--facedetBatch',         type=int,   default=128,  help='Batch size for S3FD face detection (increase on 24GB GPUs)')
parser.add_argument('--idBatch',              type=int,   default=64,   help='Batch size for identity embedding (MagFace/Facenet)')
parser.add_argument('--asdBatch',             type=int,   default=64,   help='Batch size for ASD window inference')
parser.add_argument('--cropWorkers',          type=int,   default=8,    help='Parallel workers for audio cut + mux per track')
parser.add_argument('--sceneWorkers',         type=int,   default=6,    help='Parallel workers for in-memory ASD by scene')
parser.add_argument('--sceneMinSec',         type=float, default=1.0,  help='Minimum scene length in seconds (detector min_scene_len)')

args, unknown  = parser.parse_known_args()

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)


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
    # GPU: Face detection from container video stream with torch batch processing
    DET = S3FD(device='cuda')
    cap = cv2.VideoCapture(args.videoFilePath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for detection: {args.videoFilePath}")

    dets = []
    fidx = 0
    batch_imgs = []
    batch_idx = []
    B = max(1, int(args.facedetBatch))

    def flush_batch():
        nonlocal batch_imgs, batch_idx, dets
        if not batch_imgs:
            return
        # Run batched S3FD on RGB images
        batched_boxes = DET.detect_faces_batch(batch_imgs, conf_th=0.9, scales=[args.facedetScale])
        for local_i, boxes in enumerate(batched_boxes):
            fi = batch_idx[local_i]
            frame_dets = []
            for bbox in boxes:
                frame_dets.append({'frame': fi, 'bbox': (bbox[:-1]).tolist(), 'conf': float(bbox[-1])})
            dets.append(frame_dets)
            if fi % 50 == 0:
                sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fi, len(frame_dets)))
        batch_imgs = []
        batch_idx = []

    while True:
        ret, image = cap.read()
        if not ret:
            break
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch_imgs.append(imageNumpy)
        batch_idx.append(fidx)
        if len(batch_imgs) >= B:
            flush_batch()
        fidx += 1
    cap.release()
    flush_batch()

    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets

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
    command = ("ffmpeg -y -i %s -c:a pcm_s16le -ac 1 -vn -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
              (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
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
    # 1) Prepare 25fps time grid and per-track start/end seconds via constant FPS timeline
    if not hasattr(args, 'videoFps') or args.videoFps is None or float(args.videoFps) <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot align in-memory ASD.")
    FPS_SRC = float(args.videoFps)
    median_dt = 1.0 / max(1.0, FPS_SRC)

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
        # Use CFR timeline derived from effective FPS for robust alignment
        t_s = float(s_f) / FPS_SRC
        t_e = float(e_f) / FPS_SRC
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
                t_src = float(fidx) / FPS_SRC
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
            t_s = frame_times_sec[s_f]; t_e = frame_times_sec[e_f]
            if not (isinstance(t_s, float) and isinstance(t_e, float)):
                raise RuntimeError("Invalid PTS timestamps for track start/end")
            track_start_sec[len(track_ranges)-1] = float(t_s)
            track_end_sec[len(track_ranges)-1] = float(t_e)

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
            t_src = frame_times_sec[fidx]
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
            "ffmpeg -y -i %s -c:a pcm_s16le -ac 1 -vn -ar 16000 -threads %d -ss %.6f -to %.6f %s -loglevel panic"
            % (audioPath, ff_threads, float(aStart), float(aEnd), audioTmp)
        )
        r1 = subprocess.call(cmd1, shell=True, stdout=None)
        if r1 != 0:
            return (tidx, False, f"audio cut failed rc={r1}")
        cmd2 = (
            "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
            % (cropFile, audioTmp, ff_threads, cropFile)
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

    Args: (idxs, tr_sub, s_f, e_f, minimal_dict)
    Returns: (idxs, scores_sublist)
    """
    from types import SimpleNamespace
    idxs, tr_sub, s_f, e_f, minimal = args_pack
    a = SimpleNamespace(**minimal)
    return (idxs, evaluate_network_in_memory(tr_sub, a, frame_start=s_f, frame_end=e_f))

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet (batched windows)
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()
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
        for ident in uniq:
            hval = int(hashlib.md5(ident.encode('utf-8')).hexdigest()[:8], 16)
            h = (hval % 360) / 360.0
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

        # Bottom-left Memory Bank overlay (tracking-based faces)
        mem_entries = mem_rois_by_frame.get(fidx, [])
        if mem_entries:
            max_cols = max(1, min(6, len(mem_entries)))
            rows = 1 if len(mem_entries) <= max_cols else 2
            cols = max_cols if rows == 1 else int(math.ceil(len(mem_entries) / 2.0))
            limit = rows * cols
            block_w = cols * tile_w + (cols - 1) * margin
            block_h = rows * tile_h + (rows - 1) * margin + label_height
            y0 = max(0, fh - block_h - 10)
            x0 = 10
            cv2.rectangle(image, (x0 - 6, y0 - 6), (x0 + block_w + 6, y0 + block_h + 6), (0, 0, 0), thickness=-1)
            cv2.putText(image, 'Memory', (x0, y0 + label_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            start_y = y0 + label_height
            for idx_m, (gf, mx, my, ms) in enumerate(mem_entries[:limit]):
                r = idx_m // cols
                c = idx_m % cols
                ty = start_y + r * (tile_h + margin)
                tx = x0 + c * (tile_w + margin)
                thumb = get_face_thumb_from_flist(int(gf), mx, my, ms)
                if thumb is None:
                    continue
                h_t, w_t = thumb.shape[:2]
                if ty + h_t <= fh and tx + w_t <= fw:
                    image[ty:ty+h_t, tx:tx+w_t] = thumb
                    cv2.rectangle(image, (tx, ty), (tx + w_t, ty + h_t), (120, 120, 120), 1)
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
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


def speech_diarization(min_speakers: int = None, max_speakers: int = None):
    # device = "cpu"
    device = "cuda"
    audio_file = os.path.join(args.pyaviPath, "audio.wav")
    batch_size = 32 # reduce if low on GPU mem
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
        dev_idx = 0 if device == "cuda" else -1
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model=(model_name if model_name else "BELLE-2/Belle-whisper-large-v3-zh"),
            device=dev_idx,
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
        # WhisperX internal loader (faster-whisper / ct2)
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        print(result.get("segments", []))  # before alignment
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # segments after ASR (+optional alignment)

    # 3. Assign speaker labels
    # Read HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HuggingFace token: set HF_TOKEN or HUGGINGFACE_TOKEN in environment.")

    try:
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
    """Assign Person_* to diarization segments by maximizing ASD energy overlap.

    Replaces binary count of (score > 0) with sum(ReLU(score)) over the overlap to reduce
    sensitivity to thresholding and fragmented positives, improving robustness.
    """
    matched_results = []

    for diar in diarization_result:
        if "speaker" not in diar:
            continue
        start_time = float(diar.get("start", diar.get("start_time", 0.0)))
        end_time = float(diar.get("end", diar.get("end_time", start_time)))
        if end_time <= start_time:
            continue
        speaker = diar["speaker"]

        ds = int(start_time * fps)
        de = int(end_time * fps)

        best_match_identity = None
        best_energy = -1.0

        for i, tr in enumerate(vidTracks):
            identity = tr.get("identity", None)
            if not (isinstance(identity, str) and identity not in (None, 'None')):
                continue
            frames = tr["track"]["frame"]
            sc = scores[i] if i < len(scores) else []
            if not isinstance(sc, (list, tuple)) or len(sc) == 0:
                continue

            t0 = int(frames[0]); t1 = int(frames[-1])
            a = max(t0, ds); b = min(t1, de)
            if a >= b:
                continue
            s_idx = max(0, a - t0)
            e_idx = min(len(sc) - 1, b - t0)
            if e_idx < s_idx:
                continue
            # Energy = sum of positive scores in overlap
            e = 0.0
            for j in range(s_idx, e_idx + 1):
                v = float(sc[j])
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
    """Split each diarization segment into subsegments using ASD>0 evidence first, then fill.

    Strategy (simple and robust):
    - Build two maps:
      (1) id_pos_frames: identity -> set of global frames where its ASD score > 0 (merge tracks).
      (2) per_frame_scores: frame -> {identity: score} to resolve ties/adjacency if needed.
    - For a sentence [s,e), compute the sentence-level top identity by positive-frame count within [s,e).
      If no identity has positive frames in [s,e), raise (no fake mapping).
    - For each frame f in [s,e):
        * If exactly one identity has score>0 at f -> assign that identity.
        * Otherwise -> assign sentence-level top identity to avoid gaps.
    - Run-length compress assignments; absorb runs shorter than min_run_frames into neighbors; coalesce.
    - Return list of {'start','end','identity','text'}.
    """
    # 1) Build maps
    from collections import defaultdict
    id_pos = defaultdict(set)
    per_frame = defaultdict(dict)
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
            f = int(fr_list[j]); v = float(sc_arr[j])
            if v > 0.0:
                id_pos[ident].add(f)
            m = per_frame[f]
            if (ident not in m) or (v > m[ident]):
                m[ident] = v

    out = []
    for seg in raw_segments:
        s = float(seg.get('start', seg.get('start_time', 0.0)))
        e = float(seg.get('end', seg.get('end_time', s)))
        if e <= s:
            continue
        a = int(s * float(fps)); b = int(e * float(fps))
        if b <= a:
            continue
        # sentence-level positive winner
        pos_counts = defaultdict(int)
        for ident, aset in id_pos.items():
            # count positive frames within [a,b)
            c = 0
            # fast path: iterate over f in [a,b) checking membership
            for f in range(a, b):
                if f in aset:
                    c += 1
            if c > 0:
                pos_counts[ident] += c
        if not pos_counts:
            raise RuntimeError("No identity with positive ASD within diarization segment; cannot assign labels.")
        sent_top = max(pos_counts.items(), key=lambda x: x[1])[0]

        # framewise assignment with fill
        winners = []
        frames = list(range(a, b))
        for f in frames:
            m = per_frame.get(f, None)
            if not m:
                winners.append(sent_top)
                continue
            pos = [i for i, v in m.items() if v > 0.0]
            if len(pos) == 1:
                winners.append(pos[0])
            else:
                winners.append(sent_top)

        # RLE
        runs = []
        cur_ident = None
        cur_start = None
        for idx, ident in enumerate(winners):
            if ident != cur_ident:
                if cur_ident is not None:
                    runs.append((frames[cur_start], frames[idx], cur_ident))
                cur_ident = ident
                cur_start = idx
        if cur_ident is not None:
            runs.append((frames[cur_start], frames[-1] + 1, cur_ident))

        # absorb short runs
        if min_run_frames and len(runs) >= 2:
            merged = []
            i = 0
            while i < len(runs):
                rs, re, rid = runs[i]
                L = re - rs
                if L >= min_run_frames or len(runs) == 1:
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

        # emit segments
        for rs, re, rid in runs:
            st = max(s, rs / float(fps))
            et = min(e, re / float(fps))
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
        color = id_colors_map.get(ident, id_colors_map.get(_normalize_identity_prefix(ident), (255,255,255)))
        ass_hex = _bgr_to_ass_hex(color)
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
        color = id_colors_map.get(ident, id_colors_map.get(_normalize_identity_prefix(ident), (255,255,255)))
        ass_hex = _bgr_to_ass_hex(color)
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
    # Build per-frame overlays without using pyframes
    faces_by_frame = defaultdict(list)
    for tidx, track in enumerate(tracks):
        if tidx >= len(scores):
            continue
        identity = track.get('identity', 'None')
        score = scores[tidx]
        frames_arr = track['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for lidx, f in enumerate(frames_list):
            # average smoothing around local index
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

    # Stable color map
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
        for ident in uniq:
            hval = int(hashlib.md5(ident.encode('utf-8')).hexdigest()[:8], 16)
            h = (hval % 360) / 360.0
            s = 0.65
            v = 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors[ident] = (int(b * 255), int(g * 255), int(r * 255))
        return colors

    ID_COLORS = _id_color_map(tracks)

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
        for face in faces_by_frame.get(fidx, []):
            ident = face['identity']
            # Only visualize faces for identities that have spoken
            if isinstance(ident, str) and ident not in spoken_identities:
                continue
            color = ID_COLORS.get(ident, (200, 200, 200))
            bbox_x = int(face['x'] - face['s'])
            bbox_y = int(face['y'] - face['s'])
            bbox_x = max(0, min(bbox_x, fw - 1))
            bbox_y = max(0, min(bbox_y, fh - 1))
            cv2.rectangle(image,
                          (bbox_x, bbox_y),
                          (int(face['x'] + face['s']), int(face['y'] + face['s'])),
                          color, 5)
            # Speech bubble disabled (rolled back by request)
        # Bottom-left Global Identity Memory (unique per Person_*)
        if ID_THUMBS:
            id_list = sorted(ID_THUMBS.keys())
            max_cols = max(1, min(6, len(id_list)))
            rows = int(math.ceil(len(id_list) / float(max_cols)))
            cols = max_cols
            block_w = cols * tile_w + (cols - 1) * margin
            block_h = rows * tile_h + (rows - 1) * margin + label_height
            y0 = max(0, fh - block_h - 10)
            x0 = 10
            cv2.rectangle(image, (x0 - 6, y0 - 6), (x0 + block_w + 6, y0 + block_h + 6), (0, 0, 0), thickness=-1)
            cv2.putText(image, 'Memory', (x0, y0 + label_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            start_y = y0 + label_height
            for idx_m, ident in enumerate(id_list):
                r = idx_m // cols
                c = idx_m % cols
                ty = start_y + r * (tile_h + margin)
                tx = x0 + c * (tile_w + margin)
                thumb = ID_THUMBS.get(ident)
                if thumb is None:
                    continue
                h_t, w_t = thumb.shape[:2]
                if ty + h_t <= fh and tx + w_t <= fw:
                    image[ty:ty+h_t, tx:tx+w_t] = thumb
                    # Use identity color for tile border
                    color = ID_COLORS.get(ident, (120,120,120))
                    cv2.rectangle(image, (tx, ty), (tx + w_t, ty + h_t), color, 2)

        vOut.write(image)
        fidx += 1
    cap.release()
    vOut.release()

    # Generate ASS subtitles and merge variants
    output_ass_path = os.path.join(args.pyaviPath, 'subtitles.ass')
    fonts_dir_abs, font_name = _ensure_chinese_font()
    generate_ass_seq_wordtimed(diarization_results, output_ass_path, ID_COLORS, font_name_override=font_name or 'Noto Sans CJK SC', words_list=words_list)

    video_with_audio_path = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
              (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, video_with_audio_path))
    subprocess.call(command, shell=True)

    video_with_subtitles_path = os.path.join(args.pyaviPath, 'video_out_with_subtitles.avi')
    vf = f"subtitles={output_ass_path}:fontsdir={fonts_dir_abs}"
    command_with_subtitles = (
        "ffmpeg -y -i %s -vf \"%s\" -c:v libx264 -preset ultrafast -crf 23 %s -loglevel panic" %
        (video_with_audio_path, vf, video_with_subtitles_path)
    )
    subprocess.call(command_with_subtitles, shell=True)

    video_without_subtitles_path = os.path.join(args.pyaviPath, 'video_out_without_subtitles.avi')
    command_without_subtitles = ("ffmpeg -y -i %s -c copy %s -loglevel panic" %
                                (video_with_audio_path, video_without_subtitles_path))
    subprocess.call(command_without_subtitles, shell=True)

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

    # Extract video
    # Re-encode input to constant 25fps for robust, consistent timeline
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.mp4')
    def _ensure_cfr25(input_path: str, output_path: str, start: float = 0.0, duration: float = 0.0, threads: int = 4):
        import subprocess, os
        # Always produce 25fps CFR H.264 MP4 with audio preserved
        ss_to = ''
        if duration and duration > 0:
            ss_to = f" -ss {start:.3f} -to {start+duration:.3f}"
        cmd = (
            f"ffmpeg -y -i {input_path}{ss_to} -r 25 -vsync cfr -pix_fmt yuv420p "
            f"-c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k -threads {int(threads)} {output_path} -loglevel panic"
        )
        return subprocess.call(cmd, shell=True, stdout=None)

    need_encode = True
    if os.path.exists(args.videoFilePath):
        # Check if already 25fps
        try:
            out = subprocess.check_output([
                'ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=r_frame_rate,avg_frame_rate',
                '-of','default=noprint_wrappers=1:nokey=1',
                args.videoFilePath
            ], stderr=subprocess.STDOUT).decode('utf-8').strip().splitlines()
            def _parse_frac(s: str) -> float:
                try:
                    if '/' in s:
                        a,b = s.split('/')
                        return float(a)/float(b) if float(b) != 0 else 0.0
                    return float(s)
                except Exception:
                    return 0.0
            r_fps = _parse_frac(out[0].strip()) if len(out) >= 1 else 0.0
            avg_fps = _parse_frac(out[1].strip()) if len(out) >= 2 else 0.0
            eff = r_fps or avg_fps
            need_encode = abs(eff - 25.0) > 0.01
        except Exception:
            need_encode = True
    if need_encode:
        print("Re-encoding input to 25fps CFR...")
        rc = _ensure_cfr25(args.videoPath, args.videoFilePath, float(args.start), float(args.duration) if args.duration else 0.0, int(args.nDataLoaderThread))
        if rc != 0:
            raise RuntimeError("Failed to re-encode input to 25fps for processing")
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Re-encoded the video to 25fps CFR at %s \r\n" %(args.videoFilePath))
    else:
        sys.stderr.write("25fps CFR video already exists, skipping re-encode: %s \r\n" % args.videoFilePath)
    force_rebuild = need_encode
    
    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    if os.path.exists(args.audioFilePath):
        sys.stderr.write("Audio already exists, skipping extraction: %s \r\n" % args.audioFilePath)
    else:
        print("Extracting audio...")
        command = ("ffmpeg -y -i %s -c:a pcm_s16le -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
            (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

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
    # Force to exactly 25 for downstream alignment since we re-encoded to CFR 25
    args.videoFps = 25.0
    sys.stderr.write(f"Using effective FPS for timeline alignment: {args.videoFps:.6f}\n")

    # Scene detection for the video frames
    scene_path = os.path.join(args.pyworkPath, 'scene.pckl')
    if (not os.path.exists(scene_path)) or force_rebuild:
        scene = scene_detect(args)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))    
    else:
        sys.stderr.write("Loading existing scene detection from %s \r\n" % scene_path)
        with open(scene_path, 'rb') as fil:
            scene = pickle.load(fil)

    # Face detection for the video frames
    faces_path = os.path.join(args.pyworkPath, 'faces.pckl')
    if (not os.path.exists(faces_path)) or force_rebuild:
        faces = inference_video(args)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))
    else:
        sys.stderr.write("Loading existing face detection from %s \r\n" % faces_path)
        with open(faces_path, 'rb') as fil:
            faces = pickle.load(fil)

    # Face tracking
    tracks_path = os.path.join(args.pyworkPath, 'tracks.pckl')
    if (not os.path.exists(tracks_path)) or force_rebuild:
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
    # Run workers
    import torch.multiprocessing as mp
    minimal = dict(videoFilePath=args.videoFilePath, pyaviPath=args.pyaviPath, pretrainModel=args.pretrainModel, cropScale=args.cropScale, asdBatch=int(getattr(args,'asdBatch',64)), videoFps=float(args.videoFps))
    results = []
    if scene_tasks:
        if int(getattr(args,'sceneWorkers',6)) > 1:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=int(getattr(args,'sceneWorkers',6))) as pool:
                for idxs, sc_scores in pool.map(_asd_scene_worker, [(idxs, tr_sub, s_f, e_f, minimal) for (idxs,tr_sub,s_f,e_f) in scene_tasks]):
                    results.append((idxs, sc_scores))
        else:
            from types import SimpleNamespace
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
        sys.stderr.write("Clustering visual identities with constraints (in-memory embeddings if needed)...\r\n")
        # Pass ASD scores to enable active-frame gating inside clustering/embedding
        annotated_tracks = cluster_visual_identities(
            vidTracks,
            scores_list=scores if 'scores' in locals() else None,
            batch_size=int(getattr(args, 'idBatch', 64)),
            face_sim_thresh=0.50,
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

    # If frame lengths mismatch scores lengths (due to 25fps resampling), build resampled tracks for diarization
    mism = [i for i in range(min(len(annotated_tracks), len(scores))) if len(scores[i]) != (len(annotated_tracks[i]['track']['frame']) if 'track' in annotated_tracks[i] and 'frame' in annotated_tracks[i]['track'] else 0)]
    if mism:
        sys.stderr.write(f"Resampling tracks to 25fps grid for diarization (mismatch count={len(mism)})\n")
        annotated_tracks_25 = _resample_tracks_to_scores(annotated_tracks, scores)
    else:
        annotated_tracks_25 = annotated_tracks

    # Run WhisperX diarization without constraining K (use only for words + timings)
    raw_diarization_path = os.path.join(args.pyworkPath, 'raw_diriazation.pckl')
    if os.path.exists(raw_diarization_path):
        sys.stderr.write("Loading existing raw diarization from %s \r\n" % raw_diarization_path)
        with open(raw_diarization_path, 'rb') as fil:
            raw_segments = pickle.load(fil)
        raw_results = {"segments": raw_segments}
    else:
        raw_results = speech_diarization()
        with open(raw_diarization_path, 'wb') as fil:
            pickle.dump(raw_results["segments"], fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Raw diarization extracted and saved in %s \r\n" %args.pyworkPath)

    # Per-segment framewise argmax mapping with minimal run-length smoothing
    matched_diarization_path = os.path.join(args.pyworkPath, 'matched_diriazation.pckl')
    # Important: use original annotated_tracks with absolute frame indices for mapping.
    # Resampled tracks (0..T-1) lose absolute timing and break alignment with diarization.
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
    visualization(annotated_tracks, scores, diar_for_subs, args, words_list=flat_words)    

if __name__ == '__main__':
    process_folder()
