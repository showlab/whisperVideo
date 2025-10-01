import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
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
    sceneManager.add_detector(ContentDetector())
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
        sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
    return sceneList

def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
          dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath,'faces.pckl')
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

def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        # Keep TalkNet's expected 4:1 audio:video temporal ratio (100 Hz audio, 25 fps video)
        # Use videoFeature frames count normalized by 25 to define the common length baseline.
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * 25)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)    
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
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
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("medium", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

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

def match_speaker_identity(vidTracks, scores, diarization_result, fps=25):
    matched_results = []

    # Iterate over diarization results to match each speaker segment
    for diarization in diarization_result:
        # Skip segments without speaker information
        if "speaker" not in diarization:
            continue
        start_time = diarization["start"]
        end_time = diarization["end"]
        speaker = diarization["speaker"]

        # Convert diarization time to frames based on the given FPS
        diarization_start_frame = int(start_time * fps)
        diarization_end_frame = int(end_time * fps)

        # Iterate over vidTracks to find the identity matching this time segment
        best_match_identity = None
        max_active_frames = 0

        for i, track_data in enumerate(vidTracks):
            frames = track_data["track"]["frame"]
            identity = track_data["identity"]
            score = scores[i]  # Corresponding score sublist

            # Convert frame indices to time intervals (assuming 1 frame per 1/fps seconds)
            track_start_frame = frames[0]
            track_end_frame = frames[-1]

            # Check for overlap with the diarization segment in frame units
            overlap_start = max(track_start_frame, diarization_start_frame)
            overlap_end = min(track_end_frame, diarization_end_frame)

            if overlap_start < overlap_end:
                # Calculate active speaking frames during the overlapping time
                start_frame_idx = overlap_start - track_start_frame
                end_frame_idx = overlap_end - track_start_frame

                # Ensure the indices are within the bounds of the score list
                start_frame_idx = max(0, min(start_frame_idx, len(score) - 1))
                end_frame_idx = max(0, min(end_frame_idx, len(score) - 1))

                # Count active frames (where score > 0 indicates speaking)
                active_frames = sum(1 for j in range(start_frame_idx, end_frame_idx + 1) if score[j] > 0)

                # Select the identity with the most active frames during this overlap
                if active_frames > max_active_frames:
                    max_active_frames = active_frames
                    best_match_identity = identity

        matched_results.append({
            "speaker": speaker,
            "identity": best_match_identity,
            "text": diarization["text"],
            "start_time": start_time,
            "end_time": end_time
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

def refine_diarization_with_visual(annotated_tracks, scores, raw_segments, fps=25, tau=0.3, min_seg=0.08, merge_gap=0.2):
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
            best_ov = 0.0
            for ident, ivs in track_intervals:
                ov = 0.0
                for iv in ivs:
                    ov += _overlap_dur((a, b), iv)
                if ov > best_ov:
                    best_ov = ov
                    best_id = ident
            ratio = best_ov / max(1e-6, (b - a))
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

def _wrap_text_for_ass(text: str, max_chars_cn: int = 18, max_chars_lat: int = 28) -> str:
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

def visualization(tracks, scores, diarization_results, args):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    if len(flist) == 0:
        raise RuntimeError(f"No frames found under {args.pyframesPath}")

    num_frames = len(flist)
    faces = [[] for _ in range(num_frames)]

    skipped_oob = 0
    for tidx, track in enumerate(tracks):
        if tidx >= len(scores):
            continue
        identity = track.get('identity', 'None')
        score = scores[tidx]
        frames_arr = track['track']['frame']
        frames_list = frames_arr.tolist() if hasattr(frames_arr, 'tolist') else list(frames_arr)
        for fidx, frame in enumerate(frames_list):
            s_window = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]  # average smoothing
            s_val = float(np.mean(s_window)) if len(s_window) > 0 else 0.0
            if frame < 0 or frame >= num_frames:
                skipped_oob += 1
                continue
            faces[frame].append({
                'track': tidx,
                'score': s_val,
                'identity': identity,
                's': track['proc_track']['s'][fidx],
                'x': track['proc_track']['x'][fidx],
                'y': track['proc_track']['y'][fidx]
            })

    if skipped_oob > 0:
        sys.stderr.write(f"Visualization: skipped {skipped_oob} out-of-bounds frames\n")

    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot render visualization with correct timing.")
    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, 'video_only.avi'),
        cv2.VideoWriter_fourcc(*'XVID'),
        float(args.videoFps),
        (fw, fh),
        True,
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

    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)

        for face in faces[fidx]:
            ident = face['identity']
            color = ID_COLORS.get(ident, (200, 200, 200))
            bbox_x = int(face['x'] - face['s'])
            bbox_y = int(face['y'] - face['s'])
            bbox_x = max(0, min(bbox_x, fw - 1))
            bbox_y = max(0, min(bbox_y, fh - 1))

            # Draw bounding box (always). Always show ID label; append " (speaking)" when speaking
            cv2.rectangle(image,
                          (bbox_x, bbox_y),
                          (int(face['x'] + face['s']), int(face['y'] + face['s'])),
                          color, 5)
            if isinstance(ident, str) and ident != 'None':
                label = ident + (" (speaking)" if face['score'] > 0 else "")
                cv2.putText(image, label,
                            (bbox_x, max(0, bbox_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color, 3)

        vOut.write(image)

    vOut.release()

    # Generate ASS subtitles with color-matched Person_[ID] and line-wrapping
    output_ass_path = os.path.join(args.pyaviPath, 'subtitles.ass')
    # Ensure we select a font; reuse the same Chinese-capable font name
    fonts_dir_abs, font_name = _ensure_chinese_font()
    generate_ass_seq(diarization_results, output_ass_path, ID_COLORS, font_name_override=font_name or 'Noto Sans CJK SC')

    # Optimize audio/video merging with stream copying
    video_with_audio_path = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
              (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, video_with_audio_path))
    subprocess.call(command, shell=True)

    # Produce a video with ASS subtitles using fast encoding (force Chinese-capable font directory)
    video_with_subtitles_path = os.path.join(args.pyaviPath, 'video_out_with_subtitles.avi')
    # The 'subtitles' filter auto-detects ASS and uses embedded styles; still pass fontsdir for font discovery
    vf = f"subtitles={output_ass_path}:fontsdir={fonts_dir_abs}"
    command_with_subtitles = (
        "ffmpeg -y -i %s -vf \"%s\" -c:v libx264 -preset ultrafast -crf 23 %s -loglevel panic" %
        (video_with_audio_path, vf, video_with_subtitles_path)
    )
    subprocess.call(command_with_subtitles, shell=True)

    # Produce a video without subtitles using stream copying
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
    os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
    os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    if os.path.exists(args.videoFilePath):
        sys.stderr.write("Video already exists, skipping extraction: %s \r\n" % args.videoFilePath)
    else:
        print("Extracting video...")
        # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
        if args.duration == 0:
            command = ("ffmpeg -y -i %s -c copy -threads %d %s -loglevel panic" % \
                (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
        else:
            command = ("ffmpeg -y -i %s -c copy -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
                (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
    
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

    # Extract the video frames
    frames_exist = len(glob.glob(os.path.join(args.pyframesPath, '*.jpg'))) > 0
    if frames_exist:
        sys.stderr.write("Frames already exist, skipping extraction: %s \r\n" % args.pyframesPath)
    else:
        print("Extracting frames...")
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
            (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

    # Derive an effective FPS that aligns extracted frames with the audio timeline
    def _compute_effective_fps(pyframes_dir: str, audio_wav_path: str, fallback_video_path: str) -> float:
        flist_local = sorted(glob.glob(os.path.join(pyframes_dir, '*.jpg')))
        n_frames_local = len(flist_local)
        if n_frames_local <= 0:
            raise RuntimeError(f"No frames found under {pyframes_dir}")
        try:
            sr_local, audio_local = wavfile.read(audio_wav_path)
            if sr_local <= 0 or audio_local is None or len(audio_local) <= 0:
                raise RuntimeError("Invalid audio for FPS derivation")
            dur_local = float(len(audio_local)) / float(sr_local)
            if dur_local <= 0:
                raise RuntimeError("Non-positive audio duration for FPS derivation")
            fps_eff_local = float(n_frames_local) / dur_local
            if fps_eff_local <= 0:
                raise RuntimeError("Computed non-positive effective FPS")
            return fps_eff_local
        except Exception:
            cap_local = cv2.VideoCapture(fallback_video_path)
            fps_v = float(cap_local.get(cv2.CAP_PROP_FPS))
            cap_local.release()
            if fps_v is None or fps_v <= 0:
                raise RuntimeError("Unable to derive FPS from frames+audio or container video")
            return fps_v

    try:
        args.videoFps = _compute_effective_fps(args.pyframesPath, args.audioFilePath, args.videoFilePath)
        sys.stderr.write(f"Using effective FPS for timeline alignment: {args.videoFps:.6f}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to determine effective FPS: {e}")

    # Scene detection for the video frames
    scene_path = os.path.join(args.pyworkPath, 'scene.pckl')
    if os.path.exists(scene_path):
        sys.stderr.write("Loading existing scene detection from %s \r\n" % scene_path)
        with open(scene_path, 'rb') as fil:
            scene = pickle.load(fil)
    else:
        scene = scene_detect(args)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))    

    # Face detection for the video frames
    faces_path = os.path.join(args.pyworkPath, 'faces.pckl')
    if os.path.exists(faces_path):
        sys.stderr.write("Loading existing face detection from %s \r\n" % faces_path)
        with open(faces_path, 'rb') as fil:
            faces = pickle.load(fil)
    else:
        faces = inference_video(args)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    tracks_path = os.path.join(args.pyworkPath, 'tracks.pckl')
    if os.path.exists(tracks_path):
        sys.stderr.write("Loading existing face tracks from %s \r\n" % tracks_path)
        with open(tracks_path, 'rb') as fil:
            vidTracks = pickle.load(fil)
    else:
        allTracks, vidTracks = [], []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
                allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

        # Face clips cropping
        for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
            vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
        with open(tracks_path, 'wb') as fil:
            pickle.dump(vidTracks, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)

    # Active Speaker Detection by TalkNet (compute before identities to use ASD-gated embeddings)
    scores_path = os.path.join(args.pyworkPath, 'scores.pckl')
    if os.path.exists(scores_path):
        sys.stderr.write("Loading existing scores from %s \r\n" % scores_path)
        with open(scores_path, 'rb') as fil:
            scores = pickle.load(fil)
    else:
        files = glob.glob("%s/*.avi"%args.pycropPath)
        files.sort()
        scores = evaluate_network(files, args)
        with open(scores_path, 'wb') as fil:
            pickle.dump(scores, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

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
        sys.stderr.write("Clustering visual identities with constraints (ASD-gated embeddings)...\r\n")
        # Relax ASD gating thresholds to ensure tracks get embeddings even if ASD is conservative
        # For identity stability, cluster purely by visual similarity and constraints (no ASD gating)
        annotated_tracks = cluster_visual_identities(
            vidTracks,
            scores_list=None,
        )
        with open(identity_tracks_path, 'wb') as fil:
            pickle.dump(annotated_tracks, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Tracks with clustered identities saved in %s \r\n" % args.pyworkPath)

    # Ensure ASD scores align with current tracks; if not, recompute scores to avoid K fallback issues
    if not isinstance(scores, list) or len(scores) != len(annotated_tracks):
        files = sorted(glob.glob(os.path.join(args.pycropPath, '*.avi')))
        if not files:
            raise RuntimeError(f"No crop clips found in {args.pycropPath} to recompute ASD scores")
        sys.stderr.write("Recomputing ASD scores to align with current tracks...\n")
        scores = evaluate_network(files, args)
        with open(scores_path, 'wb') as fil:
            pickle.dump(scores, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores re-extracted and saved in %s \r\n" %args.pyworkPath)

    # Infer number of speakers (K) from ASD speaking-coverage over visual identities; fall back to existing raw diarization if needed
    # 1) Collect Person_* identities
    vis_ids = [tr.get('identity') for tr in annotated_tracks if isinstance(tr.get('identity'), str) and tr.get('identity') != 'None']
    # 2) Coverage from ASD-positive frames
    from collections import defaultdict
    id_speaking = defaultdict(int)
    for i, tr in enumerate(annotated_tracks):
        ident = tr.get('identity')
        if not (isinstance(ident, str) and ident != 'None'):
            continue
        sc = scores[i] if i < len(scores) else []
        if not isinstance(sc, (list, tuple)):
            continue
        pos = sum(1 for v in sc if float(v) > 0)
        if pos > 0:
            id_speaking[ident] += pos
    K_used = None
    if id_speaking:
        total = sum(id_speaking.values())
        items = sorted(id_speaking.items(), key=lambda x: x[1], reverse=True)
        cum = 0
        K_cov = 0
        for _, dur in items:
            cum += dur
            K_cov += 1
            if cum >= 0.90 * total:
                break
        K_used = max(1, K_cov)
    else:
        # 3) Fallback to existing raw diarization speaker count if available; else visual count
        raw_diarization_path = os.path.join(args.pyworkPath, 'raw_diriazation.pckl')
        if os.path.exists(raw_diarization_path):
            try:
                with open(raw_diarization_path, 'rb') as fil:
                    segs = pickle.load(fil)
                uniq = len(set(str(s.get('speaker')) for s in segs if isinstance(s, dict)))
                if uniq >= 1:
                    K_used = uniq
            except Exception:
                pass
        if K_used is None:
            K_used = max(1, len(set(vis_ids)))

    # 4) Run diarization constrained to K_used and overwrite cache for consistency
    raw_diarization_path = os.path.join(args.pyworkPath, 'raw_diriazation.pckl')
    sys.stderr.write(f"whisperx diarization with min/max speakers = {K_used}/{K_used}\n")
    raw_results = speech_diarization(min_speakers=K_used, max_speakers=K_used)
    with open(raw_diarization_path, 'wb') as fil:
        pickle.dump(raw_results["segments"], fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Raw diarization saved (constrained) in %s \r\n" %args.pyworkPath)

    # Match speaker identity and correct results
    matched_diarization_path = os.path.join(args.pyworkPath, 'matched_diriazation.pckl')
    # Always recompute matched diarization to reflect constrained raw results
    if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
        raise RuntimeError("Missing or invalid args.videoFps; cannot align diarization to tracks.")
    matched_results = match_speaker_identity(
        annotated_tracks, scores, raw_results["segments"], fps=float(args.videoFps)
    )
    corrected_results = autofill_and_correct_matches(matched_results)
    total_segments = len(corrected_results)
    assigned_segments = sum(1 for r in corrected_results if r['identity'] is not None and r['identity'] != 'None')
    sys.stderr.write("Speaker assignment: %d/%d segments assigned (%.1f%%)\r\n" % (
        assigned_segments, total_segments, 100.0 * assigned_segments / max(total_segments, 1)
    ))
    with open(matched_diarization_path, 'wb') as fil:
        pickle.dump(corrected_results, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Corrected diarization saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video    
    visualization(annotated_tracks, scores, corrected_results, args)    

if __name__ == '__main__':
    process_folder()
