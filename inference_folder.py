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
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = numpy.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
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
    # Use specified brand colors (OpenCV expects BGR)
    GREEN_BGR = (81, 208, 146)  # #92d051
    RED_BGR = (60, 58, 219)     # #db3a3c
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            color = GREEN_BGR if face['score'] >= 0 else RED_BGR
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])), color, 10)
            cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color,5)
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
    model = whisperx.load_model("small", device, compute_type=compute_type)

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

def generate_srt(diarization_results, output_srt_path):
    subs = pysrt.SubRipFile()

    # Use explicit SubRipTime construction to avoid parser edge-cases
    for idx, segment in enumerate(diarization_results):
        identity = segment.get('identity')
        if identity is None or identity == 'None':
            continue

        st = float(segment.get('start_time', 0.0))
        et = float(segment.get('end_time', st))
        text = segment.get('text', '')

        start_ms = int(round(st * 1000.0))
        end_ms = int(round(et * 1000.0))

        start_time = pysrt.SubRipTime(milliseconds=start_ms)
        end_time = pysrt.SubRipTime(milliseconds=end_ms)

        subtitle_text = f"{identity}: {text}"
        subs.append(pysrt.SubRipItem(index=idx + 1, start=start_time, end=end_time, text=subtitle_text))

    subs.save(output_srt_path, encoding='utf-8')

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
    # Fixed overlay colors per spec (BGR): green=#92d051, red=#db3a3c
    GREEN_BGR = (81, 208, 146)
    RED_BGR = (60, 58, 219)

    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)

        for face in faces[fidx]:
            color = GREEN_BGR if face['score'] >= 0 else RED_BGR
            txt = f"{face['identity']}: {round(face['score'], 1)}"  # Format as "ID_1: 2.5"
            bbox_x = int(face['x'] - face['s'])
            bbox_y = int(face['y'] - face['s'])
            bbox_x = max(0, min(bbox_x, fw - 1))
            bbox_y = max(0, min(bbox_y, fh - 1))

            # Draw bounding box
            cv2.rectangle(image,
                          (bbox_x, bbox_y),
                          (int(face['x'] + face['s']), int(face['y'] + face['s'])),
                          color, 5)

            # Smaller font for ID and score on the same line
            cv2.putText(image, txt,
                        (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 3)

        vOut.write(image)

    vOut.release()

    # Generate SRT file for subtitles
    output_srt_path = os.path.join(args.pyaviPath, 'subtitles.srt')
    generate_srt(diarization_results, output_srt_path)

    # Optimize audio/video merging with stream copying
    video_with_audio_path = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
              (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, video_with_audio_path))
    subprocess.call(command, shell=True)

    # Produce a video with subtitles using fast encoding
    video_with_subtitles_path = os.path.join(args.pyaviPath, 'video_out_with_subtitles.avi')
    command_with_subtitles = ("ffmpeg -y -i %s -vf subtitles=%s -c:v libx264 -preset ultrafast -crf 23 %s -loglevel panic" %
                              (video_with_audio_path, output_srt_path, video_with_subtitles_path))
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

    # Speech diarization
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

    # Match speaker identity and correct results
    matched_diarization_path = os.path.join(args.pyworkPath, 'matched_diriazation.pckl')
    if os.path.exists(matched_diarization_path):
        sys.stderr.write("Loading existing matched diarization from %s \r\n" % matched_diarization_path)
        with open(matched_diarization_path, 'rb') as fil:
            corrected_results = pickle.load(fil)
    else:
        # Simple speaker-identity matching with correctly aligned FPS
        if not hasattr(args, 'videoFps') or args.videoFps is None or args.videoFps <= 0:
            raise RuntimeError("Missing or invalid args.videoFps; cannot align diarization to tracks.")
        matched_results = match_speaker_identity(
            annotated_tracks, scores, raw_results["segments"], fps=float(args.videoFps)
        )
        
        corrected_results = autofill_and_correct_matches(matched_results)
        
        # Speaker clustering has been removed from the cleaned pipeline.
        # If grouping is needed, perform it externally during evaluation/analysis.
        
        # Log speaker assignment statistics
        total_segments = len(corrected_results)
        assigned_segments = sum(1 for r in corrected_results if r['identity'] is not None and r['identity'] != 'None')
        
        sys.stderr.write("Speaker assignment: %d/%d segments assigned (%.1f%%)\r\n" % (
            assigned_segments, total_segments, 100.0 * assigned_segments / max(total_segments, 1)
        ))
        with open(matched_diarization_path, 'wb') as fil:
            pickle.dump(corrected_results, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Corrected diarization extracted and saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video    
    visualization(annotated_tracks, scores, corrected_results, args)    

if __name__ == '__main__':
    process_folder()
