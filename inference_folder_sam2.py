import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
from collections import defaultdict, Counter

import whisperx
import gc

import numpy as np
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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Demo")

# parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--videoFolder',           type=str, default=os.path.join(_THIS_DIR, '..', 'multi_human_talking_dataset'),  help='Path for inputs, tmps and outputs')
parser.add_argument('--episode_dir',           type=str, default=None, help='Episode directory; if set, overrides --videoFolder')
parser.add_argument('--pretrainModel',         type=str, default=os.path.join(_THIS_DIR, 'pretrain_TalkSet.model'),   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

args, unknown  = parser.parse_known_args()

# Allow --episode_dir to override videoFolder to match dataset usage
USE_EPISODE = bool(getattr(args, 'episode_dir', None))
if USE_EPISODE:
    args.videoFolder = args.episode_dir

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)
    
from sam2.sam2.build_sam import build_sam2_video_predictor

# Use repo-local SAM2 assets and Hydra config name (not absolute path)
sam2_checkpoint = os.path.join(_THIS_DIR, 'sam2', 'checkpoints', 'sam2.1_hiera_small.pt')
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, vos_optimized=False)

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


def load_scene_or_detect(args):
    scene_pkl = os.path.join(args.pyworkPath, 'scene.pckl')
    if os.path.isfile(scene_pkl):
        with open(scene_pkl, 'rb') as f:
            return pickle.load(f)
    return scene_detect(args)

def _load_s3fd_faces_or_fail(args):
    faces_pkl = os.path.join(args.pyworkPath, 'faces.pckl')
    if not os.path.isfile(faces_pkl):
        raise FileNotFoundError(f"Missing precomputed S3FD detections: {faces_pkl}. No fallback will be used.")
    with open(faces_pkl, 'rb') as f:
        return pickle.load(f)


def inference_video(args, scene):
    """Use precomputed S3FD detections as prompts per shot and propagate with SAM2.
    Saves per-frame detections to faces_sam2.pckl in args.pyworkPath.
    """
    torch.cuda.empty_cache()
    SAM2 = predictor

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    s3fd_faces = _load_s3fd_faces_or_fail(args)

    dets = []
    base_shot_dir = os.path.join(args.pyworkPath, 'sam2_shots')
    os.makedirs(base_shot_dir, exist_ok=True)

    for shot in scene:
        shot_start, shot_end = shot[0].get_frames(), shot[1].get_frames()
        shot_frames = flist[shot_start:shot_end + 1]

        # Materialize shot frames into a temporary directory
        shot_dir = os.path.join(base_shot_dir, f'shot_{shot_start}_{shot_end}')
        os.makedirs(shot_dir, exist_ok=True)
        for idx, frame_path in enumerate(shot_frames):
            dst = os.path.join(shot_dir, f'{idx}.jpg')
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(frame_path), dst)
                except Exception:
                    img = cv2.imread(frame_path)
                    cv2.imwrite(dst, img)

        # Initialize SAM2 with the shot frames
        inference_state = SAM2.init_state(video_path=shot_dir)

        # Find first frame in this shot with S3FD faces
        face_found = False
        bboxes = []
        first_frame_idx = None
        for rel_idx in range(len(shot_frames)):
            global_idx = shot_start + rel_idx
            frame_faces = s3fd_faces[global_idx] if global_idx < len(s3fd_faces) else []
            if len(frame_faces) > 0:
                face_found = True
                first_frame_idx = rel_idx
                bboxes = [ff['bbox'] for ff in frame_faces]
                break

        if not face_found:
            for _ in range(shot_start, shot_end + 1):
                dets.append([])
            continue

        # Add S3FD bboxes as prompts
        for ann_obj_id, bbox in enumerate(bboxes, start=1):
            box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
            SAM2.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=first_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )

        # Propagate and convert masks to bboxes per frame
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in SAM2.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for fidx in range(shot_start, shot_end + 1):
            frame_dets = []
            local_idx = fidx - shot_start
            if local_idx in video_segments:
                for obj_id, mask in video_segments[local_idx].items():
                    mask = np.asarray(mask)
                    if mask.ndim > 2:
                        mask = np.squeeze(mask)
                    y_idx, x_idx = np.where(mask)
                    if len(y_idx) > 0 and len(x_idx) > 0:
                        x_min, x_max = x_idx.min(), x_idx.max()
                        y_min, y_max = y_idx.min(), y_idx.max()
                        bbox = [x_min, y_min, x_max, y_max]
                        frame_dets.append({'frame': fidx, 'bbox': bbox, 'conf': 1.0})
            dets.append(frame_dets)
            sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(frame_dets)))

    # Save as a separate file to avoid overwriting S3FD results
    savePath = os.path.join(args.pyworkPath, 'faces_sam2.pckl')
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
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
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
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
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
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    # Fixed overlay colors per spec (BGR): green=#92d051, red=#db3a3c
    GREEN_BGR = (81, 208, 146)
    RED_BGR = (60, 58, 219)
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            color = GREEN_BGR if face['score'] >= 0 else RED_BGR
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])), color, 10)
            cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5)
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)

def speech_diarization():
    # device = "cpu"
    device = "cuda"
    audio_file = os.path.join(args.pyaviPath, "audio.wav")
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("small", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    # fill in HuggingFace Token
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="hf_REDACTED", device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    return result

def match_speaker_identity(vidTracks, scores, diarization_result, fps=25):
    matched_results = []

    # Iterate over diarization results to match each speaker segment
    for diarization in diarization_result:
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

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def generate_srt(diarization_results, output_srt_path):
    subs = pysrt.SubRipFile()

    for idx, segment in enumerate(diarization_results):
        if segment['identity'] != 'None':  # Only add segments with valid identity
            start_time = seconds_to_srt_time(segment['start_time'])
            end_time = seconds_to_srt_time(segment['end_time'])

            # Include identity in the subtitle text
            subtitle_text = f"{segment['identity']}: {segment['text']}"
            subs.append(pysrt.SubRipItem(index=idx + 1, start=start_time, end=end_time, text=subtitle_text))

    subs.save(output_srt_path, encoding='utf-8')

def visualization(tracks, scores, diarization_results, args):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        identity = track['identity']
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]  # average smoothing
            s = np.mean(s)
            faces[frame].append({
                'track': tidx,
                'score': float(s),
                'identity': identity,
                's': track['proc_track']['s'][fidx],
                'x': track['proc_track']['x'][fidx],
                'y': track['proc_track']['y'][fidx]
            })

    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh), True)
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

    # Change the bitrate settings in the command for better speed and quality
    video_with_audio_path = os.path.join(args.pyaviPath, 'video_out_with_audio.avi')
    command = ("ffmpeg -y -i %s -i %s -threads %d -b:v 8000k -c:v copy -c:a copy %s -loglevel panic" %
              (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'),
                args.nDataLoaderThread, video_with_audio_path))
    subprocess.call(command, shell=True)

    # Produce a video with subtitles using the faster bitrate setting
    video_with_subtitles_path = os.path.join(args.pyaviPath, 'video_out_with_subtitles.avi')
    command_with_subtitles = ("ffmpeg -y -i %s -vf subtitles=%s -b:v 8000k %s -loglevel panic" %
                              (video_with_audio_path, output_srt_path, video_with_subtitles_path))
    subprocess.call(command_with_subtitles, shell=True)

    # Produce a video without subtitles using the faster bitrate setting
    video_without_subtitles_path = os.path.join(args.pyaviPath, 'video_out_without_subtitles.avi')
    command_without_subtitles = ("ffmpeg -y -i %s -b:v 8000k -c copy %s -loglevel panic" %
                                (video_with_audio_path, video_without_subtitles_path))
    subprocess.call(command_without_subtitles, shell=True)

def process_folder():
    # Episode mode: use prepared assets and only generate faces_sam2.pckl then exit.
    if USE_EPISODE:
        args.savePath = args.episode_dir
        args.pyaviPath = os.path.join(args.savePath, 'avi')
        args.pyframesPath = os.path.join(args.savePath, 'frame')
        args.pyworkPath = os.path.join(args.savePath, 'result')
        args.pycropPath = os.path.join(args.savePath, 'crop')

        # Validate structure
        if not os.path.isdir(args.pyaviPath):
            raise FileNotFoundError(f"Missing avi dir: {args.pyaviPath}")
        if not os.path.isfile(os.path.join(args.pyaviPath, 'video.avi')):
            raise FileNotFoundError(f"Missing video.avi in {args.pyaviPath}")
        if not os.path.isfile(os.path.join(args.pyaviPath, 'audio.wav')):
            raise FileNotFoundError(f"Missing audio.wav in {args.pyaviPath}")
        if not os.path.isdir(args.pyframesPath):
            raise FileNotFoundError(f"Missing frame dir: {args.pyframesPath}")
        if not os.path.isdir(args.pyworkPath):
            raise FileNotFoundError(f"Missing result dir: {args.pyworkPath}")

        args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
        args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')

        scene = load_scene_or_detect(args)
        inference_video(args, scene)
        print(f"faces_sam2.pckl saved to: {os.path.join(args.pyworkPath, 'faces_sam2.pckl')}")
        return

    # Otherwise enumerate all files directly under videoFolder (legacy behavior)
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
    # If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
            (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
            (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
    
    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))    

    # Face detection for the video frames (seed with S3FD, propagate with SAM2)
    faces = inference_video(args, scene)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)

    # Example usage

    verifier = IdentityVerifier()
    annotated_tracks = verifier.assign_identities(vidTracks)
    savePath = os.path.join(args.pyworkPath, 'tracks_identity.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(annotated_tracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Tracks with identity extracted and saved in %s \r\n" %args.pyworkPath)

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi"%args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

    raw_results = speech_diarization()
    savePath = os.path.join(args.pyworkPath, 'raw_diriazation.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(raw_results["segments"], fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Raw diarization extracted and saved in %s \r\n" %args.pyworkPath)

    matched_results = match_speaker_identity(annotated_tracks, scores, raw_results["segments"], fps=25)

    corrected_results = autofill_and_correct_matches(matched_results)
    savePath = os.path.join(args.pyworkPath, 'matched_diriazation.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(corrected_results, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Corrected diarization extracted and saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video    
    visualization(annotated_tracks, scores, corrected_results, args)    

if __name__ == '__main__':
    process_folder()
