# WhisperV

> **Visually-Grounded Speaker Transcription for Long-Form Multi-Speaker Videos**<br>
> Agent Framework Integrating Visual Identity Tracking and Audio Transcription<br>

---

<a id="overview"></a>
## 📖 Overview

Transcribing long-form, multi-speaker videos (e.g., talk shows) while correctly attributing speech to the corresponding on-screen speaker (i.e., visual-grounded) remains a significant challenge. It requires the coordination of multiple capabilities such as face detection, audio transcription, and cross-modality identity assignment.

**WhisperV** is a novel agent framework that integrates specialized experts to achieve visually grounded speaker transcription for long videos.

### Three-Stage Pipeline:

**(i) Visual Identity Detection & Tracking**
- Efficient face detection and tracking using SAM3 segmentation
- Persistent speaker identities established on initial shot frames
- Segmentation propagation across temporal frames
- Face embeddings (MagFace) to differentiate multiple speakers

**(ii) Precise Audio Transcription**
- Visual-blind transcription of full audio stream using WhisperX
- High-quality speech-to-text with accurate timestamps

**(iii) Cross-Modal Matching**
- Active speaker detection (TalkNet) as bridge between modalities
- Robust alignment of visual identities with transcribed speech segments
- Temporal synchronization for speaker-transcript attribution

---

<a id="quick-start"></a>
## ⚙️ Quick Start

### 1. Environment Setup

**Requirements:**
- Python 3.10+ (use the provided conda environment)
- CUDA-capable GPU (recommended)

**Installation:**
```bash
# Use the configured environment
conda activate whisperv

# Python path
/home/siyuan/miniconda3/envs/whisperv/bin/python
```

**Dependencies:**
- PyTorch with CUDA support
- WhisperX for speech recognition
- DeepFace for face verification
- SAM3 for segmentation
- TalkNet for active speaker detection

### 2. Configure API Keys

Set required environment variables:
```bash
# HuggingFace token for diarization models
export HF_TOKEN="your_huggingface_token"
```

### 3. Run Inference

**Default (SAM3-based):**
```bash
/home/siyuan/miniconda3/envs/whisperv/bin/python inference_folder_sam3.py \
  --videoFolder /path/to/video/folder \
  --pretrainModel pretrain_TalkSet.model
```

**Standard Inference:**
```bash
/home/siyuan/miniconda3/envs/whisperv/bin/python inference_folder.py \
  --videoFolder /path/to/video/folder \
  --pretrainModel pretrain_TalkSet.model
```

---

<a id="data-layout"></a>
## 🗂️ Data Layout

**Processing Structure:**
```text
video_folder/
  video.avi                    # Input video
  audio.wav                    # Extracted audio
  pywork/
    faces.pckl                 # Detected face tracks
    scene.pckl                 # Scene detection results
    scores.pckl                # TalkNet active speaker scores
    embeddings.pckl            # Face embeddings
  pyavi/
    video_out.avi              # Output with annotations
```

---

<a id="pipeline"></a>
## 🔄 Processing Pipeline

### Stage 1: Visual Identity Detection & Tracking
1. **Shot Detection**: Segment video into coherent shots
2. **Face Detection**: Detect faces on initial frames of each shot using SAM3
3. **Segmentation Propagation**: Track face masks across temporal frames
4. **Identity Differentiation**: Extract face embeddings (MagFace) to distinguish speakers
5. **Persistent Tracking**: Maintain consistent identity assignments across shots

### Stage 2: Audio Transcription
1. **Audio Extraction**: Extract and normalize audio stream
2. **WhisperX Transcription**: Precise speech-to-text with word-level timestamps
3. **Speaker Diarization**: Segment audio by speaker turns (visual-blind)

### Stage 3: Cross-Modal Matching
1. **Active Speaker Detection**: TalkNet scores audio-visual synchronization
2. **Temporal Alignment**: Match visual face tracks with audio segments
3. **Identity Assignment**: Assign transcribed speech to corresponding visual speakers
4. **Output Generation**:
   - Annotated video with speaker labels
   - Visually-grounded subtitles (SRT format)
   - Structured results (JSON/pickle)

---

<a id="core-components"></a>
## 🧩 Core Components

**Models:**
- `model/` - TalkNet architecture components
  - `audioEncoder.py` - Audio feature extraction
  - `visualEncoder.py` - Visual feature extraction
  - `talkNetModel.py` - Audio-visual fusion model

**Identity Management:**
- `identity_verifier.py` - DeepFace-based verification
- `identity_cluster.py` - Automatic speaker clustering
- `embedders/` - Face embedding models (MagFace)

**Inference Pipelines:**
- `inference_folder_sam3.py` - SAM3-enhanced pipeline (recommended)
- `inference_folder.py` - Standard pipeline
- `talkNet.py` - TalkNet model wrapper

**Utilities:**
- `utils/` - Performance evaluation tools
- `dataLoader.py` - Data loading utilities
- `pretrain_TalkSet.model` - Pre-trained TalkNet weights (61MB)

---

<a id="configuration"></a>
## ⚙️ Configuration

**Key Parameters:**
- `--videoFolder`: Path to video directory
- `--pretrainModel`: Path to TalkNet weights
- `--nDataLoaderThread`: Number of parallel workers
- `--eval_frames`: Frames per evaluation segment

**Environment Variables:**
- `HF_TOKEN`: HuggingFace API token (required for diarization)

---

<a id="project-structure"></a>
## 📁 Project Structure

```
whisperv/
├── inference_folder_sam3.py     # Main pipeline (SAM3-based, recommended)
├── inference_folder.py          # Legacy pipeline
├── talkNet.py                   # TalkNet model wrapper
├── identity_verifier.py         # Cross-modal identity verification
├── identity_cluster.py          # Multi-speaker clustering
├── dataLoader.py                # Data utilities
├── model/                       # TalkNet architecture
│   ├── audioEncoder.py         # Stage 3: Audio features
│   ├── visualEncoder.py        # Stage 3: Visual features
│   └── talkNetModel.py         # Stage 3: Audio-visual fusion
├── sam3-main/                   # Stage 1: SAM3 segmentation
├── embedders/                   # Stage 1: Face embeddings (MagFace)
├── utils/                       # Evaluation tools
├── pretrained/                  # Model checkpoints
├── pretrain_TalkSet.model       # TalkNet weights (61MB)
└── output/                      # Processing results
```

---

<a id="notes"></a>
## 📝 Notes

- **Token Requirements**: HuggingFace token needed for WhisperX and speaker diarization models
- **SAM3-Based**: The SAM3 pipeline (`inference_folder_sam3.py`) is the recommended and default method
- **Applications**: Automatic transcription for online meetings, talk shows, interviews, podcasts, etc.

---

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

- **TalkNet**: Active speaker detection framework
- **WhisperX**: Speech recognition and alignment
- **SAM3**: Segment Anything Model 3 for visual segmentation
- **DeepFace**: Face recognition and verification
- **Pyannote**: Speaker diarization toolkit
