<!-- 
    title: WhisperVideo: SAM3 + WhisperX Multimodal Speaker Tracking
    description: End-to-end video understanding demo with SAM3 segmentation, WhisperX ASR, diarization, and active-speaker memory panels
    keywords: video understanding, active speaker detection, diarization, whisperx, sam3, talknet, face tracking, subtitles
    author: ShowLab
    version: 1.0.0
    last-updated: 2026-01-09
    product-type: AI Multimedia Research Code
    platforms: Linux
    technology-stack: SAM3, WhisperX, TalkNet, Pyannote, CUDA, FFmpeg
    license: Research
-->

<p align="center">
  <img alt="WhisperVideo Banner" src="assets/whispervideo_banner.png" style="width: 50%; max-width: 1200px;" />
</p>

<p align="center">
  <h1 align="center">🎬 WhisperVideo</h1>
  <p align="center"><i align="center">Visually grounded speaker transcription for long videos</i></p>
  <p align="center"><b>Track who speaks, and align speech to faces</b></p>
</p>

<h4 align="center">
  <a href="https://github.com/showlab/whispervideo">
    <img src="https://img.shields.io/github/stars/showlab/whispervideo" alt="stars" style="height: 20px;">
  </a>
  <a href="https://github.com/showlab/whispervideo/releases">
    <img src="https://img.shields.io/github/v/release/showlab/whispervideo" alt="release" style="height: 20px;">
  </a>
  <a href="https://github.com/showlab/whispervideo/issues">
    <img src="https://img.shields.io/github/issues/showlab/whispervideo" alt="issues" style="height: 20px;">
  </a>
  <a href="https://github.com/showlab/whispervideo/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/showlab/whispervideo" alt="license" style="height: 20px;">
  </a>
</h4>

## 🎙️ Overview

WhisperVideo is a clean demo for long-form, multi-speaker videos.
It links speech to on-screen speakers and keeps identities consistent.
It is built for real conversations, not short clips.

- 🔎 **SAM3 video segmentation** for robust face masks
- 🗣️ **Active speaker detection** with TalkNet (audio-visual)
- 🧠 **Identity memory** with visual embeddings and track clustering
- 📝 **Aligned subtitles** with speaker IDs and panel overlays
- 🎥 **Panel visualization** for compact review and demo videos

## ✨ Features

- [x] Visually grounded speaker attribution
- [x] Long-video friendly
- [x] Identity memory and clean speaker labels
- [x] Panel view and subtitles for review

## 🖼️ Comparison

<table>
  <tr>
    <th align="center">Model</th>
    <th align="center">Output Example</th>
  </tr>
  <tr>
    <td align="center"><b>WhisperX</b><br/>Text-only transcript</td>
    <td align="center"><img src="assets/whisperx_text.png" alt="WhisperX text-only example" style="width: 100%; max-width: 900px;" /></td>
  </tr>
  <tr>
    <td align="center"><b>WhisperVideo (Ours)</b><br/>Visually grounded panel</td>
    <td align="center"><img src="assets/whispervideo_ttbb.png" alt="WhisperVideo panel example" style="width: 100%; max-width: 900px;" /></td>
  </tr>
</table>

## 🔍 WhisperX vs WhisperVideo (Demo)

<table>
  <thead>
    <tr>
      <th align="left" style="width: 45%;">WhisperX (Plain Text)</th>
      <th align="left" style="width: 55%;">WhisperVideo (Panel + Video)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="vertical-align: top;">
        <pre>It's interesting.
Okay.
How bad is it?
Let me put it this way.
Do you own a barrel and suspenders?
Are you serious?
I'm not wearing this visor to play women's golf.
Hi.
Oh, let me guess.
You guys are drafting your fantasy accounting firms.
We're helping Raj.</pre>
      </td>
      <td style="vertical-align: top;">
        <video src="demos/ttbb/ttbb.mp4" controls muted style="max-width: 100%; border-radius: 8px;"></video>
        <div>If the video doesn't render, open <code>demos/ttbb/ttbb.mp4</code>.</div>
      </td>
    </tr>
  </tbody>
</table>

## 🧩 Install and Run

### 1. Create / use environment

We recommend using the existing environment:

```bash
/home/siyuan/miniconda3/envs/whisperv/bin/python -V
```

### 2. Optional dependencies

If you need to (re)install packages, install the core stack:

```bash
pip install torch torchvision torchaudio
pip install whisperx pyannote.audio scenedetect opencv-python python_speech_features pysrt
```

TalkNet checkpoint auto-download uses `gdown` (included in `whisperv/requirement.txt`):

```bash
pip install gdown
```

### 3. Set HF token

Create a `.env` file at repo root:

```bash
HF_TOKEN=your_huggingface_token
```

## 🚀 Quick Start

```bash
/home/siyuan/miniconda3/envs/whisperv/bin/python whisperv/inference_folder_sam3.py \
  --videoFolder demos/your_video_folder \
  --renderPanel \
  --panelTheme twitter \
  --panelCompose subtitles \
  --subtitle
```

## 📦 Outputs

The main results are written under:

```
<videoFolder>/pyavi/video_with_panel.mp4
<videoFolder>/pywork/*.pckl
```

## 📌 Notes

- The TalkNet checkpoint will auto-download if missing.
- A HuggingFace token is required for diarization.
- For best results, use a CUDA GPU.

## 🙏 Acknowledgements

- SAM3, WhisperX, TalkNet, and Pyannote: [SAM 3](https://ai.meta.com/sam3), [WhisperX](https://github.com/m-bain/whisperX), [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD), [Pyannote](https://github.com/pyannote/pyannote-audio)
- Open-source video processing tools: [FFmpeg](https://ffmpeg.org/), [SceneDetect](https://scenedetect.com/)
