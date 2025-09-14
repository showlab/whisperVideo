# Parallel Video Inference

This provides a simple top-level parallelization for processing multiple videos across 8 GPUs.

## Usage

### Basic Usage
```bash
# Process all videos in /workspace/siyuan/siyuan/whisperv_proj/data/video using 8 GPUs
cd /workspace/siyuan/siyuan/whisperv_proj/whisperV/inference_folder
python parallel_runner.py --video_root /workspace/siyuan/siyuan/whisperv_proj/data/video
```

### Custom GPU Count
```bash
# Use only 4 GPUs
python parallel_runner.py --video_root /workspace/siyuan/siyuan/whisperv_proj/data/video --num_gpus 4
```

### Specific Series
```bash
# Process only Frasier videos
python parallel_runner.py --video_root /workspace/siyuan/siyuan/whisperv_proj/data/video/Frasier
```

### Custom Script
```bash
# Use different inference script
python parallel_runner.py --video_root /workspace/siyuan/siyuan/whisperv_proj/data/video --script inference_folder_sam2.py
```

## How It Works

1. **Discovers videos**: Finds all `.mp4`, `.avi`, `.mov`, `.mkv` files in the specified directory
2. **GPU assignment**: Assigns each video to a GPU using round-robin (video 0→GPU 0, video 1→GPU 1, ..., video 8→GPU 0, etc.)
3. **Parallel execution**: Uses `CUDA_VISIBLE_DEVICES` to isolate GPU access per process
4. **Output preservation**: Each video creates its output folder in the same location as the original script behavior

## Output Structure

For each video like `/data/video/Frasier/Frasier_02x01.mp4`, the output will be created at:
```
/data/video/Frasier/Frasier_02x01/
├── pyavi/           # Audio/video files
├── pyframes/        # Extracted frames  
├── pywork/          # Intermediate results (pickle files)
├── pycrop/          # Face crops
└── ...              # Other outputs
```

This matches the exact same folder structure as running the original script.

## Advantages

- ✅ **Simple**: No complex modifications to existing inference code
- ✅ **Isolated**: Each process uses only its assigned GPU via `CUDA_VISIBLE_DEVICES`  
- ✅ **Compatible**: Same output format and folder structure as original
- ✅ **Scalable**: Easy to adjust number of GPUs used
- ✅ **Robust**: Each video processed independently, failures don't affect others