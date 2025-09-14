from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "/home/aiassist/siyuan/whisperV/inference_folder/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "/home/aiassist/siyuan/whisperV/inference_folder/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, vos_optimized=True)