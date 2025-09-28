import os
from sam2.build_sam import build_sam2_video_predictor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sam2_checkpoint = os.path.join(_THIS_DIR, 'checkpoints', 'sam2.1_hiera_small.pt')
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, vos_optimized=True)
