import os
import subprocess
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Speed up ONNX Runtime + CUDA initialization to reduce first-run stalls
os.environ.setdefault('ORT_CUDNN_CONV_ALGO_SEARCH', 'DEFAULT')  # avoid EXHAUSTIVE search
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')            # lazy CUDA module loading


# ------------------------------
# iResNet (from official MagFace repo, abridged for inference)
# ------------------------------

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock supports groups=1 and base_width=64 only')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in IBasicBlock')

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, num_classes=512, replace_stride_with_dilation=None):
        super(IResNet, self).__init__()
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation must have length 3')

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_classes)
        self.features = nn.BatchNorm1d(num_classes, eps=2e-05, momentum=0.9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=2e-05, momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet50(num_classes=512):
    return IResNet(IBasicBlock, [3, 4, 14, 3], num_classes=num_classes)


def iresnet100(num_classes=512):
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_classes=num_classes)


# ------------------------------
# MagFace embedder
# ------------------------------

_GDRIVE_IDS = {
    'iresnet100': '1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H',  # MS1MV2, official
    'iresnet50': '1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG',   # MS1MV2, official
}


class MagFaceEmbedder:
    def __init__(self, device: str = 'cuda', batch_size: int = 16, backbone: str = 'iresnet100', weights_path: Optional[str] = None):
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for MagFace but no GPU is available.')
        if backbone not in ('iresnet100', 'iresnet50'):
            raise RuntimeError(f'Unsupported MagFace backbone: {backbone}')

        self.device = torch.device(device if device == 'cuda' else 'cpu')
        self.batch_size = int(batch_size)
        self.backbone = backbone

        # Ensure weights
        if weights_path is None:
            weights_path = self._ensure_weights(backbone)
        if not os.path.isfile(weights_path):
            raise RuntimeError(f'MagFace weights not found: {weights_path}')

        # Build model
        if backbone == 'iresnet100':
            net = iresnet100(num_classes=512)
        else:
            net = iresnet50(num_classes=512)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        # Expect dict with key 'state_dict'
        if 'state_dict' not in state:
            raise RuntimeError('Unexpected MagFace checkpoint format: missing state_dict')
        state_dict = self._clean_state_dict(net, state['state_dict'])
        net.load_state_dict(state_dict, strict=True)
        # Force MagFace backbone to use float32 weights to avoid dtype
        # mismatches under external mixed-precision/autocast environments.
        net = net.to(dtype=torch.float32)
        self.model = net.eval().to(self.device)
        # Enable multi-GPU inference for MagFace backbone when available
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Face alignment via InsightFace landmarks
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:
            raise RuntimeError(
                'insightface is required for MagFace alignment. Install it or set IDENTITY_EMBEDDER=facenet.'
            ) from e
        # Restrict modules to essentials to avoid unnecessary model loads
        self.fa = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','landmark_2d_106','recognition'])
        ctx_id = 0 if self.device.type == 'cuda' else -1
        # Allow overriding detection input size for faster alignment via env MAGFACE_DET_SIZE (e.g., 160x160)
        det_env = os.environ.get('MAGFACE_DET_SIZE', '').strip()
        det_size = (224, 224)
        if det_env:
            try:
                if 'x' in det_env.lower():
                    w, h = det_env.lower().split('x')
                elif ',' in det_env:
                    w, h = det_env.split(',')
                else:
                    w = h = det_env
                det_size = (int(w), int(h))
            except Exception:
                det_size = (224, 224)
        self.fa.prepare(ctx_id=ctx_id, det_size=det_size)
        # ArcFace 112x112 5-point template
        self.dst5 = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def _ensure_weights(self, backbone: str) -> str:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained')
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, f'magface_{backbone}_ms1mv2.pth')
        if os.path.isfile(out_path):
            return out_path
        file_id = _GDRIVE_IDS[backbone]
        # Prefer Python gdown; if unavailable, try to install; finally try CLI/curl
        ret = 1
        try:
            import gdown  # type: ignore
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, out_path, quiet=False)
            ret = 0
        except Exception:
            # Try installing gdown
            pybin = os.environ.get('PYTHON_BIN') or 'python'
            try:
                inst = subprocess.call(f"{pybin} -m pip install --no-input gdown", shell=True)
                if inst == 0:
                    import gdown  # type: ignore
                    url = f"https://drive.google.com/uc?id={file_id}"
                    gdown.download(url, out_path, quiet=False)
                    ret = 0
            except Exception:
                pass
            if ret != 0:
                # Try CLI gdown
                from shutil import which
                if which('gdown') is not None:
                    cmd = f"gdown --id {file_id} -O {out_path}"
                    ret = subprocess.call(cmd, shell=True)
                if ret != 0:
                    # curl fallback for Google Drive large file download (best-effort)
                    cookie = os.path.join(base_dir, '.gdcookie')
                    try:
                        os.remove(cookie)
                    except Exception:
                        pass
                    get_confirm = (
                        "curl -c {cookie} -s -L 'https://drive.google.com/uc?export=download&id={fid}' "
                        "| sed -n 's/.*confirm=\\([0-9A-Za-z_]*\\).*/\\1/p'"
                    ).format(cookie=cookie, fid=file_id)
                    try:
                        confirm = subprocess.check_output(get_confirm, shell=True).decode('utf-8').strip()
                    except Exception:
                        confirm = ''
                    if not confirm:
                        dl = f"curl -L -b {cookie} 'https://drive.google.com/uc?export=download&id={file_id}' -o {out_path}"
                        ret = subprocess.call(dl, shell=True)
                    else:
                        dl = (
                            "curl -L -b {cookie} 'https://drive.google.com/uc?export=download&confirm={confirm}&id={fid}' -o {out}"
                        ).format(cookie=cookie, confirm=confirm, fid=file_id, out=out_path)
                        ret = subprocess.call(dl, shell=True)
        if ret != 0 or (not os.path.isfile(out_path)):
            raise RuntimeError('Failed to download MagFace weights (gdown/curl); please install gdown or ensure curl can access Google Drive, and network is available.')
        return out_path

    def _clean_state_dict(self, model: nn.Module, ckpt_state: dict) -> dict:
        # Adapted from MagFace inference builder: map 'features.module.xxx' or 'module.features.xxx'
        new_state = {}
        model_keys = set(model.state_dict().keys())
        for k, v in ckpt_state.items():
            parts = k.split('.')
            # Try strip leading prefixes
            if len(parts) >= 3 and parts[0] in ('features', 'module'):
                if parts[0] == 'features' and parts[1] == 'module':
                    nk = '.'.join(parts[2:])
                elif parts[0] == 'module' and parts[1] == 'features':
                    nk = '.'.join(parts[2:])
                else:
                    nk = '.'.join(parts[1:])
                if nk in model_keys and model.state_dict()[nk].size() == v.size():
                    new_state[nk] = v
                    continue
            # Fallback: try strip single leading 'module.'
            if k.startswith('module.'):
                nk = k[len('module.') :]
                if nk in model_keys and model.state_dict()[nk].size() == v.size():
                    new_state[nk] = v
                    continue
            # Direct match
            if k in model_keys and model.state_dict()[k].size() == v.size():
                new_state[k] = v
        if len(new_state) != len(model.state_dict()):
            raise RuntimeError(f'Not all MagFace weights loaded: model {len(model.state_dict())}, loaded {len(new_state)}')
        return new_state

    def _align_and_preprocess(self, frame_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return None
        # Detect landmarks on BGR image
        faces = self.fa.get(frame_bgr)
        if not faces:
            return None
        # Choose the face with largest bbox area
        def _area(face):
            box = face.bbox.astype(np.float32)  # x1,y1,x2,y2
            return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        face = max(faces, key=_area)
        if not hasattr(face, 'kps'):
            return None
        src5 = np.array(face.kps, dtype=np.float32)
        if src5.shape != (5, 2):
            return None
        # Estimate similarity transform to ArcFace template
        M, inliers = cv2.estimateAffinePartial2D(src5, self.dst5, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(frame_bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # To RGB and [0,1]
        img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img)

    def _sample_indices(self, total_frames: int, frame_indices: Optional[List[int]], max_samples: int = 15) -> List[int]:
        if total_frames <= 0:
            return []
        if frame_indices and len(frame_indices) > 0:
            idx = sorted(set(int(min(max(0, i), total_frames - 1)) for i in frame_indices))
            if len(idx) > max_samples:
                step = max(1, len(idx) // max_samples)
                idx = idx[::step][:max_samples]
            return idx
        # default spread sampling
        return [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, max(0, total_frames - 1)]

    @torch.no_grad()
    def track_embedding(self, track_file: str, active_indices: Optional[List[int]] = None) -> Optional[torch.Tensor]:
        cap = cv2.VideoCapture(track_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        positions = self._sample_indices(total_frames, active_indices)
        if not positions:
            cap.release()
            return None
        tensors = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = cap.read()
            if not ret:
                continue
            t = self._align_and_preprocess(frame)
            if t is not None:
                tensors.append(t)
        cap.release()
        if not tensors:
            return None

        feats = []
        mags = []
        # batch forward
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i : i + self.batch_size], dim=0).to(
                self.device, dtype=torch.float32
            )
            # Disable external autocast to keep conv input/weights both float32
            with torch.autocast(device_type='cuda', enabled=False):
                out = self.model(batch)  # (B, 512), BN applied
            # Use raw BN output as feature; compute magnitude
            mag = torch.norm(out, p=2, dim=1)  # (B,)
            # Normalize features for cosine usage
            out_n = F.normalize(out, p=2, dim=1)
            feats.append(out_n)
            mags.append(mag)
        Fcat = torch.cat(feats, dim=0)  # (N, 512)
        Mcat = torch.cat(mags, dim=0)   # (N,)
        if Fcat.size(0) == 0:
            return None
        # Quality-weighted average
        w = Mcat.view(-1, 1)
        emb = (Fcat * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def track_embedding_from_frames(self, frames_bgr: List[np.ndarray], active_indices: Optional[List[int]] = None, max_samples: int = 15) -> Optional[torch.Tensor]:
        if not frames_bgr:
            return None
        # Select indices
        total = len(frames_bgr)
        if active_indices and len(active_indices) > 0:
            idx = sorted(set(int(min(max(0, i), total - 1)) for i in active_indices))
            if len(idx) > max_samples:
                step = max(1, len(idx) // max_samples)
                idx = idx[::step][:max_samples]
        else:
            idx = [0, total // 4, total // 2, (3 * total) // 4, max(0, total - 1)]
        tensors = []
        for i in idx:
            frame = frames_bgr[i]
            t = self._align_and_preprocess(frame)
            if t is not None:
                tensors.append(t)
        if not tensors:
            return None
        feats = []
        mags = []
        for j in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[j : j + self.batch_size], dim=0).to(
                self.device, dtype=torch.float32
            )
            with torch.autocast(device_type='cuda', enabled=False):
                out = self.model(batch)
            mag = torch.norm(out, p=2, dim=1)
            out_n = F.normalize(out, p=2, dim=1)
            feats.append(out_n)
            mags.append(mag)
        Fcat = torch.cat(feats, dim=0)
        Mcat = torch.cat(mags, dim=0)
        if Fcat.size(0) == 0:
            return None
        w = Mcat.view(-1, 1)
        emb = (Fcat * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    @torch.no_grad()
    def frame_embeddings(
        self,
        track_file: str,
        active_indices: Optional[List[int]] = None,
        max_samples: int = 15,
        top_k: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return per-frame embeddings and quality magnitudes for a track.

        - Embeddings are L2-normalized (N, 512) float tensors on self.device
        - Magnitudes are L2 norms before normalization (N,) indicating quality
        - If top_k specified, keep the top_k frames by magnitude
        """
        cap = cv2.VideoCapture(track_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        positions = self._sample_indices(total_frames, active_indices, max_samples=max_samples)
        if not positions:
            cap.release()
            return None
        tensors = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = cap.read()
            if not ret:
                continue
            t = self._align_and_preprocess(frame)
            if t is not None:
                tensors.append(t)
        cap.release()
        if not tensors:
            return None
        feats = []
        mags = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i : i + self.batch_size], dim=0).to(
                self.device, dtype=torch.float32
            )
            with torch.autocast(device_type='cuda', enabled=False):
                out = self.model(batch)  # (B, 512)
            mag = torch.norm(out, p=2, dim=1)  # (B,)
            out_n = F.normalize(out, p=2, dim=1)
            feats.append(out_n)
            mags.append(mag)
        Fcat = torch.cat(feats, dim=0)
        Mcat = torch.cat(mags, dim=0)
        if Fcat.size(0) == 0:
            return None
        if top_k is not None and Fcat.size(0) > top_k:
            vals, idx = torch.topk(Mcat, k=top_k, largest=True)
            Fcat = Fcat[idx]
            Mcat = Mcat[idx]
        return Fcat, Mcat
