import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
except Exception as e:
    InceptionResnetV1 = None
    fixed_image_standardization = None


class IdentityVerifier:
    def __init__(self, device: str = "cuda", batch_size: int = 16, similarity_threshold: float = 0.6):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for IdentityVerifier but no GPU is available.")

        if InceptionResnetV1 is None:
            raise RuntimeError(
                "facenet-pytorch is required for IdentityVerifier. Install with `pip install facenet-pytorch`"
            )

        self.device = torch.device(device if device == "cuda" else "cpu")
        self.registered_ids = {}
        self.registered_embeds = {}  # id -> embedding tensor (L2-normalized)
        self.pseudo_id_counter = 0
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold

        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Enable multi-GPU inference for facenet backbone when available
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            fixed_image_standardization,
        ])

    def get_pseudo_id(self):
        self.pseudo_id_counter += 1
        return f"ID_{self.pseudo_id_counter}"

    def preprocess_frame(self, frame_bgr):
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return None
        # Convert BGR -> RGB for torchvision
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            tensor = self.tf(frame_rgb)  # 3x160x160, float
            return tensor
        except Exception:
            return None

    def get_key_frames(self, track_file, frame_indices=None, max_samples: int = 15):
        cap = cv2.VideoCapture(track_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # Determine sampling positions
        if frame_indices is not None and len(frame_indices) > 0:
            # Clamp and unique-sort
            idx = sorted(set(int(min(max(0, i), total_frames - 1)) for i in frame_indices))
            if len(idx) > max_samples:
                # Uniformly sample up to max_samples from active indices
                step = max(1, len(idx) // max_samples)
                idx = idx[::step][:max_samples]
            positions = idx
        else:
            # Default: 5 spread-out positions across the clip
            positions = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, max(0, total_frames - 1)]

        tensors = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = cap.read()
            if ret:
                tensor = self.preprocess_frame(frame)
                if tensor is not None:
                    tensors.append(tensor)
        cap.release()
        return tensors

    @torch.no_grad()
    def _embed_batch(self, batch_tensors):
        if not batch_tensors:
            return None
        x = torch.stack(batch_tensors, dim=0).to(self.device)
        embeds = self.model(x)
        embeds = F.normalize(embeds, p=2, dim=1)
        return embeds

    def _track_embedding(self, track_file, active_indices=None):
        tensors = self.get_key_frames(track_file, frame_indices=active_indices)
        if not tensors:
            return None
        # Batch through the network
        embeds_all = []
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            embeds = self._embed_batch(batch)
            if embeds is not None:
                embeds_all.append(embeds)
        if not embeds_all:
            return None
        embeds_cat = torch.cat(embeds_all, dim=0)
        emb = embeds_cat.mean(dim=0, keepdim=True)
        emb = F.normalize(emb, p=2, dim=1)  # 1 x 512
        return emb

    @torch.no_grad()
    def track_embedding_from_frames(self, frames_bgr, active_indices=None, max_samples: int = 15):
        if not frames_bgr:
            return None
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
            t = self.preprocess_frame(frame)
            if t is not None:
                tensors.append(t)
        if not tensors:
            return None
        embeds_all = []
        mags_all = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i:i + self.batch_size], dim=0).to(self.device)
            out = self.model(batch)
            mag = torch.norm(out, p=2, dim=1)
            out_n = F.normalize(out, p=2, dim=1)
            embeds_all.append(out_n)
            mags_all.append(mag)
        if not embeds_all:
            return None
        E = torch.cat(embeds_all, dim=0)
        M = torch.cat(mags_all, dim=0)
        if E.size(0) == 0:
            return None
        w = M.view(-1, 1)
        emb = (E * w).sum(dim=0, keepdim=True) / (w.sum() + 1e-8)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def frame_embeddings(self, track_file, active_indices=None, max_samples: int = 15, top_k: int = None):
        """Return per-frame embeddings (L2-normalized) and raw magnitudes.

        For facenet backend, we treat magnitudes as the L2 norm before normalization.
        Returns tuple (E, mags) where E is (N,512) on current device, mags is (N,).
        """
        tensors = self.get_key_frames(track_file, frame_indices=active_indices, max_samples=max_samples)
        if not tensors:
            return None
        feats = []
        mags = []
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            if not batch:
                continue
            x = torch.stack(batch, dim=0).to(self.device)
            out = self.model(x)  # (B,512)
            mag = torch.norm(out, p=2, dim=1)
            out_n = F.normalize(out, p=2, dim=1)
            feats.append(out_n)
            mags.append(mag)
        if not feats:
            return None
        E = torch.cat(feats, dim=0)
        M = torch.cat(mags, dim=0)
        if E.size(0) == 0:
            return None
        if top_k is not None and E.size(0) > top_k:
            vals, idx = torch.topk(M, k=top_k, largest=True)
            E = E[idx]
            M = M[idx]
        return E, M

    def verify_identity(self, track_file, registered_id):
        # Compute embedding for current track
        track_emb = self._track_embedding(track_file)
        if track_emb is None:
            return None

        ref_emb = self.registered_embeds.get(registered_id)
        if ref_emb is None:
            return None

        # Cosine similarity in [-1,1]
        sim = F.cosine_similarity(track_emb, ref_emb).item()
        if sim >= self.similarity_threshold:
            return registered_id
        return None

    def register_identity(self, track_file):
        new_id = self.get_pseudo_id()
        # Persist the crop for reference (optional)
        import shutil
        os.makedirs("cropVideo", exist_ok=True)
        destination = f"cropVideo/{new_id}.avi"
        try:
            if os.path.exists(track_file):
                shutil.copy2(track_file, destination)
        except Exception as e:
            print(f"Error copying file: {e}")

        # Store embedding
        emb = self._track_embedding(track_file)
        if emb is None:
            return None
        self.registered_embeds[new_id] = emb
        return new_id

    def is_overlapping(self, frames1, frames2):
        range1_start, range1_end = min(frames1), max(frames1)
        range2_start, range2_end = min(frames2), max(frames2)
        return not (range1_end < range2_start or range2_end < range1_start)

    def annotate_identities(self, vidTracks):
        for i, track_data in enumerate(vidTracks):
            frames = track_data["track"]["frame"]
            crop_file = track_data["cropFile"] + ".avi"

            matched_id = None
            # Try to match against existing identities that do not overlap in time
            for registered_id, registered_frame_range in self.registered_ids.items():
                if not self.is_overlapping(registered_frame_range, frames):
                    matched_id = self.verify_identity(crop_file, registered_id)
                    if matched_id:
                        break

            if matched_id:
                new_id = matched_id
            else:
                new_id = self.register_identity(crop_file)

            if new_id:
                track_data["identity"] = new_id
                self.registered_ids[new_id] = frames

        return vidTracks
