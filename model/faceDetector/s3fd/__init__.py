import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_

_THIS_DIR = os.path.dirname(__file__)
PATH_WEIGHT = os.path.join(_THIS_DIR, 'sfd_face.pth')
if os.path.isfile(PATH_WEIGHT) == False:
    os.makedirs(_THIS_DIR, exist_ok=True)
    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
    subprocess.call(cmd, shell=True, stdout=None)
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # Use all available GPUs for face detection when possible.
        if isinstance(self.device, str):
            dev_type = self.device
        else:
            try:
                dev_type = self.device.type
            except Exception:
                dev_type = 'cpu'
        if dev_type == 'cuda' and torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes

    def detect_faces_batch(self, images, conf_th=0.8, scales=[1]):
        """Batch face detection for a list of RGB images.

        Args:
            images: list[np.ndarray HxWx3 RGB]
            conf_th: confidence threshold
            scales: list of scale factors (same meaning as single-image path)

        Returns:
            list of np.ndarray of shape (N_i, 5) per image, where columns are (x1,y1,x2,y2,score).
        """
        if not images:
            return []
        # Original unscaled sizes per image
        sizes = [(img.shape[1], img.shape[0]) for img in images]

        # Accumulate boxes per image over all scales, then NMS per image
        per_image_boxes = [np.empty((0, 5), dtype=np.float32) for _ in images]

        with torch.no_grad():
            for s in scales:
                # Preprocess batch for this scale, match single-image code exactly
                batch = []
                for img in images:
                    scaled_img = cv2.resize(img, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                    scaled_img = np.swapaxes(scaled_img, 1, 2)
                    scaled_img = np.swapaxes(scaled_img, 1, 0)
                    scaled_img = scaled_img[[2, 1, 0], :, :]
                    scaled_img = scaled_img.astype('float32')
                    scaled_img -= img_mean
                    scaled_img = scaled_img[[2, 1, 0], :, :]
                    batch.append(scaled_img)
                x = torch.from_numpy(np.stack(batch, axis=0)).to(self.device)
                y = self.net(x)
                detections = y.data  # (B, num_classes, top_k, 5)

                for b, (w, h) in enumerate(sizes):
                    scale_vec = torch.Tensor([w, h, w, h]).to(detections.device)
                    # Collect detections for all classes like single-image path
                    for i in range(detections.size(1)):
                        j = 0
                        # detections[b, i, j, 0] is score
                        while j < detections.size(2) and detections[b, i, j, 0] > conf_th:
                            score = float(detections[b, i, j, 0].item())
                            pt = (detections[b, i, j, 1:] * scale_vec).detach().cpu().numpy()
                            bbox = (pt[0], pt[1], pt[2], pt[3], score)
                            per_image_boxes[b] = np.vstack((per_image_boxes[b], bbox))
                            j += 1

        # NMS per image
        out = []
        for boxes in per_image_boxes:
            if boxes.size == 0:
                out.append(boxes)
            else:
                keep = nms_(boxes, 0.1)
                out.append(boxes[keep])
        return out
