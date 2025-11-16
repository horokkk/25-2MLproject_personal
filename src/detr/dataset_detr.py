# src/dataset_detr.py
import os
import glob
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class BeeDetrDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.label_dir = label_dir

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def _load_yolo_label(self,
                         label_path: str,
                         img_w: int,
                         img_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """YOLO txt(cls xc yc w h, 0~1) -> (boxes_xyxy[pixel], labels)."""
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls, xc, yc, bw, bh = map(float, line.split())
                    labels.append(int(cls))

                    xc *= img_w
                    yc *= img_h
                    bw *= img_w
                    bh *= img_h

                    x1 = xc - bw / 2.0
                    y1 = yc - bh / 2.0
                    x2 = xc + bw / 2.0
                    y2 = yc + bh / 2.0
                    boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        return boxes_tensor, labels_tensor

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        fname = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, fname + ".txt")

        boxes, labels = self._load_yolo_label(label_path, orig_w, orig_h)

        img = self.transforms(img)
        _, new_h, new_w = img.shape
        sx = new_w / orig_w
        sy = new_h / orig_h
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return img, target


def detr_collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)
