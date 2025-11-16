import os
import glob
import json
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class BeeDetrDataset(Dataset):
    """
    Bee disease dataset for DETR-lite.

    - 이미지: data/images/train/*.jpg (또는 val)
    - 라벨:   data/labels/train/*.json (또는 val)
    - JSON 형식: COCO-style (categories, image, annotations[bbox, category_id])
    """

    def __init__(self, img_dir: str, label_dir: str, transforms=None) -> None:
        self.img_paths: List[str] = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.label_dir = label_dir

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_json_label(
        self,
        label_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        JSON 라벨에서 bbox, labels 읽기.

        JSON 구조 예:
        {
          "categories": [...],
          "image": {...},
          "annotations": [
            {
              "category_id": 2,
              "bbox": [x_min, y_min, x_max, y_max],
              ...
            }, ...
          ]
        }
        """
        boxes: List[List[float]] = []
        labels: List[int] = []

        if not os.path.exists(label_path):
            # 어쩌다 라벨이 없는 경우 방어용
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            return boxes_tensor, labels_tensor

        with open(label_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            x1, y1, x2, y2 = ann["bbox"]  # 이미 픽셀 좌표의 xyxy

            labels.append(int(cat_id))
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        return boxes_tensor, labels_tensor

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # (width, height)

        # 이미지 파일명 → 같은 이름의 json 라벨
        fname = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, fname + ".json")

        boxes, labels = self._load_json_label(label_path)

        # 이미지 변환 (Resize, ToTensor, Normalize 등)
        img = self.transforms(img)
        _, new_h, new_w = img.shape

        # bbox도 리사이즈된 크기에 맞게 스케일 조정
        if boxes.numel() > 0:
            sx = new_w / orig_w
            sy = new_h / orig_h
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        target = {
            "boxes": boxes,                      # [N,4] xyxy, 512x512 기준
            "labels": labels,                    # [N]
            "image_id": torch.tensor([idx]),     # 이미지 id
            "size": torch.tensor([new_h, new_w]) # (H, W) – 원하면 loss에서 사용
        }
        return img, target


def detr_collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

