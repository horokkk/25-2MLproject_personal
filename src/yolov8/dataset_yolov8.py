import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BeeYoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir

        # 이미지 파일 목록
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # 기본 transform: PIL -> Tensor
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, filename + ".json")

        # --- 1. 이미지 로드 (항상 PIL) ---
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # --- 2. 라벨 파싱 (YOLO txt: cls xc yc w h) ---
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ann in data.get("annotations", []):
                cat_id = ann["category_id"]
                x1, y1, x2, y2 = ann["bbox"]  # 이미 픽셀 좌표의 xyxy

                labels.append(int(cat_id))
                boxes.append([float(x1), float(y1), float(x2), float(y2)])

        # --- 3. transform 적용 (PIL 이미지만 ToTensor 통과) ---
        if self.transforms is not None:
            # 여기서 img는 항상 PIL.Image.Image 여야 함
            img = self.transforms(img)   # -> torch.Tensor [C, H, W]

        # --- 4. target dict 구성 ---
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "orig_size": torch.tensor([img_h, img_w], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return img, target


def yolo_collate_fn(batch):
    """
    batch: list of (img, target)
    imgs: list of Tensor(C,H,W)
    targets: list of dict
    """
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)



    
