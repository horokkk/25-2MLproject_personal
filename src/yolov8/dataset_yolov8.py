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
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
        ])
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

        # 3) 이미지 변환 (리사이즈 + 텐서화)
        img = self.transforms(img)          # -> [C, H, W]
        _, new_h, new_w = img.shape        # 리사이즈 후 크기

        # 4) bbox도 리사이즈된 크기에 맞게 스케일 조정
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

            sx = new_w / orig_w            # 가로 스케일 비율
            sy = new_h / orig_h            # 세로 스케일 비율

            # x좌표(0,2)에 sx, y좌표(1,3)에 sy 곱해주기
            boxes_tensor[:, [0, 2]] *= sx
            boxes_tensor[:, [1, 3]] *= sy

            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,                          # 리사이즈된 이미지 기준 xyxy
            "labels": labels_tensor,
            "orig_size": torch.tensor([orig_h, orig_w]),    # 원본
            "size": torch.tensor([new_h, new_w]),           # 리사이즈 후
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



    
