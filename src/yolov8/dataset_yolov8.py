import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class BeeYoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.label_dir = label_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, filename + '.txt')

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = []
        labels = []

        # YOLO txt parsing
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.strip().split())
                    
                    # ⭐ YOLO normalized → pixel xyxy 변환
                    x1 = (xc - w/2) * img_w
                    y1 = (yc - h/2) * img_h
                    x2 = (xc + w/2) * img_w
                    y2 = (yc + h/2) * img_h
                    
                    labels.append(int(cls))
                    boxes.append([x1, y1, x2, y2])  # pixel xyxy format

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),   # pixel xyxy format
            "labels": torch.tensor(labels, dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
def yolo_collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


    
