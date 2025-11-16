import os
import glob
import torch
from torch.util.data import Dataset
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

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.strip().split())
                    labels.append(int(cls))
                    boxes.append([xc, yc, w, h])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
