import torch
from torch.utils.data import Dataset

class UnnormalizeWrapper(Dataset):
    """
    기존 BeeDetrDataset을 감싸서
    Normalize 된 이미지를 다시 0~1 범위로 되돌리는 래퍼.
    (HuggingFace DetrImageProcessor에 넣기용)
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
        # BeeDetrDataset에서 쓴 mean/std와 반드시 동일하게!
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]  # img: normalized tensor

        # unnormalize: x = x*std + mean
        img_unnorm = img * self.std + self.mean
        img_unnorm = img_unnorm.clamp(0.0, 1.0)  # 혹시 모를 오차 방어

        return img_unnorm, target