import os

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import detr_resnet50

from dataset_detr import BeeDetrDataset, detr_collate_fn


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torchvision DETR는 train 모드에서 loss dict를 바로 반환
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-LIB][Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss


def main():
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")

    num_classes = 3  # 실제 클래스 개수
    batch_size = 2
    num_epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detr_collate_fn,
    )

    # torchvision DETR: num_classes는 "object classes 개수"
    model = detr_resnet50(weights=None, num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints_lib", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        torch.save(model.state_dict(), f"checkpoints_lib/detr_lib_epoch{epoch}.pth")


if __name__ == "__main__":
    main()
