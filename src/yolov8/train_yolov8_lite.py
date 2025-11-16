import os
import torch
from torch.utils.data import DataLoader

from yolov8_lite import YoloV8Lite
from dataset_yolo import BeeYoloDataset, yolo_collate_fn  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img_dir = "data/images/train"
    train_label_dir = "data/labels/train"

    dataset = BeeYoloDataset(train_img_dir, train_label_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=4, collate_fn=yolo_collate_fn)

    model = YoloV8Lite(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[YOLOv8-LITE][Epoch {epoch}] loss = {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "checkpoints/yolov8_lite.pth")


if __name__ == "__main__":
    main()
