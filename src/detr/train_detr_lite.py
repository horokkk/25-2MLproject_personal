import os
import torch
from torch.utils.dataset import DataLoader

from dataset_detr import BeeDetrDataset, detr_collate_fn
from detr_lite import DETRLite

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-Lite] [Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss

def main():
    # 경로, 클래스 수 수정 필요
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")

    num_classes = 7  # {'id': 0, 'name': '유충_정상'},{'id': 1, 'name': '유충_응애'},{'id': 2, 'name': '유충_석고병'},{'id': 3, 'name': '유충_부저병'},{'id': 4, 'name': '성충_정상'},{'id': 5, 'name': '성충_응애'},{'id': 6, 'name': '성충_날개불구바이러스감염증'}
    batch_size = 2
    num_epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=detr_collate_fn)

    model = DETRLite(num_classes=num_classes, num_queries=100)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        torch.save(model.state_dict(), f"checkpoints/detr_lite_epoch{epoch}.pth")

if __name__ == "__main__":
    main()

