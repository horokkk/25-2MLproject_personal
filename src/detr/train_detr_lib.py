import os
import torch, torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import detr_resnet50, Detr_ResNet50_Weights

from dataset_detr import BeeDetrDataset, detr_collate_fn
from eval_detr_lib import evaluate_model 


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    log_interval = 100

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torchvision DETR는 train 모드에서 loss dict를 바로 반환
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            current_avg_loss = total_loss / (batch_idx + 1)
            print(
                f"[DETR-LIB] [Epoch {epoch}, "
                f"Step {batch_idx+1}/{len(dataloader)}] "
                f"current avg loss = {current_avg_loss:.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-LIB][Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss


def main():
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")
    val_img_dir = os.path.join("data", "images", "val")
    val_label_dir = os.path.join("data", "labels", "val")

    num_classes = 7  # 실제 클래스 개수
    batch_size = 8
    num_epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train / val dataset & dataloader
    train_dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detr_collate_fn,
    )

    val_dataset = BeeDetrDataset(val_img_dir, val_label_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=detr_collate_fn,
    )

    # torchvision DETR: num_classes는 "object classes 개수"
    model = detr_resnet50(weights=None, num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints_lib", exist_ok=True)

    best_mAP = 0.0

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # --- Validation & mAP 계산 ---
        print("-" * 40)
        print(f"[DETR-LIB] -> Starting Validation for Epoch {epoch}...")
        metrics = evaluate_model(model, val_loader, device)

        current_mAP = metrics["mAP"]
        print(f"[DETR-LIB][Validation] Epoch {epoch}: mAP = {current_mAP:.4f}")

        # best mAP 모델 저장
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            save_path = f"checkpoints_lib/detr_lib_best_mAP_{best_mAP:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** [DETR-LIB] NEW BEST MODEL SAVED: {save_path} ***")
        else:
            torch.save(model.state_dict(), f"checkpoints_lib/detr_lib_epoch{epoch}.pth")

        print("-" * 40)


if __name__ == "__main__":
    main()
