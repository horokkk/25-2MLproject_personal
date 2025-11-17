import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from dataset_detr import BeeDetrDataset, detr_collate_fn
from eval_detr import evaluate_model


# ---------------------------------------
# Train 1 epoch
# ---------------------------------------
def train_one_epoch(model, dataloader, optimizer, processor, device, epoch):
    model.train()
    total_loss = 0.0
    log_interval = 100  # every 100 steps

    for batch_idx, (images, targets) in enumerate(dataloader):
        # images: [img1, img2, ...]
        # targets: list of dict — {"boxes":[N,4], "labels":[N]}

        # 1) processor가 원하는 형태로 변환
        pixel_values = processor(
            images=[img for img in images],
            return_tensors="pt"
        )["pixel_values"].to(device)

        # 2) target 변환 (HF DETR은 COCO format 요구)
        new_targets = []
        for t in targets:
            new_targets.append({
                "class_labels": t["labels"].to(device),
                "boxes": t["boxes"].to(device)
            })

        # 3) forward
        outputs = model(pixel_values=pixel_values, labels=new_targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print(f"[DETR-FB][Epoch {epoch}][Step {batch_idx+1}/{len(dataloader)}] "
                  f"avg loss = {total_loss / (batch_idx+1):.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-FB][Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss


# ---------------------------------------
# Main training loop
# ---------------------------------------
def main():
    # Dataset directories
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")
    val_img_dir = os.path.join("data", "images", "val")
    val_label_dir = os.path.join("data", "labels", "val")

    num_classes = 7  # 너 데이터 기준
    batch_size = 4   # DETR은 메모리 크므로 batch=4 권장
    num_epochs = 5
    lr = 1e-5  # HF DETR 권장 learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    train_dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=detr_collate_fn, num_workers=2
    )

    val_dataset = BeeDetrDataset(val_img_dir, val_label_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=detr_collate_fn, num_workers=2
    )

    # ---------------------------------------
    # Load original DETR (ResNet-50)
    # ---------------------------------------
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # classifier head 재설정
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints_fb", exist_ok=True)
    best_mAP = 0.0
    epoch_times = []

    total_start = time.time()

    # ---------------------------------------
    # Training loop
    # ---------------------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, processor, device, epoch)

        # Validation (mAP)
        print("-" * 50)
        print(f"→ Starting Validation for Epoch {epoch}...")

        metrics = evaluate_model(model, val_loader, device, num_classes)
        current_mAP = metrics["mAP"]

        print(f"[DETR-FB][Epoch {epoch}] mAP = {current_mAP:.4f}")

        # Save
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            save_path = f"checkpoints_fb/detr_fb_best_{best_mAP:.4f}.pth"
        else:
            save_path = f"checkpoints_fb/detr_fb_epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        # epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[DETR-FB] Epoch {epoch} time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        print("-" * 50)

    total_time = time.time() - total_start
    print(f"[DETR-FB] Total Training Time: {total_time/60:.2f} min")
    print("Per-epoch times:", [round(t, 2) for t in epoch_times])


if __name__ == "__main__":
    main()

