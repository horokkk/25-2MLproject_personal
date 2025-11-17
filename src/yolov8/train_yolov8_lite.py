import os
import time
import torch
from torch.utils.data import DataLoader

from yolov8_lite import YoloV8Lite
from dataset_yolov8 import BeeYoloDataset, yolo_collate_fn
from eval_yolov8 import compute_ap50   # mAP 계산 함수 재사용


def evaluate_yolov8_lite(model, val_loader, device):
    """validation에서 mAP50 계산"""
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs, targets=None)

            for p in preds:
                all_preds.append({
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu(),
                })

            for t in targets:
                all_gts.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                })

    mAP50 = compute_ap50(all_preds, all_gts, iou_thr=0.5)
    return mAP50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img_dir = "data/images/train"
    train_label_dir = "data/labels/train"
    val_img_dir = "data/images/val"
    val_label_dir = "data/labels/val"

    batch_size = 8
    num_classes = 7
    num_epochs = 10

    # -------------------------
    # Dataset
    # -------------------------
    train_dataset = BeeYoloDataset(train_img_dir, train_label_dir)
    val_dataset = BeeYoloDataset(val_img_dir, val_label_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            collate_fn=yolo_collate_fn)

    # -------------------------
    # Model
    # -------------------------
    model = YoloV8Lite(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints_yololite", exist_ok=True)

    train_losses = []
    mAP50_list = []
    epoch_times = []

    best_mAP = 0.0
    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
    
    # -------------------------
    # Training loop
    # -------------------------
        for step, (imgs, targets) in enumerate(train_loader, start=1):
            imgs = [img.to(device) for img in imgs]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in t.items()}
                for t in targets
            ]

            if epoch == 1 and step <= 3:  # 앞의 몇 배치만 확인
                print(f"[DEBUG] Epoch {epoch} Step {step}")
                print("  num boxes per image:",
                      [t["boxes"].shape[0] for t in targets])

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # DETR처럼 100 step마다 출력
            if step % 100 == 0:
                avg_loss_so_far = total_loss / step
                print(
                    f"[YOLOv8-LITE][Epoch {epoch}]"
                    f"[Step {step}/{len(train_loader)}] "
                    f"avg loss = {avg_loss_so_far:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[YOLOv8-LITE][Epoch {epoch}] loss = {avg_loss:.4f}")

        # -------------------------
        # Validation (DETR 스타일이랑 비슷하게)
        # -------------------------
        print("--------------------------------------------------")
        print(f"→ Starting Validation for Epoch {epoch}...")

        model.eval()
        with torch.no_grad():
            mAP50 = evaluate_yolov8_lite(model, val_loader, device)

        mAP50_list.append(mAP50)
        print(f"[YOLOv8-LITE][Eval] mAP50 = {mAP50:.4f}")
        print(f"[YOLOv8-LITE][Epoch {epoch}] mAP50 = {mAP50:.4f}")

        # -------------------------
        # Checkpoint 저장
        # -------------------------
        if mAP50 > best_mAP:
            best_mAP = mAP50
            save_path = f"checkpoints_yololite/yolov8_lite_best_{mAP50:.4f}.pth"
        else:
            save_path = f"checkpoints_yololite/yolov8_lite_epoch{epoch}.pth"

        torch.save(model.state_dict(), save_path)

        # -------------------------
        # Epoch 시간 출력
        # -------------------------
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[YOLOv8-LITE] Epoch {epoch} time: {epoch_time:.2f}s "
            f"({epoch_time/60:.2f} min)")
        print("--------------------------------------------------")

    # -------------------------
    # 전체 학습 시간
    # -------------------------
    total_time = time.time() - total_start
    print(f"[YOLOv8-LITE] Total Training Time: {total_time/60:.2f} min")
    print("Per-epoch times:", [round(t, 2) for t in epoch_times])


if __name__ == "__main__":
    main()

