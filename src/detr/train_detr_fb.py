import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from dataset_detr import BeeDetrDataset, detr_collate_fn
from eval_detr import evaluate_model


# -----------------------------------------
# Simple DETR Loss (Lite에서 쓰던 것 그대로 재사용)
# -----------------------------------------
from detr_lite import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou

def simple_detr_loss(outputs, targets, num_classes):
    """
    손실 구성:
    - IoU 기반 simple matching (Lite와 동일)
    - CrossEntropy(class)
    - L1 bbox
    """
    pred_logits = outputs["pred_logits"]     # [B, Q, num_classes+1]
    pred_boxes  = outputs["pred_boxes"]      # [B, Q, 4]

    device = pred_logits.device
    B, Q, _ = pred_logits.shape

    # 모든 query를 background로 초기화
    target_classes = torch.full((B, Q), num_classes, dtype=torch.long, device=device)

    loss_bbox = torch.tensor(0.0, device=device)
    num_pos = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"]      # [G, 4] xyxy
        gt_labels = targets[b]["labels"]    # [G]

        if gt_boxes.numel() == 0:
            continue

        # normalize 0~1
        img_size = 512  # dataset에서 resize = 512로 맞춰서 고정됨
        gt_norm = gt_boxes / img_size

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [Q,4], 0~1
        ious = box_iou(pred_xyxy, gt_norm)             # [Q,G]

        max_iou, best_gt = ious.max(dim=1)

        pos_mask = max_iou > 0.3

        if pos_mask.any():
            num_pos += pos_mask.sum().item()

            target_classes[b, pos_mask] = gt_labels[best_gt[pos_mask]]

            matched_gt_boxes = gt_norm[best_gt[pos_mask]]
            gt_cxcywh = box_xyxy_to_cxcywh(matched_gt_boxes)

            pred_cxcywh = pred_boxes[b, pos_mask]
            loss_bbox += torch.nn.functional.l1_loss(pred_cxcywh, gt_cxcywh, reduction="sum")

    if num_pos > 0:
        loss_bbox /= num_pos

    loss_cls = torch.nn.functional.cross_entropy(
        pred_logits.view(-1, num_classes + 1),
        target_classes.view(-1)
    )

    return loss_cls + loss_bbox



# -----------------------------------------
# Training Loop
# -----------------------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch, num_classes):
    model.train()
    total_loss = 0.0
    log_interval = 100

    for step, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss = simple_detr_loss(outputs, targets, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            print(f"[FB-DETR][Epoch {epoch}][Step {step+1}] avg_loss={total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[FB-DETR][Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss



# -----------------------------------------
# Main
# -----------------------------------------
def main():
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")
    val_img_dir = os.path.join("data", "images", "val")
    val_label_dir = os.path.join("data", "labels", "val")

    num_classes = 7
    batch_size = 8
    num_epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = BeeDetrDataset(train_img_dir, train_label_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=detr_collate_fn)

    val_ds = BeeDetrDataset(val_img_dir, val_label_dir)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=detr_collate_fn)

    # -----------------------------------------
    # Facebook original DETR 불러오기 + class head 교체
    # -----------------------------------------
    model = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet50',
        pretrained=True
    )

    hidden_dim = model.class_embed.in_features
    model.class_embed = nn.Linear(hidden_dim, num_classes + 1)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints_fb", exist_ok=True)

    # 시간 측정
    total_start = time.time()
    epoch_times = []
    best_mAP = 0.0

    # -----------------------------------------
    # Epoch Loop (Lite랑 완전 동일 구조)
    # -----------------------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, num_classes)

        print("-" * 40)
        print(f"-> Starting Validation for Epoch {epoch}...")
        metrics = evaluate_model(model, val_loader, device, num_classes)

        current_mAP = metrics["mAP"]
        print(f"[FB-DETR] Epoch {epoch}: mAP = {current_mAP:.4f}")

        if current_mAP > best_mAP:
            best_mAP = current_mAP
            save_path = f"checkpoints_fb/fb_detr_best_{best_mAP:.4f}.pth"
        else:
            save_path = f"checkpoints_fb/fb_detr_epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        # 시간 출력
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[FB-DETR] Epoch {epoch} time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)")
        print("-" * 40)

    total_time = time.time() - total_start
    print(f"[FB-DETR] Total training time: {total_time/60:.2f} min")
    print("Epoch times:", [round(t, 2) for t in epoch_times])


if __name__ == "__main__":
    main()
