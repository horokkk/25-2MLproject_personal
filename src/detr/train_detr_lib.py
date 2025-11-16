import os
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset_detr import BeeDetrDataset, detr_collate_fn
from detr_lite import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou  # 이미 만들어둔 유틸 재사용


# ------------------ Loss (DETR-Lite 버전 재활용) ------------------ #

def detr_simple_loss(outputs, targets, num_classes: int, img_size: int = 512):
    """
    facebook DETR 출력(outputs)에 대해,
    DETR-Lite에서 쓰던 간단 IoU 매칭 + CE + L1 loss를 그대로 적용.

    outputs:
      - "pred_logits": [B,Q,num_classes+1]
      - "pred_boxes":  [B,Q,4] (cx,cy,w,h) 0~1

    targets:
      - list of dict: {"boxes": [Ni,4] (pixel xyxy, 512x512), "labels": [Ni]}
    """
    device = outputs["pred_logits"].device
    pred_logits = outputs["pred_logits"]   # [B,Q,C+1]
    pred_boxes = outputs["pred_boxes"]     # [B,Q,4] cxcywh(0~1)

    B, Q, _ = pred_logits.shape
    background_idx = num_classes  # 마지막이 background

    target_classes = torch.full((B, Q), background_idx, dtype=torch.long, device=device)
    loss_bbox = torch.tensor(0.0, device=device)
    num_pos = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)    # [G,4] pixel xyxy
        gt_labels = targets[b]["labels"].to(device)  # [G]
        G = gt_boxes.size(0)

        if G == 0:
            continue

        # GT를 0~1 정규화
        gt_boxes_norm = gt_boxes / float(img_size)  # [G,4] xyxy, 0~1

        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [Q,4] 0~1
        ious = box_iou(pred_boxes_xyxy, gt_boxes_norm)       # [Q,G]

        max_iou, best_gt_idx = ious.max(dim=1)  # [Q]

        iou_threshold = 0.3
        positive = max_iou > iou_threshold

        if positive.any():
            num_pos += positive.sum().item()

            target_classes[b, positive] = gt_labels[best_gt_idx[positive]]

            matched_gt_boxes = gt_boxes_norm[best_gt_idx[positive]]  # [Np,4] xyxy
            gt_cxcywh = box_xyxy_to_cxcywh(matched_gt_boxes)
            pred_cxcywh = pred_boxes[b, positive]

            loss_bbox += F.l1_loss(pred_cxcywh, gt_cxcywh, reduction="sum")

    if num_pos > 0:
        loss_bbox /= num_pos

    loss_cls = F.cross_entropy(
        pred_logits.view(-1, num_classes + 1),
        target_classes.view(-1)
    )

    return {
        "loss_cls": loss_cls,
        "loss_bbox": loss_bbox,
    }


# ------------------ Eval (torchmetrics mAP) ------------------ #

@torch.no_grad()
def outputs_to_predictions(outputs, images: List[torch.Tensor], score_thresh: float = 0.0):
    """
    facebook DETR outputs -> torchmetrics 형식 변환.
    images: list of [C,H,W] (이미 512x512)
    """
    pred_logits = outputs["pred_logits"]          # [B,Q,C+1]
    pred_boxes  = outputs["pred_boxes"]           # [B,Q,4] cxcywh(0~1)

    probs = pred_logits.softmax(-1)[..., :-1]     # background 제외
    scores, labels = probs.max(-1)               # [B,Q]

    boxes = box_cxcywh_to_xyxy(pred_boxes)       # [B,Q,4] 0~1

    batch_preds = []
    for i, img in enumerate(images):
        _, H, W = img.shape
        b = boxes[i].clone()
        s = scores[i]
        l = labels[i]

        # 0~1 -> pixel
        b[:, 0] *= W
        b[:, 2] *= W
        b[:, 1] *= H
        b[:, 3] *= H

        if score_thresh is not None:
            keep = s > score_thresh
            b = b[keep]
            s = s[keep]
            l = l[keep]

        batch_preds.append({
            "boxes":  b.to(torch.float32),
            "scores": s.to(torch.float32),
            "labels": l.to(torch.int64),
        })

    return batch_preds


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes, score_thresh: float = 0.0):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy").to(device)

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets_tm = []
        for t in targets:
            targets_tm.append({
                "boxes": t["boxes"].to(device).to(torch.float32),
                "labels": t["labels"].to(device).to(torch.int64),
            })

        outputs = model(images)                        # facebook DETR raw outputs
        preds = outputs_to_predictions(outputs, images, score_thresh)

        metric.update(preds, targets_tm)

    map_dict = metric.compute()
    result = {
        "mAP": map_dict["map"].item(),
        "mAP_50": map_dict["map_50"].item(),
        "mAP_75": map_dict["map_75"].item(),
        "mAR_100": map_dict["mar_100"].item(),
    }

    print("[Eval-FB] mAP(all) :", result["mAP"])
    print("[Eval-FB] mAP@0.50:", result["mAP_50"])
    print("[Eval-FB] mAP@0.75:", result["mAP_75"])
    print("[Eval-FB] mAR@100 :", result["mAR_100"])

    return result


# ------------------ Train Loop ------------------ #

def train_one_epoch(model, dataloader, optimizer, device, epoch, num_classes):
    model.train()
    total_loss = 0.0
    log_interval = 100

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)  # raw outputs dict
        loss_dict = detr_simple_loss(outputs, targets, num_classes=num_classes, img_size=512)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            cur_avg = total_loss / (batch_idx + 1)
            print(f"[DETR-FB][Epoch {epoch} Step {batch_idx+1}/{len(dataloader)}] "
                  f"loss={cur_avg:.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-FB][Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss


def main():
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")
    val_img_dir = os.path.join("data", "images", "val")
    val_label_dir = os.path.join("data", "labels", "val")

    num_classes = 7
    batch_size = 4
    num_epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=detr_collate_fn
    )

    val_dataset = BeeDetrDataset(val_img_dir, val_label_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=detr_collate_fn
    )

    # 1) facebook DETR 불러오기
    model = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet50',
        pretrained=True
    )

    # 2) class head를 우리 데이터셋에 맞게 교체
    hidden_dim = model.class_embed.in_features
    model.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 = background

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints_fb", exist_ok=True)
    best_mAP = 0.0

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, num_classes)

        print("-" * 40)
        print(f"[DETR-FB] -> Starting Validation for Epoch {epoch}...")
        metrics = evaluate_model(model, val_loader, device, num_classes, score_thresh=0.0)
        current_mAP = metrics["mAP"]
        print(f"[DETR-FB][Validation] Epoch {epoch}: mAP = {current_mAP:.4f}")

        # best mAP 기준으로 저장
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            save_path = f"checkpoints_fb/detr_fb_best_mAP_{best_mAP:.4f}.pth"
            print(f"*** [DETR-FB] NEW BEST MODEL SAVED: {save_path} ***")
        else:
            save_path = f"checkpoints_fb/detr_fb_epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        epoch_time = time.time() - epoch_start
        print(f"[DETR-FB] Epoch {epoch} time: {epoch_time:.2f} sec "
              f"({epoch_time/60:.2f} min)")
        print("-" * 40)

    total_time = time.time() - total_start
    print(f"[DETR-FB] Total training+val time: {total_time/60:.2f} min")


if __name__ == "__main__":
    main()