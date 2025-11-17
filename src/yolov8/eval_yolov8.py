# -----------------------------------------
# eval_yolov8.py  — for YoloV8Lite ONLY
# -----------------------------------------

import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from yolov8_lite import YoloV8Lite
from dataset_yolo import BeeYoloDataset, yolo_collate_fn


# -------------------------------
# IoU
# -------------------------------
def box_iou(box1, box2):
    """
    box1: [4] x1,y1,x2,y2
    box2: [M,4]
    return: [M] IoU
    """
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


# -------------------------------
# AP 계산 (mAP50)
# -------------------------------
def compute_ap50(preds: List[Dict], gts: List[Dict], iou_thr=0.5):
    """
    preds: [{"boxes": Nx4, "scores": N, "labels": N}]
    gts:   [{"boxes": Mx4, "labels": M}]
    """
    all_scores = []
    all_matches = []

    for pred, gt in zip(preds, gts):
        p_boxes = pred["boxes"].cpu()
        p_scores = pred["scores"].cpu()
        p_labels = pred["labels"].cpu()

        g_boxes = gt["boxes"].cpu()
        g_labels = gt["labels"].cpu()

        used = torch.zeros(len(g_boxes), dtype=torch.bool)

        for pb, ps, pl in zip(p_boxes, p_scores, p_labels):
            all_scores.append(ps.item())

            # GT 중 같은 class만 비교
            mask = (g_labels == pl)
            if mask.sum() == 0:
                all_matches.append(0)
                continue

            g_candidates = g_boxes[mask]
            match_idx = torch.where(mask)[0]

            ious = box_iou(pb, g_candidates)
            max_iou, max_idx = ious.max(0)

            # 1) IoU >= threshold
            # 2) 해당 GT 아직 매칭 안 됨
            if max_iou >= iou_thr and (not used[match_idx[max_idx]]):
                used[match_idx[max_idx]] = True
                all_matches.append(1)
            else:
                all_matches.append(0)

    # 정렬 (score 내림차순)
    if len(all_scores) == 0:
        return 0.0

    scores = torch.tensor(all_scores)
    matches = torch.tensor(all_matches)
    order = scores.argsort(descending=True)
    matches = matches[order]

    tp = matches
    fp = 1 - matches
    tp_cum = torch.cumsum(tp, 0)
    fp_cum = torch.cumsum(fp, 0)

    prec = tp_cum / (tp_cum + fp_cum + 1e-6)
    rec = tp_cum / (tp.sum() + 1e-6)

    ap = torch.trapz(prec, rec).item()
    return ap


# -------------------------------
# Main evaluation for YoloV8Lite
# -------------------------------
def evaluate_yolov8_lite(model_path: str,
                         img_dir: str,
                         label_dir: str,
                         num_classes: int = 7,
                         batch_size: int = 8,
                         device="cuda"):
    device = torch.device(device)

    # Load dataset
    dataset = BeeYoloDataset(img_dir, label_dir)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2,
                        collate_fn=yolo_collate_fn)

    # Load model
    model = YoloV8Lite(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]

            preds = model(imgs, targets=None)

            # CPU로 이동
            for p in preds:
                all_preds.append({
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                })

            for t in targets:
                all_gts.append({
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                })

    mAP50 = compute_ap50(all_preds, all_gts, iou_thr=0.5)
    print(f"[YOLOv8-Lite] mAP50 = {mAP50:.4f}")

    return mAP50


# Run directly
if __name__ == "__main__":
    evaluate_yolov8_lite(
        model_path="checkpoints/yolov8_lite.pth",
        img_dir="data/images/val",
        label_dir="data/labels/val",
        num_classes=7
    )
