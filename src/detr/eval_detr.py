from typing import Dict, List, Any
import torch
import numpy as np
if not hasattr(np, "float"):
    np.float = float

from torchmetrics.detection.mean_ap import MeanAveragePrecision


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    score_thresh: float = 0.5,               # 조정 필요 -> 0.1
    fb_mode: bool = False   # ★ DETR-Lite=False, FB DETR=True
) -> Dict[str, float]:
    """
    공통 mAP 평가 함수
    - DETR-Lite: fb_mode=False
    - Facebook DETR (HuggingFace): fb_mode=True
    - torchvision DETR: 자동 try/except 경로 (기존 코드 구조 존재)
    """

    model.eval()

    metric = MeanAveragePrecision(box_format="xyxy").to(device)

    for images, targets in dataloader:
        images_dev = [img.to(device) for img in images]

        # -----------------------------
        # 1) GT 준비 (공통)
        # -----------------------------
        targets_tm = []
        for t in targets:
            targets_tm.append({
                "boxes": t["boxes"].to(device).to(torch.float32),
                "labels": t["labels"].to(device).to(torch.int64),
            })

        # -----------------------------
        # 2) 모델 inference
        # -----------------------------
        if fb_mode:
            # ---- Facebook DETR(HF) ----
            pixel_values = torch.stack(images).to(device)  # [B,3,H,W]
            outputs = model(pixel_values=pixel_values)

            logits = outputs.logits        # [B,Q,num_cls+1]
            pred_boxes = outputs.pred_boxes  # [B,Q,4] cxcywh 0~1

            probs = logits.softmax(-1)[..., :-1]
            scores, labels = probs.max(-1)

            preds_tm = []
            B, Q, _ = pred_boxes.shape

            for i in range(B):
                _, H, W = images[i].shape

                cx, cy, w, h = pred_boxes[i].unbind(-1)
                cx, w = cx * W, w * W
                cy, h = cy * H, h * H

                x1, y1 = cx - 0.5 * w, cy - 0.5 * h
                x2, y2 = cx + 0.5 * w, cy + 0.5 * h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                scores_i = scores[i]
                labels_i = labels[i]

                keep = scores_i > score_thresh
                preds_tm.append({
                    "boxes": boxes_xyxy[keep].to(device).to(torch.float32),
                    "scores": scores_i[keep].to(device).to(torch.float32),
                    "labels": labels_i[keep].to(device).to(torch.int64),
                })

        else:
            # ---- DETR-Lite or torchvision DETR ----
            try:
                outputs = model(images_dev, targets=None)
            except TypeError:
                outputs = model(images_dev)

            preds_tm = []
            for out in outputs:
                boxes = out["boxes"].to(device).to(torch.float32)
                scores = out["scores"].to(device).to(torch.float32)
                labels = out["labels"].to(device).to(torch.int64)

                keep = scores > score_thresh
                preds_tm.append({
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                })

        # -----------------------------
        # 3) metric 업데이트
        # -----------------------------
        metric.update(preds_tm, targets_tm)

    # -----------------------------
    # 4) 전체 결과 계산
    # -----------------------------
    map_dict = metric.compute()

    result = {
        "mAP": map_dict["map"].item(),
        "mAP_50": map_dict["map_50"].item(),
        "mAP_75": map_dict["map_75"].item(),
        "mAR_100": map_dict["mar_100"].item(),
    }

    print("[Eval] mAP(all) :", result["mAP"])
    print("[Eval] mAP_50   :", result["mAP_50"])
    print("[Eval] mAP_75   :", result["mAP_75"])
    print("[Eval] mAR_100  :", result["mAR_100"])

    return result
