import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes: int, score_threshold: float = 0.5):
    """
    HuggingFace DetrForObjectDetection 전용 평가 함수.
    - dataloader에서 (images, targets)를 받음
      images: list of [3,512,512] tensor (이미 resize + normalize됨)
      targets: list of dict {"boxes": [N,4] (xyxy, 512 기준), "labels": [N]}
    - model: DetrForObjectDetection
    - 반환: {"mAP": float}
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    for images, targets in dataloader:
        # 1) list -> tensor [B,3,H,W]
        pixel_values = torch.stack(images).to(device)

        # 2) forward (inference 모드: labels 안 줌)
        outputs = model(pixel_values=pixel_values)

        # outputs.logits: [B, num_queries, num_classes+1]  (마지막이 no-object)
        # outputs.pred_boxes: [B, num_queries, 4] (cx, cy, w, h, 0~1 정규화)
        logits = outputs.logits.detach().cpu()
        pred_boxes = outputs.pred_boxes.detach().cpu()

        probs = logits.softmax(-1)[..., :-1]  # no-object 채널 제외
        scores, labels = probs.max(-1)       # 각 query마다 class 선택

        preds = []
        B, num_queries, _ = logits.shape
        H, W = 512, 512  # BeeDetrDataset에서 Resize(512,512) 고정이므로

        for b in range(B):
            scores_b = scores[b]    # [num_queries]
            labels_b = labels[b]    # [num_queries]
            boxes_b = pred_boxes[b] # [num_queries, 4] (cx,cy,w,h)

            # score threshold로 필터링
            keep = scores_b > score_threshold
            scores_b = scores_b[keep]
            labels_b = labels_b[keep]
            boxes_b = boxes_b[keep]

            if boxes_b.numel() == 0:
                preds.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                })
                continue

            # cx,cy,w,h (0~1) -> xyxy (픽셀 좌표)
            cx = boxes_b[:, 0] * W
            cy = boxes_b[:, 1] * H
            bw = boxes_b[:, 2] * W
            bh = boxes_b[:, 3] * H

            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            preds.append({
                "boxes": boxes_xyxy,
                "scores": scores_b,
                "labels": labels_b.to(torch.int64),
            })

        # GT 타깃도 metric 포맷으로 맞춰주기
        targets_metric = []
        for t in targets:
            targets_metric.append({
                "boxes": t["boxes"].cpu().to(torch.float32),
                "labels": t["labels"].cpu().to(torch.int64),
            })

        metric.update(preds, targets_metric)

    result = metric.compute()
    mAP = result["map"].item()
    print(f"[DETR-FB][Eval] mAP = {mAP:.4f}")
    return {"mAP": mAP}
