import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_model(model, dataloader, processor, device):
    """
    HuggingFace DetrForObjectDetection 전용 평가 함수.
    - dataloader: (images, targets)
    - images: list of [3,512,512] tensor
    - targets: list of {"boxes":xyxy, "labels":int}

    반환:
        {"mAP": float}
    """

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    for images, targets in dataloader:
        # -------------------------------------------------------
        # 1) list of tensors → (B,3,H,W) tensor
        # -------------------------------------------------------
        pixel_values = torch.stack(images).to(device)

        # -------------------------------------------------------
        # 2) DETR forward (labels=None)
        # -------------------------------------------------------
        outputs = model(pixel_values=pixel_values)

        # -------------------------------------------------------
        # 3) HF 공식 post-process 사용 (핵심)
        # -------------------------------------------------------
        # BeeDetrDataset은 Resize(512,512) 적용하므로 고정
        target_sizes = torch.tensor(
            [[512, 512]] * pixel_values.shape[0], 
            device=device
        )

        processed = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes
        )

        # -------------------------------------------------------
        # 4) metric 입력 형식으로 변환
        # -------------------------------------------------------
        preds = []
        for p in processed:
            preds.append({
                "boxes": p["boxes"].cpu(),               # xyxy(pixel)
                "scores": p["scores"].cpu(),             # float
                "labels": p["labels"].cpu().to(torch.int64),
            })

        # GT도 metric 형식으로 맞추기
        targets_metric = []
        for t in targets:
            targets_metric.append({
                "boxes": t["boxes"].cpu().to(torch.float32),
                "labels": t["labels"].cpu().to(torch.int64),
            })

        metric.update(preds, targets_metric)

    # -------------------------------------------------------
    # 5) 최종 mAP 계산
    # -------------------------------------------------------
    result = metric.compute()
    mAP = result["map"].item()

    print(f"[DETR-FB][Eval] mAP = {mAP:.4f}")

    return {"mAP": mAP}


