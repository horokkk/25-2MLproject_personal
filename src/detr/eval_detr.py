from typing import Dict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@torch.no_grad()
def evaluate_model(model, dataloader, device, score_thresh: float = 0.5) -> Dict[str, float]:
    """
    torchmetrics MeanAveragePrecision으로 mAP 계산.
    DETRLite / torchvision DETR 둘 다 사용 가능.

    Args:
        model: DETR 계열 모델 (DETRLite or torchvision.models.detection.detr_resnet50 등)
        dataloader: 검증용 DataLoader (BeeDetrDataset + detr_collate_fn)
        device: torch.device
        score_thresh: confidence threshold

    Returns:
        {
          "mAP": ...,
          "mAP_50": ...,
          "mAP_75": ...,
          "mAR_100": ...,
        }
    """

    model.eval()

    # torchmetrics에서 box_format은 기본이 "xyxy"
    # 우리는 모델의 _postprocess에서 이미 pixel 기준 xyxy로 만들어주고 있음.
    metric = MeanAveragePrecision(box_format="xyxy").to(device)

    for images, targets in dataloader:
        # images: list of tensors [C, H, W]
        # targets: list of dicts {"boxes": [Ni,4], "labels": [Ni]}

        # 1) 이미지, 타깃을 device로
        images = [img.to(device) for img in images]
        targets_tm = []
        for t in targets:
            targets_tm.append({
                "boxes": t["boxes"].to(device).to(torch.float32),   # [N,4] (pixel xyxy)
                "labels": t["labels"].to(device).to(torch.int64),   # [N]
            })

        # 2) 모델 inference (targets=None → _postprocess 호출)
        outputs = model(images, targets=None)
        # outputs: list of dicts {"boxes", "scores", "labels"}

        preds_tm = []
        for out in outputs:
            boxes = out["boxes"].to(device).to(torch.float32)       # [Q,4]
            scores = out["scores"].to(device).to(torch.float32)     # [Q]
            labels = out["labels"].to(device).to(torch.int64)       # [Q]

            # confidence threshold로 필터링
            if score_thresh is not None:
                keep = scores > score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            preds_tm.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        # 3) metric 업데이트
        metric.update(preds_tm, targets_tm)

    # 4) 전체 에포크 평가 결과 계산
    map_dict = metric.compute()

    # 필요있는 값들만 추려서 반환
    result = {
        "mAP": map_dict["map"].item(),          # mAP@[0.50:0.95]
        "mAP_50": map_dict["map_50"].item(),    # IoU=0.50
        "mAP_75": map_dict["map_75"].item(),    # IoU=0.75
        "mAR_100": map_dict["mar_100"].item(),  # AR@100
    }

    print("[Eval] mAP(all)   :", result["mAP"])
    print("[Eval] mAP@0.50   :", result["mAP_50"])
    print("[Eval] mAP@0.75   :", result["mAP_75"])
    print("[Eval] mAR@100    :", result["mAR_100"])

    return result
