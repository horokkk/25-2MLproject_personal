from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 작은 기본 블록들 ---------

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2fBlock(nn.Module):
    """
    YOLOv8 C2f 간단 버전.
    채널을 반으로 나눠서 여러 conv를 돌리고 concat 후 1x1 conv로 합침.
    """
    def __init__(self, in_c, out_c, n=1):
        super().__init__()
        hidden = out_c // 2
        self.cv1 = ConvBNAct(in_c, out_c, k=1, s=1, p=0)
        self.cv2 = ConvBNAct(out_c, out_c, k=1, s=1, p=0)
        self.m = nn.ModuleList([ConvBNAct(hidden, hidden) for _ in range(n)])

    def forward(self, x):
        x = self.cv1(x)
        y1, y2 = torch.chunk(x, 2, dim=1)
        ys = [y1, y2]
        for m in self.m:
            y2 = m(y2)
            ys.append(y2)
        out = torch.cat(ys, dim=1)
        # 채널 수를 다시 out_c로 맞춤
        return self.cv2(out)


# --------- 유틸 함수들 ---------

def xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_h: int, img_w: int):
    """
    boxes: [N,4] in pixel (x1,y1,x2,y2)
    return: [N,4] (cx,cy,w,h) in 0~1
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    cx /= img_w
    cy /= img_h
    w /= img_w
    h /= img_h

    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_norm_to_xyxy(boxes: torch.Tensor, img_h: int, img_w: int):
    """
    boxes: [N,4] (cx,cy,w,h) in 0~1
    return: [N,4] (x1,y1,x2,y2) pixel
    """
    cx, cy, w, h = boxes.unbind(-1)
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# --------- YOLOv8-lite 단일 스케일 ---------

class YoloV8Lite(nn.Module):
    """
    단일 스케일 YOLOv8-like detector (프로젝트용 라이트 버전)

    입력:
        images: list[Tensor 3xH xW]
        targets (train일 때만): list[dict(boxes [N,4] pixel xyxy, labels [N])]

    출력:
        train 모드: loss dict
        eval  모드(targets=None): list[dict(boxes,scores,labels)]
    """

    def __init__(self, num_classes: int, in_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.in_size = in_size  # H=W=512 가정

        # Backbone (stride: 2x2x2x2 = 16)
        self.stem = ConvBNAct(3, 32, k=3, s=2)        # 256x256
        self.stage1 = nn.Sequential(
            ConvBNAct(32, 64, k=3, s=2),              # 128x128
            C2fBlock(64, 64, n=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128, k=3, s=2),             # 64x64
            C2fBlock(128, 128, n=2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, k=3, s=2),            # 32x32 (stride 16)
            C2fBlock(256, 256, n=2),
        )

        # Head: 단일 스케일 (P3: 32x32)에서만 예측
        self.head_conv = ConvBNAct(256, 256, k=3, s=1)

        self.obj_head = nn.Conv2d(256, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(256, 4, kernel_size=1)   # (cx,cy,w,h), 0~1

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.l1 = nn.L1Loss(reduction="mean")

    # --------- forward ---------

    def forward(self, images: List[torch.Tensor], targets=None):
        """
        images: list of [3,H,W], H=W=self.in_size
        """
        x = torch.stack(images, dim=0)   # [B,3,H,W]
        B, _, H, W = x.shape

        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat = self.stage3(x)           # [B,256,32,32] if H=W=512

        B, C, Hf, Wf = feat.shape

        # Head
        h = self.head_conv(feat)
        obj_logits = self.obj_head(h)        # [B,1,Hf,Wf]
        cls_logits = self.cls_head(h)        # [B,Cc,Hf,Wf]
        box_raw = self.box_head(h)           # [B,4,Hf,Wf]

        # box는 sigmoid로 0~1로 제한
        box_pred = box_raw.sigmoid()

        if targets is None:
            return self._inference(obj_logits, cls_logits, box_pred, H, W)

        loss_dict = self._loss(
            obj_logits, cls_logits, box_pred, targets, H, W
        )
        return loss_dict

    # --------- loss ---------

    def _loss(self,
              obj_logits: torch.Tensor,
              cls_logits: torch.Tensor,
              box_pred: torch.Tensor,
              targets: List[Dict],
              img_h: int,
              img_w: int):
        """
        간단 anchor-free 라이트 버전:
          1) feature map grid center 좌표 계산 (정규화)
          2) GT box center와 가장 가까운 grid cell 하나를 positive로 지정
          3) 그 셀에 대해서:
             - obj = 1
             - cls = GT class
             - box = GT (cx,cy,w,h) normalized
          나머지는 obj = 0, cls/box는 학습 안 함(negative는 cls에 background 안 씀)
        """
        device = obj_logits.device
        B, _, Hf, Wf = obj_logits.shape

        # grid center (0~1)
        ys = (torch.arange(Hf, device=device) + 0.5) / Hf
        xs = (torch.arange(Wf, device=device) + 0.5) / Wf
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hf,Wf]

        # reshape predictions
        # obj: [B,1,Hf,Wf] -> [B,N]
        obj_logits_flat = obj_logits.view(B, -1)
        # cls: [B,Cc,Hf,Wf] -> [B,N,Cc]
        Cc = cls_logits.size(1)
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, Cc)
        # box: [B,4,Hf,Wf] -> [B,N,4]
        box_pred_flat = box_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

        N = Hf * Wf

        # target tensors
        obj_target = torch.zeros((B, N), device=device, dtype=torch.float32)
        cls_target = torch.full(
            (B, N), -100, device=device, dtype=torch.long
        )  # ignore index용
        box_target = torch.zeros((B, N, 4), device=device, dtype=torch.float32)

        num_pos = 0

        for b in range(B):
            gt_boxes = targets[b]["boxes"]  # [G,4] pixel xyxy
            gt_labels = targets[b]["labels"]  # [G]
            G = gt_boxes.size(0)

            if G == 0:
                continue

            # GT box → cx,cy,w,h (0~1)
            gt_cxcywh = xyxy_to_cxcywh_norm(gt_boxes, img_h, img_w)  # [G,4]
            gt_centers = gt_cxcywh[:, :2]                             # [G,2]

            # grid center [N,2]
            grid_centers = torch.stack(
                [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
            )  # (x,y)

            # 각 GT center와 모든 grid center 거리 계산 [G,N]
            # dist^2 = (xg - xgt)^2 + (yg - ygt)^2
            diff = grid_centers[None, :, :] - gt_centers[:, None, :]  # [G,N,2]
            dist2 = (diff ** 2).sum(dim=-1)                           # [G,N]

            # 각 GT에 대해 가장 가까운 grid index
            best_idx = dist2.argmin(dim=1)  # [G]

            for g in range(G):
                idx = best_idx[g].item()
                obj_target[b, idx] = 1.0
                cls_target[b, idx] = int(gt_labels[g].item())
                box_target[b, idx] = gt_cxcywh[g]
                num_pos += 1

        # objectness loss (모든 셀)
        loss_obj = self.bce(obj_logits_flat, obj_target)

        # classification loss (positive 셀만)
        pos_mask = cls_target != -100
        if pos_mask.any():
            cls_logits_pos = cls_logits_flat[pos_mask]      # [P, Cc]
            cls_target_pos = cls_target[pos_mask]           # [P]
            loss_cls = self.ce(cls_logits_pos, cls_target_pos)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # box regression loss (positive 셀만)
        if pos_mask.any():
            box_pred_pos = box_pred_flat[pos_mask]          # [P,4]
            box_target_pos = box_target[pos_mask]           # [P,4]
            loss_box = self.l1(box_pred_pos, box_target_pos)
        else:
            loss_box = torch.tensor(0.0, device=device)

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
        }

    # --------- inference ---------

    def _inference(self,
                   obj_logits: torch.Tensor,
                   cls_logits: torch.Tensor,
                   box_pred: torch.Tensor,
                   img_h: int,
                   img_w: int):
        """
        간단 inference:
          - obj score에 sigmoid
          - cls score는 softmax
          - obj * cls score로 점수 계산
          - threshold 넘는 것만 반환 (NMS는 생략)
        """
        device = obj_logits.device
        B, _, Hf, Wf = obj_logits.shape
        N = Hf * Wf

        obj_scores = obj_logits.sigmoid().view(B, N)             # [B,N]
        Cc = cls_logits.size(1)
        cls_scores = cls_logits.permute(0, 2, 3, 1).reshape(B, N, Cc)  # [B,N,Cc]
        cls_probs = cls_scores.softmax(-1)                        # [B,N,Cc]

        box_pred_flat = box_pred.permute(0, 2, 3, 1).reshape(B, N, 4)

        outputs = []
        for b in range(B):
            obj = obj_scores[b]          # [N]
            cls_prob = cls_probs[b]      # [N,Cc]
            box_cxcywh = box_pred_flat[b]

            # 각 셀에서 가장 높은 class 선택
            cls_score, cls_label = cls_prob.max(dim=-1)   # [N]

            scores = obj * cls_score                      # [N]
            keep = scores > 0.3                           # threshold

            if keep.sum() == 0:
                outputs.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            scores_keep = scores[keep]
            labels_keep = cls_label[keep]
            boxes_keep = cxcywh_norm_to_xyxy(
                box_cxcywh[keep], img_h, img_w
            )

            outputs.append({
                "boxes": boxes_keep,
                "scores": scores_keep,
                "labels": labels_keep,
            })

        return outputs
