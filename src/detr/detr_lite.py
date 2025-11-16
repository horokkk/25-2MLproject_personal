from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbinde(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def box_iou(boxes1, boxes2):
    """
    boxes1: [N,4], boxes2: [M,4]  (xyxy, 0~1 범위)
    return: [N,M] IoU
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)
    
    #intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])        

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h # [N,M]

    #areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)  # [M]  

    union = area1[:, None] + area2 - inter  # [N,M]
    iou = inter / union.clamp(min=1e-6)
    return iou

class DETRLite(nn.Module):
    """
    ML 프로젝트용 Lite DETR 모델:
    - Backbone: ResNet-50
    - Transformer: encoder / decoder 각각 3개 layer
    - Object queries: num_queries개
    - head: class + box
    - loss: greedy IoU matching + CE + L1
    """
    def __init__(self,
                 num_classes: int,
                 num_queries: int = 100,
                 hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes  # 실제 클래스 개수
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.background_idx = num_classes  # 마지막 로짓을 background로 사용

        # 1) Backbone: ResNet-50 마지막 conv layer까지 사용 (Backbone + 1x1 conv로 transforemr에 넣을 feature 차원(hidden_dim) 맞추기))
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # [B,2048,Hf,Wf]
        self.conv.proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)  # [B,hidden_dim,Hf,Wf]

        # 2) Positional embedding (row/col embedding) / DETR 논문은 sine-cosine 기반 -> 학습 가능한 2D 위치 임베딩으로 단순화
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)

        # 3) Transformer encoder / decoder 
        # CNN에서 flatten한 feature sequence를 입력으로 받아 context-aware한 feature로 변환
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        # object query들을 넣어서 각 물체에 해당하는 표현으로 바꿔줌
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)

        # 4) Object queries (학습가능한 임베딩)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 5) Prediction heads (Head(class, box))
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.box_head = nn.Linear(hidden_dim, 4)  # (cx,cy,w,h) 0~1 범위

        nn.init.xavier_uniform_(self.conv.proj.weight, gain=1.0)

    def _positional_encoding(self, H: int, W: int, device):
        i = torch.arange(W, device=device)
        j = torch.arange(H, device=device)
        x_emb = self.col_embed(i)  # [W, C/2]
        y_emb = self.row_embed(j)  # [H, C/2]

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1)
        ], dim=-1)  # [H, W, C]
        return pos.permute(2, 0, 1)  # [C, H, W]
    
    # backbone -> transformer -> head
    # targets 있으면 loss_dict, 없으면 postprocess된 결과 리턴
    def forward(self, images: List[torch.Tensor], targets=None):
        device = images[0].device
        x = torch.stack(images, dim=0)  # [B,3,H,W], H=W=512 가정

        # Backbone+projection+positional encoding
        feats = self.backbone(x) # [B,2048,Hf,Wf]
        feats = self.conv.proj(feats)  # [B,C, Hf,Wf]
        B, C, Hf, Wf = feats.shape

        pos = self._positional_encoding(Hf, Wf, device)  # [C,Hf,Wf]
        feats = feats + pos.unsqueeze(0)  # [B,C,Hf,Wf]

        # Transformer 입력 형태로 바꾸기
        # [B, C, Hf * Wf] -> [HW * Wf, B, C]
        src = feats.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src)  # [HW, B, C]

        # Decoder (query 활용)
        # queries: [num_queries, B, C]
        query_embed = self.query_embed.weight #[Q, C]
        tgt = torch.zeros_like(query_embed) # [Q, C]
        tgt = tgt.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]

        hs = self.decoder(tgt, memory, query_pos=query_embed)  # [Q, B, C]
        hs = hs.transpose(0, 1) # [B, Q, C] 각 object query에 해당하는 최종 표현

        # Head로 class, box 예측
        pred_logits = self.class_head(hs) #[B, Q, num_classes+1]
        pred_boxes = self.box_head(hs).sigmoid()  # [B, Q, 4] (cx,cy,w,h) 0~1 기준

        if targets in None:
            return self._postprocess(pred_logits, pred_boxes, images)

        loss_dict = self._loss(pred_logits, pred_boxes, targets)
        return loss_dict
    
    # ----------------- Loss ------------------

    # 간단 IoU 매칭 + CE + L1
    def _loss(self, pred_logits, pred_boxes, targets):
        """
        간단 버전:
        - 각 query는 하나의 GT와 IoU로 매칭 (greedy X, 그냥 argmax 기준)
        - IoU > 0.3인 query만 positive, 나머지는 background로 처리
        - CE(class) + L1(box) 사용해서 계산
        """
        device = pred_logits.device
        B, Q, _ = pred_logits.shape

        # 1) classification target 초기화: 모두 background로
        target_classes = torch.full((B, Q), self.background_idx, dtype=torch.long, device=device)

    loss_bbox = torch.tensor(0.0, device=device)
    num_pos = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"] # [G,4] xyxy, 512 기준
        gt_labels = targets[b]["labels"]  # [G], 0~num_classes-1
        G = gt_boxes.size(0)

        if G == 0: # GT가 없는 이미지는 전부 backgrounnd 유지
            continue

        # GT box를 0~1로 정규화(이미지 크기 512x512 가정)
        gt_boxes_norm = gt_boxes / 512.0 #[G,4] xyxy, 0~1
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [Q,4] xyxy, 0~1
        ious = box_iou(pred_boxes_xyxy, gt_boxes_norm)  # [Q,G]

        # 각 query별 best GT 매칭
        max_iou, best_gt_idx = ious.max(dim=1)  # [Q]

        # IoU threshhold 이상인 query만 positive로 설정
        iou_threshjold = 0.3
        positive = max_iou > iou_threshold 

        if positive.any():
            num_pos += positive.sum().item()
            # class target 설정
            target_classes[b, positive] = gt_labels[best_gt_idx[positive]]

            # box target (cx, cy, w, h, 0~1)
            matched_gt_boxes = gt_boxes_norm[best_gt_idx[positive]]  # [Np, 4] xyxy
            gt_cxcywh = box_xyxy_to_cxcywh(matched_gt_boxes)  # [Np,4] cxcywh
            pred_cxcywh = pred_boxes[b, positive]  # [Np,4] cxcywh
            loss_bbox += F.l1_loss(pred_cxcywh, gt_cxcywh, reduction="sum")

        if num_pos > 0:
            loss_bbox /= num_pos

        # 2) classification loss
        loss_cls = F.cross_entropy(pred_logits.view(-1, self.num_classes + 1), target_classes.view(-1))

        return{
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox
        }

    # ---------------- Inference -----------------

    def _postprocess(self, pred_logits, pred_boxes, images):
        outputs = []
        probs = pred_logits # [B, Q, C+1]
        scores, labels = probs[..., :-1].max(-1) # background 제외
        
        B, Q, _ = pred_boxes.shape
        for b in range(B):
            _, H, W = images[b].shape

            # 마지막 클래스(배경) 제외하고, 진짜 클래스들 중 최대 확률/해당 클래스 id 찾기
            boxes_cxcywh = pred_boxes[b]  # [Q,4] cxcywh, 0~1
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)  # [Q,4] xyxy, 0~1

            # image 크기 곱해서 픽셀 좌표로 되돌리기
            box_xyxy[..., 0::2] *= W
            box_xyxy[..., 1::2] *= H

            outputs.append({
                "boxes": boxes_xyxy,
                "scores": scores[b],
                "labels": labels[b]     
            })

        return outputs
    

