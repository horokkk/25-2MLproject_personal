import os
import torch
from torch.utils.data import DataLoader

from dataset_detr import BeeDetrDataset, detr_collate_fn
from detr_lite import DETRLite
from eval_detr import evaluate_model

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    log_interval = 100 # 100 스텝마다 로그 출력

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 디버깅용 print
        # print(f"[DEBUG] batch {batch_idx}")
        # print(f"[DEBUG] images: {len(images)}, targets: {len(targets)}")
        # print(f"[DEBUG] targets[0]: {targets[0]}")

        loss_dict = model(images, targets)

        # 디버깅용 print
        # print("[DEBUG] loss_dict =", loss_dict)

        loss = sum(loss_dict.values())

        # 디버깅용 print
        # print("[DEBUG] total loss =", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # break # 한 배치만 디버깅용으로 처리

        # --- 추가된 부분: 100 스텝마다 Loss 출력 ---
        if (batch_idx + 1) % log_interval == 0:
            current_avg_loss = total_loss / (batch_idx + 1)
            print(f"[DETR-Lite] [Epoch {epoch}, Step {batch_idx+1}/{len(dataloader)}] current avg loss = {current_avg_loss:.4f}")
        # -------------------------------------------
    
    avg_loss = total_loss / len(dataloader)
    print(f"[DETR-Lite] [Epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss

def main():
    # 경로, 클래스 수 수정 필요
    train_img_dir = os.path.join("data", "images", "train")
    train_label_dir = os.path.join("data", "labels", "train")
    val_img_dir = os.path.join("data", "images", "val")
    val_label_dir = os.path.join("data", "labels", "val")

    num_classes = 7  # {'id': 0, 'name': '유충_정상'},{'id': 1, 'name': '유충_응애'},{'id': 2, 'name': '유충_석고병'},{'id': 3, 'name': '유충_부저병'},{'id': 4, 'name': '성충_정상'},{'id': 5, 'name': '성충_응애'},{'id': 6, 'name': '성충_날개불구바이러스감염증'}
    batch_size = 8  # T4-2 # A100 8
    num_epochs = 5 
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BeeDetrDataset(train_img_dir, train_label_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=detr_collate_fn) # 0->4로 수정

    val_dataset = BeeDetrDataset(val_img_dir, val_label_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=detr_collate_fn)

    model = DETRLite(num_classes=num_classes, num_queries=100)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_mAP = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch)
        
        # 검증 및 mAP 계산
        print("-" * 40)
        print(f"-> Starting Validation for Epoch {epoch}...")
        metrics = evaluate_model(model, val_dataloader, device, num_classes)
        
        current_mAP = metrics['mAP']
        print(f"[Validation Result] Epoch {epoch}: mAP = {current_mAP:.4f}")
        
        # 최고 성능 모델 저장
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            save_path = f"checkpoints/detr_lite_best_mAP_{best_mAP:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** NEW BEST MODEL SAVED: {save_path} ***")
        else:
             # mAP가 최고 기록을 갱신하지 못해도 현재 Epoch 모델을 저장할 경우
             torch.save(model.state_dict(), f"checkpoints/detr_lite_epoch{epoch}.pth")

        print("-" * 40)

if __name__ == "__main__":
    main()

