import os
import torch
from torch.utils.data import DataLoader
import time

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

    os.makedirs("checkpoints_lite", exist_ok=True)
    best_mAP = 0.0

    epoch_times = []  # 에포크별 시간 기록용

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # 1) 학습
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch)

        # 2) 검증 + mAP
        print("-" * 40)
        print(f"-> Starting Validation for Epoch {epoch}...")
        metrics = evaluate_model(model, val_dataloader, device, num_classes)
        
        current_mAP = metrics['mAP']
        print(f"[Validation Result] Epoch {epoch}: mAP = {current_mAP:.4f}")
        
        # 3) 체크포인트 저장
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            path = f"checkpoints_lite/detr_lite_best_mAP_{best_mAP:.4f}.pth"
        else:
            path = f"checkpoints_lite/detr_lite_epoch{epoch}.pth"
        torch.save(model.state_dict(), path)

        # 4) 에포크 시간 계산
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[DETR-Lite] Epoch {epoch} time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)")
        print("-" * 40)

    total_time = time.time() - total_start
    print(f"[DETR-Lite] Total training time: {total_time/60:.2f} min")
    print("Epoch times (sec):", [round(t, 2) for t in epoch_times])

if __name__ == "__main__":
    main()

