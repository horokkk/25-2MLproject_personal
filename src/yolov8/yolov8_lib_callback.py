import time
from ultralytics.engine.callbacks import Callback

class CustomYoloCallback(Callback):

    def __init__(self):
        super().__init__()
        self.epoch_start = 0
        self.best_map = 0.0

    # --------------------------
    # Epoch 시작
    # --------------------------
    def on_train_epoch_start(self, trainer):
        self.epoch_start = time.time()

    # --------------------------
    # Epoch 끝 → loss 출력
    # --------------------------
    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        # YOLO는 각 epoch loss를 trainer.metrics로 제공
        loss = trainer.metrics['train/box_loss'] + \
               trainer.metrics['train/cls_loss'] + \
               trainer.metrics['train/dfl_loss']

        print(f"[YOLOv8][Epoch {epoch}] Train Loss = {loss:.4f}")

    # --------------------------
    # Validation 끝 → mAP 출력
    # --------------------------
    def on_val_end(self, trainer):
        epoch = trainer.epoch + 1
        metrics = trainer.metrics
        mAP50 = metrics.get("metrics/mAP50(B)", None)

        if mAP50 is not None:
            print(f"[YOLOv8][Epoch {epoch}] mAP50 = {mAP50:.4f}")

            if mAP50 > self.best_map:
                self.best_map = mAP50
                print(f"[YOLOv8] 🔥 Best updated: {self.best_map:.4f}")

    # --------------------------
    # Epoch 종료 → 시간 출력
    # --------------------------
    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        diff = time.time() - self.epoch_start
        print(f"[YOLOv8] Epoch {epoch} time: {diff:.2f}s ({diff/60:.2f} min)")
        print("-----------------------------------------------------")
