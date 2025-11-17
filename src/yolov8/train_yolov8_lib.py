import os
from ultralytics import YOLO
from yolov8_lib_callback import CustomYoloCallback


def main():
    # --- 경로 설정 ---
    # 프로젝트 루트 기준으로 경로 잡기 (VSCode에서 실행한다고 가정)
    data_yaml = os.path.join("data", "bee_yolo.yaml")

    # --- 모델 선택 ---
    # 가장 가벼운 모델부터: yolov8n (nano)
    # 필요하면 s/m/l/x로 바꿔가면서 실험
    model = YOLO("yolov8n.pt")  # COCO pretrained weight 사용

    callbacks = [CustomYoloCallback()]

    # --- 학습 설정 ---
    results = model.train(
        data=data_yaml,
        epochs=50,          # 비교 실험용 epoch 수 (작게 시작해서 점점 늘려도 됨)
        imgsz=640,          # 입력 크기
        batch=16,           # GPU 메모리 보고 조절
        workers=4,
        lr0=0.001,          # 초기 learning rate
        optimizer="adamw",  # 기본은 SGD, 필요하면 변경
        project="runs_yolov8_lib",  # 결과 저장 디렉토리
        name="bee_yolo_v8n",        # 하위 폴더 이름
        exist_ok=True,      # 동일 이름 실험 덮어쓰기 허용
        verbose=True,
        callbacks=callbacks
    )

    # results 객체에 학습 로그/지표 경로 등이 들어있음
    print("Training finished.")
    print("Results saved in:", results.save_dir)


if __name__ == "__main__":
    main()
