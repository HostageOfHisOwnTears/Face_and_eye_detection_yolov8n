from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="E:/diplom/project/face-and-eye-detection-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="face-eye-detector",
    device="cpu"
)
