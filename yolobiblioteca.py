from ultralytics import YOLO

model = YOLO("yolov8s.pt")
print(model.names)  # Lista todas as classes disponíveis