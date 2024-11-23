import os
from ultralytics import YOLO
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 첫 번째 NVIDIA GPU를 사용
# Initialize YOLO model
model = YOLO("yolov5su.pt")

# Check if GPU is available and use it if possible
if torch.cuda.is_available():
    model.to('cuda')

classNames = ['person']

def yolo_detect(img, confidence_threshold=0.4):
    results = model(img, stream=True)
    boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls < len(classNames) and classNames[cls] == 'person' and conf >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append({'coords': (x1, y1, x2, y2)})

    return boxes

