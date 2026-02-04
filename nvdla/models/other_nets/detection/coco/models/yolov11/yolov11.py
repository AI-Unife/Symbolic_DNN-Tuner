import os
from ultralytics import YOLO

def get_yolov11(weights_dir):
    weights_path = os.path.join(weights_dir, 'yolo11n.pt')
    model = YOLO(weights_path)
    model.eval()
    return model