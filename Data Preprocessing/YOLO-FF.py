import torch
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Train the YOLOv8 model
yolo task=detect mode=train model=yolov8s.pt data=/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/data.yaml epochs=5 imgsz=800 plots=True

# Evaluate the model
yolo task=detect mode=val model=runs/detect/train4/weights/best.pt data=/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/data.yaml split=test
