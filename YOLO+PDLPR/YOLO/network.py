# network.py
# This file defines the function to load or initialize the YOLO model.

from ultralytics import YOLO

def get_yolo_model(weights_path=None):
    """
    Load a YOLO model. If a weights_path is provided, load the model with those weights.
    Otherwise, load the default YOLOv5x model.
    Args:
        weights_path: Path to the trained weights file (optional).
    Returns:
        YOLO model object.
    """
    if weights_path:
        return YOLO(weights_path)
    else:
        return YOLO('yolov5x.pt') 