# evaluation.py
# This file contains functions for evaluating the trained model and running inference.

import os
import time
from ultralytics import YOLO
import cv2

def inference_timing_test(weights_path, input_folder):
    """
    Measure the average inference time per image for the trained YOLO model.
    Args:
        weights_path: Path to the trained YOLO weights.
        input_folder: Directory containing images for timing test.
    """
    model = YOLO(weights_path)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files = image_files[:100]  # Limit to first 100 images
    inference_times = []
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        start_time = time.time()
        results = model(img_path)[0]
        end_time = time.time()
        inference_times.append(end_time - start_time)
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time per image over {len(inference_times)} images: {avg_time:.4f} seconds")
    else:
        print("No images processed.")

def run_inference_and_save(weights_path, input_output_pairs):
    """
    Run inference on images and save cropped bounding box results.
    Args:
        weights_path: Path to the trained YOLO weights.
        input_output_pairs: List of (input_folder, output_folder) tuples.
    """
    model = YOLO(weights_path)
    def crop_and_save_bbox(image_path, results, save_path):
        """
        Crop the first detected bounding box from the image and save it.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            return
        boxes = results.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            print(f"No bounding box detected in {image_path}")
            return
        x1, y1, x2, y2 = map(int, boxes[0][:4])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            print(f"Empty crop for {image_path}")
            return
        cv2.imwrite(save_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
    for input_folder, output_folder in input_output_pairs:
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(input_folder, filename)
                results = model(img_path)[0]
                save_path = os.path.join(output_folder, filename)
                crop_and_save_bbox(img_path, results, save_path)
    print("Cropping and saving completed!") 