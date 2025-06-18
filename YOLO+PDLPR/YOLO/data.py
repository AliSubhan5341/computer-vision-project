# data.py
# This file handles data management and dataset conversion for YOLO training.

import os
import cv2
from utils import parse_ccpd_filename, convert_to_yolo_format

def process_dataset(images_dir, labels_dir):
    """
    Process a dataset directory, converting CCPD image filenames to YOLO label files.
    Args:
        images_dir: Directory containing images.
        labels_dir: Directory to save YOLO label files.
    """
    os.makedirs(labels_dir, exist_ok=True)  # Ensure the labels directory exists
    for img_file in os.listdir(images_dir):
        # Only process image files
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img_height, img_width = img.shape[:2]
        try:
            # Parse bounding box from filename
            x1, y1, x2, y2 = parse_ccpd_filename(img_file)
            # Convert to YOLO format
            x_center, y_center, width, height = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
            # Write YOLO label file
            label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue 