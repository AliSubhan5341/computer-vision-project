# utils.py
# This file contains utility functions for filename parsing and bounding box conversion.

import os

def parse_ccpd_filename(filename):
    """
    Parse CCPD filename to extract bounding box coordinates.
    The filename encodes bounding box information in a specific format.
    Returns:
        x1, y1, x2, y2: Coordinates of the bounding box.
    Raises:
        ValueError: If the filename format is invalid.
    """
    filename = os.path.splitext(filename)[0]
    parts = filename.split('-')
    if len(parts) != 7:
        raise ValueError(f"Invalid filename format: {filename}")
    bbox_coords = parts[2].split('_')
    x1, y1 = map(int, bbox_coords[0].split('&'))
    x2, y2 = map(int, bbox_coords[1].split('&'))
    return x1, y1, x2, y2

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format (normalized center x, center y, width, height).
    Args:
        x1, y1, x2, y2: Bounding box coordinates.
        img_width, img_height: Image dimensions.
    Returns:
        x_center, y_center, width, height: Normalized YOLO format values.
    """
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height 