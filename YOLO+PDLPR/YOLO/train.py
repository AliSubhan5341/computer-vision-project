# train.py
# This file is the entry point for training the neural network.
# It imports necessary libraries, utility functions, and data processing functions.

import torch  # PyTorch for deep learning
import os     # OS operations
import math   # Math utilities

from utils import *  # Import all utility functions
from data import *   # Import all data management functions
from network import get_yolo_model
from globals import TRAIN_PATH, VAL_PATH

def main():
    """
    Main function to train the YOLO model on the CCPD dataset.
    """
    # Path to the dataset configuration file (YOLO expects a YAML file)
    data_yaml = "ccpd.yaml"  # Make sure this file is in your project root or update the path

    # Initialize the YOLO model (using default weights or custom if you have them)
    model = get_yolo_model()  # You can pass a weights path if you want to resume training

    # Start training
    model.train(
        data=data_yaml,      # Path to data config
        epochs=100,          # Number of epochs
        imgsz=640,           # Image size
        batch=8,             # Batch size
        name='ccpd_lp_train',
        project='runs/detect',
        save_period=1,       # Save every epoch
        save=True,           # Enable saving
        save_dir='runs/detect/ccpd_lp_train'  # Specify save directory
    )

if __name__ == "__main__":
    main()
