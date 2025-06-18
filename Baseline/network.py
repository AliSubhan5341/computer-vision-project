# network.py
# This file contains all neural network model classes used in the project.

import torch
import torch.nn as nn
from globals import provNum, alphaNum, adNum, numPoints, numClasses

class wR2(nn.Module):
    """
    Deep convolutional neural network for feature extraction and classification.
    Used as a backbone or submodule in other models.
    """
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        # Define a deep stack of convolutional, batch norm, activation, pooling, and dropout layers
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, hidden8, hidden9, hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            nn.Linear(100, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the feature extractor and classifier.
        Args:
            x: Input tensor (batch of images)
        Returns:
            Output tensor (class scores)
        """
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x

class fh02(nn.Module):
    """
    Main model for character localization and recognition, using wR2 as a backbone.
    Contains multiple classifiers for different character positions.
    """
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        # Classifiers for each character position (province, alphabet, ad, ...)
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        """
        Load the wR2 backbone weights if a path is provided.
        Args:
            path: Path to the wR2 weights file
        """
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if path is not None:
            self.wR2.load_state_dict(torch.load(path))

    def forward(self, x):
        # Implement the forward pass as in your original code
        pass 