"""
data.py
------
Defines the CCPDPlateCrops dataset class for loading and preprocessing license plate images and labels.
Handles image transformation and label encoding for model training and evaluation.
"""

import glob, os
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils import filename_to_indices
from globals import VOCAB_MAP, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

class CCPDPlateCrops(Dataset):
    """
    PyTorch Dataset for cropped license plate images from the CCPD dataset.
    Each item returns a transformed image tensor and two label tensors (input and output for seq2seq training).
    Expects directory structure:
        data/
          train/
            <image files>
          val/
            <image files>
    """
    def __init__(self, root: str, split: str = 'train'):
        """
        Args:
            root (str): Root directory containing the data splits.
            split (str): One of 'train', 'val', or 'test'.
        Raises:
            RuntimeError: If no images are found in the specified split.
        """
        assert split in ('train', 'val', 'test')
        # Recursively collect all .jpg image paths for the split
        self.paths = sorted(glob.glob(os.path.join(root, split, '**', '*.jpg'), recursive=True))
        if len(self.paths) == 0:
            raise RuntimeError(f'No images found in {root}/{split}')
        # Define image transformations: resize, convert to tensor, normalize
        self.transform = T.Compose([
            T.Resize((48, 144)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int):
        """
        Load and preprocess an image and its label.
        Args:
            idx (int): Index of the image.
        Returns:
            tuple: (image_tensor, tgt_input_tensor, tgt_output_tensor)
        """
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img)
        # Convert filename to label indices
        target_idxs = filename_to_indices(img_path)
        # Prepare input and output sequences for seq2seq (add BOS/EOS tokens)
        tgt_input = [VOCAB_MAP[BOS_TOKEN]] + target_idxs
        tgt_output = target_idxs + [VOCAB_MAP[EOS_TOKEN]]
        return tensor, torch.tensor(tgt_input, dtype=torch.long), torch.tensor(tgt_output, dtype=torch.long) 