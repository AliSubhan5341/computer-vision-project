"""
evaluation.py
-------------
Script for evaluating the inference speed of the trained license plate recognition model.
Loads the latest checkpoint, runs inference on a validation set, and reports average inference time per image.
"""

import time
import torch
from torch.utils.data import DataLoader
from data import CCPDPlateCrops
from network import PDLPR
from globals import VOCAB_MAP, PAD_TOKEN, BOS_TOKEN
import glob

# Settings for evaluation
DATA_ROOT = 'data'           # Root directory for data
BATCH_SIZE = 1               # Batch size for evaluation (1 for speed test)
NUM_IMAGES = 100             # Number of images to evaluate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
CHECKPOINT_PATTERN = 'checkpoints/epoch_*.pth'  # Pattern to find checkpoints

# Load validation dataset
val_ds = CCPDPlateCrops(DATA_ROOT, 'val')
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load the latest model checkpoint
model = PDLPR()
ckpts = sorted(glob.glob(CHECKPOINT_PATTERN), key=lambda x: int(x.split('_')[-1].split('.')[0]))
if not ckpts:
    raise RuntimeError('No checkpoint found!')
print(f'Loading latest checkpoint: {ckpts[-1]}')
model.load_state_dict(torch.load(ckpts[-1], map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Run inference and measure time
n = 0
total_time = 0.0
with torch.no_grad():
    for imgs, tgt_in, tgt_out in val_loader:
        imgs = imgs.to(DEVICE)
        tgt_in = tgt_in.to(DEVICE)
        start = time.time()
        _ = model(imgs, tgt_in)
        end = time.time()
        total_time += (end - start)
        n += 1
        if n >= NUM_IMAGES:
            break

# Report average inference time
avg_time = total_time / n
print(f"Average inference time per image over {n} images: {avg_time*1000:.2f} ms") 