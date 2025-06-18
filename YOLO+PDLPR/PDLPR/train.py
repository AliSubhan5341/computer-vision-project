"""
train.py
--------
Contains the training loop, evaluation function, and main entry point for training the license plate recognition model.
Handles data loading, model instantiation, checkpointing, and training progress reporting.
"""

import argparse, os, math, time, glob
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from data import CCPDPlateCrops
from network import PDLPR
from globals import VOCAB_MAP, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from utils import indices_to_string

def collate_fn(batch):
    """
    Custom collate function for DataLoader to pad sequences in a batch.
    Args:
        batch (list): List of (image, tgt_in, tgt_out) tuples.
    Returns:
        tuple: (batch_images, batch_tgt_in, batch_tgt_out)
    """
    imgs, tgt_in, tgt_out = zip(*batch)
    imgs = torch.stack(imgs)
    tgt_in = pad_sequence(tgt_in, batch_first=True, padding_value=VOCAB_MAP[PAD_TOKEN])
    tgt_out = pad_sequence(tgt_out, batch_first=True, padding_value=VOCAB_MAP[PAD_TOKEN])
    return imgs, tgt_in, tgt_out

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one epoch of training.
    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function.
        device: Device to run training on.
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for imgs, tgt_in, tgt_out in tqdm(loader, desc='train'):
        imgs, tgt_in, tgt_out = imgs.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(imgs, tgt_in)
        logits = logits.view(-1, logits.size(-1))
        tgt_out = tgt_out.view(-1)
        loss = criterion(logits, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, save_preds_path=None):
    """
    Evaluate the model on validation data.
    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.
        save_preds_path (str, optional): If provided, saves ground truth and predictions to file.
    Returns:
        tuple: (average_loss, sequence_accuracy, character_accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    gt_pred_pairs = []
    for imgs, tgt_in, tgt_out in tqdm(loader, desc='val'):
        imgs, tgt_in, tgt_out = imgs.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(imgs, tgt_in)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
        total_loss += loss.item()
        pred_tokens = logits.argmax(-1)[:, :-1]  # Ignore EOS shifted pos
        for p, t in zip(pred_tokens.cpu(), tgt_out.cpu()):
            # Remove PAD and EOS tokens
            p_seq = [tok for tok in p.tolist() if tok not in (VOCAB_MAP[PAD_TOKEN], VOCAB_MAP[EOS_TOKEN])]
            t_seq = [tok for tok in t.tolist() if tok not in (VOCAB_MAP[PAD_TOKEN], VOCAB_MAP[EOS_TOKEN])]
            if p_seq == t_seq:
                correct += 1
            total += 1
            min_len = min(len(p_seq), len(t_seq))
            for i in range(min_len):
                if p_seq[i] == t_seq[i]:
                    char_correct += 1
            char_total += len(t_seq)
            # Save for inspection if requested
            if save_preds_path is not None:
                gt_pred_pairs.append((indices_to_string(t_seq), indices_to_string(p_seq)))
    seq_acc = correct / total if total > 0 else 0
    char_acc = char_correct / char_total if char_total > 0 else 0
    # Save predictions if requested
    if save_preds_path is not None:
        with open(save_preds_path, 'w', encoding='utf-8') as f:
            for gt, pred in gt_pred_pairs:
                f.write(f"{gt}\t{pred}\n")
    return total_loss / len(loader), seq_acc, char_acc

def main():
    """
    Main entry point for training the model. Handles argument parsing, data loading, model setup,
    checkpointing, and training loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='root containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load training and validation datasets
    train_ds = CCPDPlateCrops(args.data_root, 'train')
    val_ds = CCPDPlateCrops(args.data_root, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize model, loss, optimizer, and scheduler
    model = PDLPR()
    model.to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB_MAP[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_acc = 0.0
    start_epoch = 1
    os.makedirs('checkpoints', exist_ok=True)

    # Try to resume from best checkpoint
    best_ckpts = sorted(glob.glob('checkpoints/best_acc_*.pth'))
    if best_ckpts:
        best_ckpt = best_ckpts[-1]
        print(f"Resuming from checkpoint: {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        try:
            best_acc = float(best_ckpt.split('_')[-1][:-4])
            start_epoch = int(best_ckpt.split('_')[-2]) + 1
        except Exception:
            pass

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        preds_file = f'predictions_epoch_{epoch}.txt'
        val_loss, val_acc, val_char_acc = evaluate(model, val_loader, criterion, args.device, save_preds_path=preds_file)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_seq_acc={val_acc:.4f} val_char_acc={val_char_acc:.4f}')
        print(f'Predictions saved to {preds_file}')
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pth')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'checkpoints/best_acc_{epoch}_{best_acc:.4f}.pth')
        scheduler.step()

if __name__ == '__main__':
    main()
