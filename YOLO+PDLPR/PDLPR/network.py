"""
network.py
----------
Defines the neural network architecture for license plate recognition, including feature extraction,
encoder-decoder transformer blocks, and the main PDLPR model class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from globals import VOCAB, VOCAB_MAP, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# ---- Basic building blocks ----
class Focus(nn.Module):
    """
    Focus module from YOLOv5: Slices input into 4 and concatenates along channel dimension.
    Used for initial spatial downsampling and channel expansion.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        B, C, H, W = x.shape
        # Slice input into 4 sub-tensors and concatenate along channel axis
        x1 = x[..., 0:H:2, 0:W:2]
        x2 = x[..., 1:H:2, 0:W:2]
        x3 = x[..., 0:H:2, 1:W:2]
        x4 = x[..., 1:H:2, 1:W:2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.act(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    """
    Standard convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU.
    """
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """
    Residual block: two ConvBlocks with a skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.block1 = ConvBlock(channels, channels)
        self.block2 = ConvBlock(channels, channels)
    def forward(self, x):
        return x + self.block2(self.block1(x))

class ConvDownSampling(nn.Module):
    """
    Downsampling block using a ConvBlock with stride 2.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, k=3, s=2, p=1)
    def forward(self, x):
        return self.block(x)

class IGFE(nn.Module):
    """
    Image Global Feature Extractor (IGFE):
    Extracts spatial features from input images using a series of convolutional and residual blocks.
    Output shape: (B, 512, 6, 18) for input (B, 3, 48, 144).
    """
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 32)
        self.conv_ds1 = ConvDownSampling(32, 64)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.conv_ds2 = ConvDownSampling(64, 128)
        self.res3 = ResBlock(128)
        self.res4 = ResBlock(128)
        self.conv_out = nn.Conv2d(128, 512, kernel_size=1)
    def forward(self, x):
        x = self.focus(x)
        x = self.conv_ds1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_ds2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.conv_out(x)
        return x

class PositionalEncoding2D(nn.Module):
    """
    Adds 2D positional encoding to feature maps for transformer input.
    """
    def __init__(self, d_model, height=6, width=18):
        super().__init__()
        pe = torch.zeros(d_model, height, width)
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")
        d_model = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(torch.log(torch.tensor(10000.0)) / d_model)))
        pos_w = torch.arange(0, width).unsqueeze(1)
        pos_h = torch.arange(0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        # Add positional encoding to input feature map
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]

class EncoderUnit(nn.Module):
    """
    Transformer encoder unit with a convolutional expansion, multi-head attention, and normalization.
    """
    def __init__(self, d_model=512, expand=1024, nhead=8):
        super().__init__()
        self.conv_up = nn.Conv1d(d_model, expand, kernel_size=1)
        self.mha = nn.MultiheadAttention(expand, nhead, batch_first=True)
        self.conv_down = nn.Conv1d(expand, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.conv_up(x)
        x = x.transpose(1, 2)
        attn_out, _ = self.mha(x, x, x)
        x = attn_out
        x = x.transpose(1, 2)
        x = self.conv_down(x)
        x = x.transpose(1, 2)
        x = self.norm(x + residual)
        return x

class DecoderUnit(nn.Module):
    """
    Transformer decoder unit with masked self-attention, cross-attention, feed-forward, and normalization.
    """
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, tgt_mask=None):
        # Masked self-attention
        _tgt, _ = self.masked_mha(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + _tgt)
        # Cross-attention with encoder memory
        _tgt, _ = self.cross_mha(tgt, memory, memory)
        tgt = self.norm2(tgt + _tgt)
        # Feed-forward network
        _tgt = self.ffn(tgt)
        tgt = self.norm3(tgt + _tgt)
        return tgt

def generate_square_subsequent_mask(sz, device):
    """
    Generate a square mask for the sequence. Used to mask future positions in the decoder.
    Args:
        sz (int): Sequence length.
        device: Device to place the mask on.
    Returns:
        torch.Tensor: The mask tensor.
    """
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask.to(device)

class PDLPR(nn.Module):
    """
    Main model class for license plate recognition.
    Combines IGFE feature extractor, positional encoding, transformer encoder/decoder, and output layer.
    """
    def __init__(self, vocab_size=len(VOCAB), num_enc=3, num_dec=3, d_model=512, nhead=8, max_len=7):
        super().__init__()
        self.igfe = IGFE()
        self.pos_enc = PositionalEncoding2D(d_model)
        self.flatten = nn.Flatten(start_dim=2)
        self.transpose = lambda x: x.permute(0, 2, 1)
        self.encoders = nn.ModuleList([EncoderUnit(d_model=d_model, nhead=nhead) for _ in range(num_enc)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoders = nn.ModuleList([DecoderUnit(d_model=d_model, nhead=nhead) for _ in range(num_dec)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
    def forward(self, imgs, tgt_input):
        """
        Forward pass for the model.
        Args:
            imgs (torch.Tensor): Batch of input images (B, 3, 48, 144).
            tgt_input (torch.Tensor): Batch of input token indices for the decoder (B, L).
        Returns:
            torch.Tensor: Output logits (B, L, vocab_size).
        """
        feat = self.igfe(imgs)
        feat = self.pos_enc(feat)
        feat = self.flatten(feat)
        feat = self.transpose(feat)
        # Transformer encoder
        for enc in self.encoders:
            feat = enc(feat)
        memory = feat
        # Prepare target embeddings
        tgt_emb = self.embedding(tgt_input)
        B, L, _ = tgt_emb.shape
        tgt_mask = generate_square_subsequent_mask(L, tgt_emb.device)
        out = tgt_emb
        # Transformer decoder
        for dec in self.decoders:
            out = dec(out, memory, tgt_mask)
        logits = self.fc_out(out)
        return logits 