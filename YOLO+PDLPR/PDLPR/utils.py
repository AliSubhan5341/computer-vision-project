"""
utils.py
--------
Provides utility functions for converting between filenames and label indices, and for converting indices back to string representations. Relies on global constants from globals.py.
"""

import re
from typing import List
from globals import PROVINCES, ALPHABETS, ADS, VOCAB, VOCAB_MAP, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

def filename_to_indices(filename: str) -> List[int]:
    """
    Parse the 7-character indices from a CCPD-style filename and convert them to label indices.
    Args:
        filename (str): The filename of the image, expected to follow CCPD naming convention.
    Returns:
        List[int]: List of integer indices corresponding to the license plate characters.
    Raises:
        ValueError: If the filename does not conform to the expected pattern.
    """
    # Extract the filename (remove directory path)
    name = filename.split('/')[-1]
    main_parts = name.split('-')
    if len(main_parts) < 5:
        raise ValueError(f"Filename {name} does not conform to CCPD pattern")
    index_part = main_parts[4]
    idxs = list(map(int, index_part.split('_')))
    if len(idxs) != 7:
        raise ValueError("Expect 7 indices in label part")
    # Map indices to actual characters using lookup tables
    chars = [
        PROVINCES[idxs[0]],
        ALPHABETS[idxs[1]],
        ADS[idxs[2]],
        ADS[idxs[3]],
        ADS[idxs[4]],
        ADS[idxs[5]],
        ADS[idxs[6]],
    ]
    # Convert characters to their corresponding indices in VOCAB
    return [VOCAB_MAP[c] for c in chars]

def indices_to_string(indices: List[int]) -> str:
    """
    Convert a list of label indices back to a string, ignoring special tokens.
    Args:
        indices (List[int]): List of integer indices.
    Returns:
        str: The decoded license plate string.
    """
    chars = []
    for i in indices:
        if i < len(VOCAB):
            ch = VOCAB[i]
            # Skip special tokens
            if ch not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                chars.append(ch)
    return ''.join(chars)
