"""
globals.py
-----------
Defines global constants and lookup tables used throughout the project, including
Chinese province abbreviations, alphabets, alphanumeric symbols, and special tokens
for license plate recognition.
"""

# List of Chinese province abbreviations (used in license plates)
PROVINCES = [
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣",
    "鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁",
    "新","警","学","O"
]

# List of allowed alphabetic characters (used in license plates)
ALPHABETS = [
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U',
    'V','W','X','Y','Z','O'
]

# List of allowed alphanumeric characters (used in license plates)
ADS = [
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U',
    'V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','O'
]

# The complete vocabulary for the model (sorted, with special tokens added)
VOCAB = sorted(set(PROVINCES + ALPHABETS + ADS))

# Special tokens for sequence modeling
BOS_TOKEN = '<bos>'  # Beginning of sequence
EOS_TOKEN = '<eos>'  # End of sequence
PAD_TOKEN = '<pad>'  # Padding token

# Final vocabulary list (special tokens first)
VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + VOCAB

# Mapping from character to index in VOCAB
VOCAB_MAP = {c: i for i, c in enumerate(VOCAB)} 