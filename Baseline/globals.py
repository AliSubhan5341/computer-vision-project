# globals.py
# This file contains global variables and constants used throughout the project.

# Number of output classes for the main task (e.g., number of character classes)
numClasses = 4
# Number of points (e.g., for keypoint or character position prediction)
numPoints = 4
# Standard image size for input to the network
imgSize = (480, 480)
# Number of provinces, alphabets, and ad characters (for license plate recognition)
provNum, alphaNum, adNum = 38, 25, 35

# List of province codes (Chinese license plate regions)
provinces = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
    "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"
]
# List of possible alphabet characters for license plates
alphabets = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', 'O'
]
# List of possible alphanumeric characters for license plates
ads = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'
] 