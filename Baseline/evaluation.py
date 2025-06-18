# evaluation.py
# This file contains all evaluation, testing, and timing logic for the project.

import os
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import argparse
from globals import numClasses, numPoints, imgSize, provinces, alphabets, ads
from data import labelTestDataLoader
from network import fh02
from utils import roi_pooling_ims
from multiprocessing import freeze_support
import time

# --- Evaluation logic from rpnetEval.py, rpnetEval100.py, demo.py, single_image_timer.py ---

def isEqual(labelGT, labelP):
    """
    Compare two label sequences and count exact and partial matches.
    Args:
        labelGT: Ground truth label sequence
        labelP: Predicted label sequence
    Returns:
        (int, bool): Number of exact matches, True if at least 6/7 match
    """
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare), sum(compare) >= 6

def get_first_100_images(directory):
    """
    Get the first 100 image files from a directory.
    Args:
        directory: Path to directory
    Returns:
        list: List of image file paths
    """
    image_files = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_files.append(os.path.join(directory, file))
            if len(image_files) >= 100:
                break
    return image_files

def eval_100_images(input_dir, model_path, store_folder):
    """
    Evaluate the model on the first 100 images in a directory and save results.
    Args:
        input_dir: Directory with images
        model_path: Path to model weights
        store_folder: Directory to save results
    """
    model_conv = fh02(numPoints, numClasses)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(model_path))
    model_conv = model_conv.cuda()
    model_conv.eval()
    count = 0
    correct = 0
    sixCorrect = 0
    total_time = 0
    image_files = get_first_100_images(input_dir)
    print(f"Found {len(image_files)} images to process")
    print("Starting evaluation over 100 images...")
    start = time.time()
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        img = cv2.resize(img, imgSize)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        label = os.path.basename(img_path).split('_')[0]
        YI = [[int(ee) for ee in label.split('_')[:7]]]
        x = Variable(img.cuda(0))
        with torch.no_grad():
            batch_start = time.time()
            fps_pred, y_pred = model_conv(x)
            batch_time = time.time() - batch_start
            total_time += batch_time
        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]
        exact_match, partial_match = isEqual(labelPred, YI[0])
        if exact_match == 7:
            correct += 1
        if partial_match:
            sixCorrect += 1
        count += 1
        if count % 10 == 0:
            avg_time = total_time / count
            print(f'Processed {count}/100 images')
            print(f'Current accuracy: {float(correct)/count:.4f}')
            print(f'Average time per image: {avg_time:.4f} seconds')
            print('-' * 50)
    total_time = time.time() - start
    avg_time = total_time / count
    print("\nFinal Results:")
    print(f'Total images processed: {count}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {float(correct)/count:.4f}')
    print(f'Six or more correct: {sixCorrect}')
    print(f'Average time per image: {avg_time:.4f} seconds')
    print(f'Total evaluation time: {total_time:.4f} seconds')
    sFolder = str(store_folder)
    sFolder = sFolder if sFolder[-1] == '/' else sFolder + '/'
    if not os.path.isdir(sFolder):
        os.mkdir(sFolder)
    with open(os.path.join(sFolder, 'eval_results_100.txt'), 'w') as f:
        f.write(f'Total images: {count}\n')
        f.write(f'Correct predictions: {correct}\n')
        f.write(f'Accuracy: {float(correct)/count:.4f}\n')
        f.write(f'Six or more correct: {sixCorrect}\n')
        f.write(f'Average time per image: {avg_time:.4f} seconds\n')
        f.write(f'Total evaluation time: {total_time:.4f} seconds\n')

def single_image_timer(input_dir, model_path):
    """
    Measure inference time and accuracy for 100 images, printing progress and results.
    Args:
        input_dir: Directory with images
        model_path: Path to model weights
    """
    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")
    model_conv = fh02(numPoints, numClasses)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(model_path))
    model_conv = model_conv.cuda()
    model_conv.eval()
    dst = labelTestDataLoader([input_dir], imgSize)
    loader = torch.utils.data.DataLoader(dst, batch_size=1, shuffle=False, num_workers=0)
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            XI, labels, ims = next(iter(loader))
            if use_gpu:
                x = Variable(XI.cuda(0))
            else:
                x = Variable(XI)
            _ = model_conv(x)
    print("\nStarting timing over 100 images...")
    total_time = 0.0
    num_images = 100
    correct = 0
    error = 0
    six_correct = 0
    processed = 0
    with torch.no_grad():
        for XI, labels, ims in loader:
            if processed >= num_images:
                break
            if use_gpu:
                x = Variable(XI.cuda(0))
            else:
                x = Variable(XI)
            start = time.time()
            fps_pred, y_pred = model_conv(x)
            end = time.time()
            inference_time = end - start
            total_time += inference_time
            pred = y_pred.data.max(1)[1]
            correct += pred.eq(labels.cuda()).sum().item()
            error += (pred != labels.cuda()).sum().item()
            if pred.item() == labels.cuda().item():
                six_correct += 1
            processed += 1
            if processed % 10 == 0:
                current_precision = correct / processed
                current_six = six_correct / processed
                current_avg_time = total_time / processed
                print(f"total {processed} correct {correct} error {error} precision {current_precision} six {current_six} avg_time {current_avg_time}")
    avg_time = total_time / num_images
    precision = correct / num_images
    six_accuracy = six_correct / num_images
    print(f"\nFinal Results:")
    print(f"Total images processed: {num_images}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time per image: {avg_time:.4f} seconds")
    print(f"FPS (images per second): {1/avg_time:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Six accuracy: {six_accuracy:.4f}")
    print(f"Correct predictions: {correct}")
    print(f"Errors: {error}")

if __name__ == '__main__':
    freeze_support()  # For Windows multiprocessing compatibility
    # Argument parsing for evaluation mode and configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['eval100', 'timer'], help='Evaluation mode: eval100 or timer')
    parser.add_argument('-i', '--input', required=True, help='Path to the input folder')
    parser.add_argument('-m', '--model', required=True, help='Path to the model file')
    parser.add_argument('-s', '--store', default='results', help='Path to the store folder (for eval100)')
    args = parser.parse_args()
    if args.mode == 'eval100':
        eval_100_images(args.input, args.model, args.store)
    elif args.mode == 'timer':
        single_image_timer(args.input, args.model) 