# train.py
# This file contains the training logic, argument parsing, and training cycle for the project.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
from multiprocessing import freeze_support
import argparse
from globals import numClasses, imgSize, numPoints
from data import ChaLocDataLoader
from network import wR2, fh02
from utils import get_n_params

if __name__ == '__main__':
    freeze_support()  # For Windows multiprocessing compatibility
    # Argument parsing for training configuration
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to the input file")
    ap.add_argument("-n", "--epochs", default=10000, help="epochs for train")
    ap.add_argument("-b", "--batchsize", default=5, help="batch size for train")
    ap.add_argument("-eb", "--eval_batchsize", default=1, help="batch size for evaluation")
    ap.add_argument("-se", "--start_epoch", required=True, help="start epoch for train")
    ap.add_argument("-t", "--test", required=True, help="dirs for test")
    ap.add_argument("-r", "--resume", default='111', help="file for re-train")
    ap.add_argument("-f", "--folder", required=True, help="folder to store model")
    ap.add_argument("-w", "--writeFile", default='fh02.out', help="file for output")
    args = vars(ap.parse_args())

    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")

    # Prepare training and output directories
    batchSize = int(args["batchsize"]) if use_gpu else 2
    trainDirs = args["images"].split(',')
    testDirs = args["test"].split(',')
    modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
    storeName = modelFolder + 'fh02.pth'
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    epochs = int(args["epochs"])
    # Initialize output file if it doesn't exist
    if not os.path.isfile(args['writeFile']):
        with open(args['writeFile'], 'wb') as outF:
            pass

    # Model initialization
    model_conv = fh02(numPoints, numClasses)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()
    print(model_conv)
    print(get_n_params(model_conv))

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

    # Data loader for training
    dst = ChaLocDataLoader(trainDirs, imgSize)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)

    def train_model(model, criterion, optimizer, num_epochs=25):
        """
        Main training loop for the model.
        Args:
            model: The neural network to train
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of epochs to train
        """
        print("Starting training...")
        for epoch in range(num_epochs):
            lossAver = []
            for data in tqdm(trainloader):
                inputs, labels = data[0], data[1]
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                lossAver.append(loss.item())
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {sum(lossAver)/len(lossAver)}")
            lrScheduler.step()
            # Save model checkpoint after each epoch
            torch.save(model.state_dict(), storeName)

    # Start training
    train_model(model_conv, criterion, optimizer_conv, epochs) 