import os
import cv2
import argparse

import torch
import torchvision

import pickle
import numpy as np

from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--cifar100_root', default='wide-resnet', type=str, help='model')

args = parser.parse_args()

if __name__ == "__main__":
    cifar100_root = args.cifar100_root
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root=cifar100_root, train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR100(root=cifar100_root, train=False, download=False, transform=None)
    meta = unpickle(os.path.join(cifar100_root, "cifar-100-python", "meta"))
    
    os.makedirs(os.path.join(cifar100_root, "imagenet_format", "train"), exist_ok=True)
    os.makedirs(os.path.join(cifar100_root, "imagenet_format", "val"), exist_ok=True)
    
    idx = 0
    for img, target in tqdm(trainset):
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        label = meta["fine_label_names"][target]
        os.makedirs(os.path.join(cifar100_root, "imagenet_format", "train", label), exist_ok=True)
        
        image_path = os.path.join(cifar100_root, "imagenet_format", "train", label, str(idx) + ".jpg")
        cv2.imwrite(image_path, image)
        idx += 1

    for img, target in tqdm(testset):
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        label = meta["fine_label_names"][target]
        os.makedirs(os.path.join(cifar100_root, "imagenet_format", "val", label), exist_ok=True)
        
        image_path = os.path.join(cifar100_root, "imagenet_format", "val", label, str(idx) + ".jpg")
        cv2.imwrite(image_path, image)
        idx += 1