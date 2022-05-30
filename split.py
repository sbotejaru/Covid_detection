import os
#import cv2
from matplotlib import pyplot as pyplot
import numpy as np
import random
import json

if __name__ == "__main__":
    # Inputs
    dataset_path = './Data/train_fix'
    split_ratio = [0.65, 0.15, 0.20]

    # Dict for keeping the data
    split = {"train": [], "test": [], "val": []}

    # Iterate folders
    for class_folder in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_folder)
        print(f"Found {len(os.listdir(class_folder))} images in class {class_folder}")

        # Compute indexes
        train_idx = int(split_ratio[0] * len(os.listdir(class_folder)))
        test_idx = int(split_ratio[1] * len(os.listdir(class_folder)))
        val_idx = int(split_ratio[2] * len(os.listdir(class_folder)))

        # Get only name, without extension
        images_list = [x.replace(".png", "") for x in os.listdir(class_folder)]

        # Shuffle
        random.shuffle(images_list)

        # Append to main dict
        split['train'].extend(images_list[:train_idx])
        split['test'].extend(images_list[train_idx:train_idx + test_idx])
        split['val'].extend(images_list[train_idx + test_idx:])
    
    print(f"Splitted {len(split['train'])} images in train, {len(split['test'])} in test and {len(split['val'])} in validation")
    
    # Save lists
    with open('split.json', 'w') as f:
        json.dump(split, f)




