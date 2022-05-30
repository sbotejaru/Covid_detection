import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Inputs
    dataset_path = './Data/train_fix'
    output_path = './Data/train_processed_fix'
    desired_res = [64, 64]

    # Make output folder if it doesn't exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate classes
    for class_folder in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_folder)
        # Iterate files
        for file in os.listdir(class_folder):
            print(f"Preprocessing {file}")
            # Read
            file_full_path = os.path.join(class_folder, file)
            img = cv2.imread(file_full_path, 0)

            # Resize
            resized_img = cv2.resize(img, (desired_res[0], desired_res[1]))

            # Norm
            resized_img = (resized_img - np.min(resized_img)) / (np.max(resized_img) - np.min(resized_img))

            # Save
            img_path = os.path.join(output_path, file.replace(".png", ".npz"))
            gr = os.path.split(class_folder)[-1]
            #print(gr)
            np.savez(img_path, data=resized_img, gr=gr)