import torch
import json
import os
import numpy as np

class BaselineDataloader(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, phase):
        self.phase = phase
        self.dataset_path = dataset_path
        
        # Load split dict
        with open(split, 'r') as f:
            split = json.loads(f.read())
        
        # Extract only samples for phase
        self.files = split[phase]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.dataset_path, self.files[index] + ".npz"))

        # Get data
        img = data['data']
        gr = data['gr']

        # Preprocess
        img = np.reshape(img, (1, img.shape[0], img.shape[1])).astype(np.float32)
        gr_int = 0 if str(gr)=="covid" else (1 if str(gr)=="normal" else 2)

        return img, gr_int