import sys, os
import numpy as np
import torch
import torch.utils.data as data

class XDataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.data = torch.from_numpy(np.load(path)).float()

    def __getitem__(self, index):
        return self.data[index,...]

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'XDataset'
