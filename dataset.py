import sys, os
import numpy as np
import torch
import torch.utils.data as data

class XDataset(data.Dataset):
    def __init__(self, path, data=None):
        self.path = path
        self.data = torch.from_numpy(np.load(path)).float() if data is None else data

    def __getitem__(self, index):
        return self.data[index,...]

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'XDataset'

    def split(self, test_set_ratio):
        n        = len(self)
        k        = np.round(n*test_set_ratio).astype(int)
        shuffled = np.random.permutation(self.data)
        traindat = shuffled[k:]
        testdat  = shuffled[:k]
        trainset = self.__class__('train_'+self.path, traindat)
        testset  = self.__class__('test_'+self.path, testdat)
        return trainset, testset
