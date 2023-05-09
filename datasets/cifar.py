import abc
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torch.nn.functional as F


class CifarDataset(Dataset):
    def __init__(self, path: str, train: bool, transform, size_limit=None):
        self.transform = transform
        cifar = CIFAR10(path, train=train, download=False)
        self.samples = [cifar[i] for i in range(len(cifar))]
        if size_limit is not None:
            self.samples = self.samples[:size_limit]

    def __len__(self):
        return len(self.samples)

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


class CifarClassification(CifarDataset):

    def __getitem__(self, item):
        img, val = self.samples[item]
        return self.transform(img), torch.zeros(1), torch.zeros(10), val


class CifarQuery(CifarDataset):
    def __getitem__(self, item):
        img, val = self.samples[item]
        flag_correct = random.random() < 0.5
        task = torch.tensor([1.0])
        if flag_correct:
            arg = F.one_hot(torch.tensor(val), 10).type(torch.FloatTensor)
            res = 1.
        else:
            false_class = random.randint(0, 8)
            if false_class >= val:
                false_class += 1
            arg = F.one_hot(torch.tensor(false_class), 10).type(torch.FloatTensor)
            res = 0.
        return self.transform(img), task, arg, res


class CifarQueryOccurrence(CifarDataset):
    def __getitem__(self, item):
        img, val = self.samples[item]
        flag_correct = random.random() < 0.5
        task = torch.tensor([1.0])
        if flag_correct:
            arg = F.one_hot(torch.tensor(val), 10).type(torch.FloatTensor)
            res = 1.
        else:
            false_class = random.randint(0, 8)
            if false_class >= val:
                false_class += 1
            arg = F.one_hot(torch.tensor(false_class), 10).type(torch.FloatTensor)
            res = 0.
        return self.transform(img), task, arg, res, val

