import os
import pickle
import abc
import random
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class EmnistDataset(Dataset):
    def __init__(self, data_root: str, transform, size_limit: Optional[int] = None):
        self.path_stubs: list[str] = []
        self.transform = transform
        for path, _, files in os.walk(data_root):
            for filename in files:
                if filename.endswith(".pkl"):
                    self.path_stubs.append(os.path.abspath(os.path.join(path, filename[:-7])))
        if size_limit is not None:
            self.path_stubs = self.path_stubs[:size_limit]

    def __len__(self):
        return len(self.path_stubs)

    @abc.abstractmethod
    def __getitem__(self, item):
        return


class EmnistExistence(EmnistDataset):

    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            namespace = pickle.load(f)
        return self.transform(img), torch.Tensor(namespace.label_existence)


class EmnistRightOfMatrix(EmnistDataset):
    def __init__(self, data_root, num_classes, transform, size_limit: Optional[int] = None):
        super().__init__(data_root, transform, size_limit)
        self.num_classes = num_classes

    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        right_of = torch.load(f"{path_stub}rightof.pt")
        return self.transform(img), right_of


class EmnistLeftRight(EmnistDataset):

    def __init__(self, data_root, num_classes, transform, size_limit: Optional[int] = None):
        super().__init__(data_root, transform, size_limit)
        self.num_classes = num_classes

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, int):
        """
        :param item: item index
        :return: Transformed image, True if the task is right-of and False otherwise, Argument class as one-hot,
                 Result class
        """
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            labels: list[int] = pickle.load(f).label_ordered[0]
        task_right_of = random.random() < 0.5
        index = random.randint(0, 4)
        if task_right_of:
            return self.transform(img), torch.tensor([1., 0.]), F.one_hot(
                torch.tensor(labels[index]), self.num_classes).type(torch.FloatTensor), labels[index + 1]
        return self.transform(img), torch.tensor([0., 1.]), F.one_hot(
            torch.tensor(labels[index + 1]), self.num_classes).type(torch.FloatTensor), labels[index]


class EmnistLRBoth(EmnistDataset):

    def __init__(self, data_root, num_classes, transform, size_limit: Optional[int] = None):
        super().__init__(data_root, transform, size_limit)
        self.num_classes = num_classes

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, int,
                                         torch.Tensor, torch.Tensor, int):
        """
        :param item: item index
        :return: Transformed image, True if the task is right-of and False otherwise, Argument class as one-hot,
                 Result class
        """
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            labels: list[int] = pickle.load(f).label_ordered[0]
        right_of_index = random.randint(0, 4)
        left_of_index = random.randint(1, 5)
        task_r, arg_r, res_r = torch.tensor([1., 0.]), F.one_hot(
            torch.tensor(labels[right_of_index]),self.num_classes).type(torch.FloatTensor), labels[right_of_index + 1]
        task_l, arg_l, res_l = torch.tensor([0., 1.]), F.one_hot(
            torch.tensor(labels[left_of_index]), self.num_classes).type(torch.FloatTensor), labels[left_of_index - 1]
        if random.random() < 0.5:
            return self.transform(img), task_r, arg_r, res_r, task_l, arg_l, res_l
        else:
            return self.transform(img), task_l, arg_l, res_l, task_r, arg_r, res_r
