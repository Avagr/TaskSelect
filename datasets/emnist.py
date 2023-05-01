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
    def __init__(self, data_root: str, num_classes, transform, size_limit: Optional[int] = None):
        self.path_stubs: list[str] = []
        self.transform = transform
        for path, _, files in os.walk(data_root):
            for filename in files:
                if filename.endswith(".pkl"):
                    self.path_stubs.append(os.path.abspath(os.path.join(path, filename[:-7])))
        if size_limit is not None:
            self.path_stubs = self.path_stubs[:size_limit]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.path_stubs)

    @abc.abstractmethod
    def __getitem__(self, item):
        return


class EmnistExistence(EmnistDataset):

    def __init__(self, data_root: str, num_classes, transform, num_tasks, size_limit: Optional[int] = None):
        super().__init__(data_root, num_classes, transform, size_limit)
        self.task_vector = torch.zeros(num_tasks)
        self.args_vector = torch.zeros(num_classes)

    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            namespace = pickle.load(f)
        return self.transform(img)[0].unsqueeze(0), self.task_vector, self.args_vector, torch.Tensor(
            namespace.label_existence)


class EmnistLocation(EmnistDataset):

    def __init__(self, data_root: str, num_classes, transform, num_tasks, size_limit: Optional[int] = None):
        super().__init__(data_root, num_classes, transform, size_limit)
        self.task_vector = torch.zeros(num_tasks)
        # self.args_vector = torch.zeros(num_classes)

    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            labels = pickle.load(f).label_ordered
        # pos = 0
        pos = random.randint(0, 23)
        return self.transform(img)[0].unsqueeze(0), self.task_vector, F.one_hot(torch.tensor(pos),
                                                                                self.num_classes).type(
            torch.FloatTensor), labels[pos // 6, pos % 6]


class EmnistRightOfMatrix(EmnistDataset):

    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        right_of = torch.load(f"{path_stub}rightof.pt")
        return self.transform(img)[0].unsqueeze(0), right_of


class Emnist6LeftRight(EmnistDataset):

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
            arg = F.one_hot(torch.tensor(labels[index]), self.num_classes).type(torch.FloatTensor)
            return self.transform(img)[0].unsqueeze(0), torch.tensor([1., 0.]), arg, labels[index + 1]
        arg = F.one_hot(torch.tensor(labels[index + 1]), self.num_classes).type(torch.FloatTensor)
        return self.transform(img)[0].unsqueeze(0), torch.tensor([0., 1.]), arg, labels[index]


class Emnist6LRBoth(EmnistDataset):

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
            torch.tensor(labels[right_of_index]), self.num_classes).type(torch.FloatTensor), labels[right_of_index + 1]
        task_l, arg_l, res_l = torch.tensor([0., 1.]), F.one_hot(
            torch.tensor(labels[left_of_index]), self.num_classes).type(torch.FloatTensor), labels[left_of_index - 1]
        if random.random() < 0.5:
            return self.transform(img)[0].unsqueeze(0), task_r, arg_r, res_r, task_l, arg_l, res_l
        else:
            return self.transform(img)[0].unsqueeze(0), task_l, arg_l, res_l, task_r, arg_r, res_r


class Emnist24Directions(EmnistDataset):
    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = Image.open(f"{path_stub}img.jpg")
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            labels = pickle.load(f).label_ordered
        task_vertical = random.random() < 0.5
        if task_vertical:  # Up/Down
            task_top = random.random() < 0.5
            x, y = random.randint(0, 2), random.randint(0, 5)
            if task_top:
                return self.transform(img)[0].unsqueeze(0), torch.tensor([1., 0., 0., 0.]), F.one_hot(
                    torch.tensor(labels[x + 1, y]), self.num_classes).type(torch.FloatTensor), labels[x, y]
            else:
                return self.transform(img)[0].unsqueeze(0), torch.tensor([0., 1., 0., 0.]), F.one_hot(
                    torch.tensor(labels[x, y]), self.num_classes).type(torch.FloatTensor), labels[x + 1, y]
        else:  # Right/Left
            task_right = random.random() < 0.5
            x, y = random.randint(0, 3), random.randint(0, 4)
            if task_right:
                return self.transform(img)[0].unsqueeze(0), torch.tensor([0., 0., 1., 0.]), F.one_hot(
                    torch.tensor(labels[x, y]), self.num_classes).type(torch.FloatTensor), labels[x, y + 1]
            else:
                return self.transform(img)[0].unsqueeze(0), torch.tensor([0., 0., 0., 1.]), F.one_hot(
                    torch.tensor(labels[x, y + 1]), self.num_classes).type(torch.FloatTensor), labels[x, y]
