import os
import pickle
import abc
import random
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class PersonsDataset(Dataset):
    def __init__(self, data_root: str, num_persons, num_features, transform, size_limit: Optional[int] = None):
        self.path_stubs: list[str] = []
        self.transform = transform
        for path, _, files in os.walk(data_root):
            for filename in files:
                if filename.endswith(".pkl"):
                    self.path_stubs.append(os.path.abspath(os.path.join(path, filename[:-7])))
        if size_limit is not None:
            self.path_stubs = self.path_stubs[:size_limit]
        self.num_features = num_features
        self.num_persons = num_persons

    def __len__(self):
        return len(self.path_stubs)

    @abc.abstractmethod
    def __getitem__(self, item):
        return


class PersonsClassification(PersonsDataset):
    def __getitem__(self, item):
        path_stub = self.path_stubs[item]
        img = self.transform(Image.open(f"{path_stub}img.jpg"))[0].unsqueeze(0)
        with open(f"{path_stub}raw.pkl", 'rb') as f:
            features = pickle.load(f).person_features

        chosen_feature = random.randint(1, self.num_features)
        task = F.one_hot(torch.tensor(features[0]), self.num_persons).type(torch.FloatTensor)
        arg = F.one_hot(torch.tensor(chosen_feature) - 1, self.num_features).type(torch.FloatTensor)

        return img, task, arg, features[chosen_feature]
