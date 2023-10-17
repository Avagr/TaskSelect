import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class GQA(Dataset):
    def __init__(self, img_dir: Path, question_file: Path, prompt: str, img_transform=None):
        with open(question_file, 'r') as f:
            self.questions = list(json.load(f).items())
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.prompt = prompt

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        _, question = self.questions[idx]
        img = Image.open(self.img_dir / f"{question['imageId']}.jpg")
        if self.img_transform is not None:
            img = self.img_transform(img)
        return question['imageId'], img, self.prompt.format(question['question']), question['answer']
