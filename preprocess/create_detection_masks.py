import json
import sys

import torch
from PIL import Image, ImageDraw
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
from tqdm.auto import tqdm

version = sys.argv[1]
questions_path = "/home/agroskin/data/datasets/gqa/questions/testdev_balanced_questions.json"
objects_path = f"/home/agroskin/data/datasets/gqa/preprocessed/testdev_balanced_objects_spacy_nouns_adj.json"
img_dir = "/home/agroskin/data/datasets/gqa/images"
save_dir = f"/home/agroskin/data/datasets/gqa/preprocessed/masks_{version}"
device = "cuda"

model_path = "/home/agroskin/ViT/Projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
weights_path = "/home/agroskin/data/models/pretrain/groundingdino_swinb_cogcoor.pth"

with open(objects_path, "r") as f:
    objects_db = json.load(f)

model = load_model(model_path, weights_path)

with open(questions_path, "r") as f:
    questions = list(json.load(f).items())

box_threshold = 0.25
text_threshold = 0.25

match version:
    case "base":
        margin = 0.0
    case "m0.5":
        margin = 0.5
    case "m1":
        margin = 1.0
    case _:
        raise ValueError("Invalid version")

# noinspection PyTypeChecker
for i, q in tqdm(questions):
    objects = objects_db[i][1]
    prompt = " . ".join(objects)
    original_image, transformed = load_image(f"{img_dir}/{q['imageId']}.jpg")
    boxes, _, _ = predict(
        model=model,
        image=transformed,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    h, w = original_image.shape[:2]
    if margin > 0:
        boxes = boxes * torch.Tensor([1., 1., 1. + margin, 1. + margin])
        boxes.clamp_(0, 1)
    boxes = boxes * torch.Tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    if len(objects) == 0 or len(boxes) == 0:
        continue
    mask = Image.new('1', (w, h))
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=1)
    mask.save(f"{save_dir}/{i}.png")
