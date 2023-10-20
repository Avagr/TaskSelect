import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM


def batch_iterate(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]


model = "meta-llama/Llama-2-7b-chat-hf"
cache_dir = "/home/agroskin/data/models/huggingface"
questions_path = "/home/agroskin/data/datasets/gqa/questions/testdev_balanced_questions.json"
res_path = f"/home/agroskin/data/datasets/gqa/preprocessed/testdev_balanced_objects_{model.split('/')[-1]}.json"
device = "cuda"

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)
model.to(device=device, dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token

prompt = """Extract the objects that are relevant to the sentences. Include all the objects mentioned. Do not answer the questions:

Sentence: Are there any blue cars or beach umbrellas in this photo?
Objects: [blue car, beach umbrella]

Sentence: What is the green food that is above the fruits to the left of the sprinkled dessert called?
Objects: [green food, fruits, sprinkled dessert]

Sentence: {}
Objects: ["""

with open(questions_path, "r") as f:
    questions = list(json.load(f).items())

res = {}
errors = []

batch_size = 2

with torch.no_grad():
    for batch in tqdm(batch_iterate(questions, batch_size)):
        batched_questions = [prompt.format(q["question"]) for _, q in batch]

        tokenized = tokenizer(batched_questions, return_tensors="pt", padding=True).to(device=device)
        output = model.generate(**tokenized, num_beams=5, do_sample=False)
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        for (i, q), obj in zip(batch, decoded):
            try:
                objects = list(map(lambda x: x.strip(), obj[obj.rfind("[") + 1:obj.rfind("]")].split(",")))
                if len(objects) == 1 and objects[0] == "":
                    objects = []
                res[i] = (q['question'], objects)
            except Exception as e:
                errors.append((i, q, e))

with open(res_path, "w") as f:
    json.dump(res, f)

with open(res_path + ".errors", "w") as f:
    json.dump(errors, f)

print(f"\r\n\n\nErrors: {len(errors)}")
