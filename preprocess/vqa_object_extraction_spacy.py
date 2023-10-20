import json

import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

questions_path = "/home/agroskin/data/datasets/gqa/questions/testdev_balanced_questions.json"

extract_adjectives = False

res_path = f"/home/agroskin/data/datasets/gqa/preprocessed/testdev_balanced_objects_spacy_nouns{'_adj' if extract_adjectives else ''}.json"

with open(questions_path, "r") as f:
    questions = list(json.load(f).items())

res = {}

with open("objects_stop_words.txt", 'r') as f:
    stop_words = set(f.read().strip().split(", "))

print(stop_words)

for i, q in tqdm(questions):
    doc = nlp(q["question"])
    if extract_adjectives:
        nouns = []
        for chunk in doc.noun_chunks:
            adjectives = [token.text.lower() for token in chunk if token.pos_ == "ADJ"]
            noun = chunk.root.text.lower()
            if chunk.root.lemma_.lower() not in stop_words:
                nouns.append(" ".join(adjectives + [noun]))
        res[i] = (q['question'], nouns)
    else:
        res[i] = (q['question'], [token.text.lower() for token in doc if
                                  token.pos_ == "NOUN" and token.lemma_.lower() not in stop_words])

with open(res_path, "w") as f:
    json.dump(res, f)
