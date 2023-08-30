import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_setting", type=str)  # "multispan" or "expanded"

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
for split in ["valid", "train", "test"]:
    print(split)
    f = f"./data/{args.dataset_setting}/raw/" + split + ".json"
    data = json.load(open(f, "r"))["data"]
    for eid, example in tqdm(enumerate(data)):
        ctx = example["context"]
        del_indices = []
        for i, word in enumerate(ctx):
            tt = tokenizer.encode(word, add_special_tokens=False)
            if len(tt) == 0:
                del_indices.append(i)
        for i in del_indices[::-1]:
            del data[eid]["context"][i]
            if split != "test":
                del data[eid]["label"][i]
    json.dump(
        {"version": 1.0, "data": data},
        open(f"./data/{args.dataset_setting}/preprocessed/" + split + ".json", "w"),
    )
