import json
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("--shot", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--task", type=str, required=True)
args = parser.parse_args()

np.random.seed(args.seed)

for mode in ["dev", "train"]:
    path = f"./dataset/{args.task}/{mode}.jsonl"
    few = []
    label_list = {}
    with open (path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data.pop("metadata")
            label = data["label"]
            if label not in label_list:
                label_list[label] = []
            label_list[label].append(data)
    for label in label_list:
        if len(label_list[label]) <= args.shot:
            few.extend(label_list[label])
        else:
            few.extend(np.random.choice(label_list[label], args.shot, replace=False))
    
    print(f"num of labels in {mode}: {len(label_list)}")
    np.random.shuffle(few)
            
    save = f"./dataset/{args.task}/{mode}_few{args.shot}.jsonl"
    with open(save, "w") as f:
        for line in few:
            f.write(json.dumps(line) + "\n")


        
