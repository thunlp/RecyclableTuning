import torch
import argparse
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
from IPython import embed
from opendelta.auto_delta import AutoDeltaModel
import math
import os
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input1", default="", type=str)
parser.add_argument(
    "--input2", default="", type=str)
parser.add_argument(
    "--adapter1", default="", type=str)
parser.add_argument(
    "--adapter2", default="", type=str)
parser.add_argument("--method", default="model", type=str)
parser.add_argument("--save_path", default="./cal_dist/test.txt", type=str)

args = parser.parse_args()

model1 = torch.load(os.path.join(args.input1, "pytorch_model.bin"))
if "train_itp" in model1:
    flat1 = model1["train_itp"]
else:
    # config1 = AutoConfig.from_pretrained(args.input1, num_labels=6)
    model1 = AutoModelForSequenceClassification.from_pretrained(args.input1)
    if args.method == "adapter":
        delta_model1 = AutoDeltaModel.from_finetuned(args.adapter1, backbone_model=model1)
        delta_model1.freeze_module(set_state_dict = False)
    if args.method == "model" or args.method == "finetune":
        # from IPython import embed
        # embed()
        keys = [k for k in model1.state_dict().keys() if "class" not in k]
        model_dict1 = [model1.state_dict()[k] for k in keys]
    elif args.method == "adapter":
        model_dict1 = [v for v in model1.state_dict().values()]
    flat1 = torch.tensor([]).cuda()
    for v in model_dict1:
        flat1 = torch.cat((flat1, v.flatten().cuda()), dim=0)

model2 = torch.load(os.path.join(args.input2, "pytorch_model.bin"))
if "train_itp" in model2:
    flat2 = model2["train_itp"]
else:
    # config2 = AutoConfig.from_pretrained(args.input2, num_labels=6)
    model2 = AutoModelForSequenceClassification.from_pretrained(args.input2)
    if args.method == "adapter":
        delta_model2 = AutoDeltaModel.from_finetuned(args.adapter2, backbone_model=model2)
        delta_model2.freeze_module(set_state_dict = False)
    if args.method == "model" or args.method == "finetune":
        keys = [k for k in model2.state_dict().keys() if "class" not in k]
        model_dict2 = [model2.state_dict()[k] for k in keys]
    elif args.method == "adapter":
        model_dict2 = [v for v in model2.state_dict().values()]
    flat2 = torch.tensor([]).cuda()
    for v in model_dict2:
        flat2 = torch.cat((flat2, v.flatten().cuda()), dim=0)
        
cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
sim = cos_sim(flat1, flat2)

dis = torch.dist(flat1, flat2).item()
_len = len(flat1)
average = math.sqrt(dis * dis / _len)

print("L2: ", dis)

with open(args.save_path, "w") as f:
    f.write("L2 distance is: \n")
    f.write(str(dis))
    f.write("\nnum of para is: \n")
    f.write(str(_len))
    f.write("\naverage L2 distance is: \n")
    f.write(str(average))
    f.write("\nCos distance is: \n")
    f.write(str(sim.item()))

print("=================finish calculating!====================")
