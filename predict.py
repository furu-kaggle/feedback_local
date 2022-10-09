
import torch
from torch.optim import lr_scheduler
from transformers import AdamW
import os, gc, copy, time, random, string, joblib, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import json
from torch.utils.data import Dataset, DataLoader


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir')
parser.add_argument('--config_path')
parser.add_argument('--testdf_path')
parser.add_argument('--model_name_or_path')
parser.add_argument('--model_weight_path')
parser.add_argument('--tokenizer_path')
parser.add_argument('--batch_size')
parser.add_argument('--fold')
parser.add_argument('--output_path')
args = parser.parse_args()

dataset_dir = args.dataset_dir
config_path = args.config_path
testdf_path = args.testdf_path
model_path =  args.model_name_or_path
weight_path = args.model_weight_path
tokenizer_path = args.tokenizer_path
batch_size = int(args.batch_size)
fold = int(args.fold)
output_path = args.output_path

with open(config_path) as f:
    config = json.load(f)

config = config[f"fold{fold}"]
config["test_df"] = testdf_path
config["model_name"] = model_path
config["weight_path"] = weight_path
config["test_batch_size"] = batch_size


sys.path.append(dataset_dir)
#sys.path.append("/kaggle/input/iterative-stratification/iterative-stratification-master")
import src
from src.models import FeedBackModel
from src.test_datasets import testCollate, feedbacktestDataset
from src.train_datasets import FeedBackDataset, Collate

device = "cuda"
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, AutoModelForMaskedLM

config["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

hlc = src.helper(config=config)
test = hlc.get_test_df()

collate_fn = testCollate(config)
test_dataset = feedbacktestDataset(test, config=config)
test_loader = DataLoader(
    test_dataset, 
    batch_size=config["test_batch_size"], 
    collate_fn = collate_fn, 
    num_workers=os.cpu_count(), 
    pin_memory=True, 
    shuffle=False,
    drop_last=False,
)

model = FeedBackModel(config["model_name"], config).to(device)
model.load_state_dict(torch.load(config["weight_path"]))
model.model.half()
model.eval()

preds = []
text_ids = []
embs = []

bar = tqdm(enumerate(test_loader), total=len(test_loader))
for step, data in bar:
    text_id = data['text_id']
    ids = data['ids'].to(device, dtype = torch.long)
    mask = data['mask'].to(device, dtype = torch.long)
    with torch.no_grad():
        outputs, emb = model.get_emb(ids, mask)
    preds.append(outputs.cpu().detach().numpy())
    embs.append(emb.cpu().detach().numpy())
    text_ids.append(text_id)

preds = np.concatenate(preds)
embs = np.concatenate(embs)
text_ids = np.concatenate(text_ids)
gc.collect()
labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions']
pred_df = pd.DataFrame([text_ids],index=["text_id"]).T
for index,label in enumerate(labels):
    pred_df[label] = preds[:,index]
for emb_index in range(embs.shape[1]):
    pred_df[f"emb_{emb_index}"] = embs[:,emb_index]

pred_df.to_csv(output_path,index=False)
