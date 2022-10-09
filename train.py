
import sys, os, json

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import os, gc, copy, time, random, string, joblib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
from textblob import TextBlob

# Utils
import torch
from tqdm import tqdm

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, AutoModelForMaskedLM

import wandb
wandb.login(key="dd1758beb9fd6044fdc028dfc9245bba1c869a29")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--savedir')
parser.add_argument('--savedir_drive')
parser.add_argument('--root_dir')
parser.add_argument('--train_fold', nargs='*')
args = parser.parse_args()
savedir = args.savedir
savedir_drive = args.savedir_drive
rootdir = args.root_dir
train_fold = np.array(args.train_fold ,dtype=np.int64)
print(train_fold)

with open(f'{savedir}/trainparam.json') as f:
    CONFIG = json.load(f)

sys.path.append(savedir)
import src

for fold in train_fold:
    print(f"====== Fold: {fold} ======")
    config = CONFIG#[f"fold{fold}"]
    config["savedir_drive"] = savedir_drive 
    hlc = src.helper(config=config)
    df = hlc.get_df()
    hlc.config["savedir"] = savedir
    hlc.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Start a wandb run
    hlc.config["fold"] = fold 
    trainer = src.Trainers(
        df = df,
        config = hlc.config,
    )
    if hlc.config["seed_average"]:
        trainer.run_seed_average()
    else:
        trainer.run_training()
    
    del trainer
    _ = gc.collect()
    print()
