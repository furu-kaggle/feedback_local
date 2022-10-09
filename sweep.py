
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

import wandb
wandb.login(key="dd1758beb9fd6044fdc028dfc9245bba1c869a29")

# Utils
import torch
from tqdm import tqdm

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, AutoModelForMaskedLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version')
parser.add_argument('--savedir')
parser.add_argument('--savedir_drive')
parser.add_argument('--root_dir')
parser.add_argument('--fold')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

version = args.version
savedir = args.savedir
savedir_drive = args.savedir_drive
rootdir = args.root_dir
fold = int(args.fold)
resume = args.resume

with open(f'{savedir}/trainparam.json') as f:
    CONFIG = json.load(f)

with open(f'{savedir}/opt_parameters.json') as f:
    opt_params = json.load(f)

#CONFIG = CONFIG[f"fold{fold}"]

opt_params["name"] = f"{version}-fold{fold}"

CONFIG["savedir"] = savedir
CONFIG["savedir_drive"] = savedir_drive
CONFIG["version"] = version
sys.path.append(CONFIG["savedir"])
CONFIG["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import src
hlc = src.helper(config=CONFIG)
df = hlc.get_df()

hlc.config["fold"] = fold
hlc.config["exp"] = f"{version}"
print(f"====== sweep mode Fold: {fold} ======")
    
trainer = src.Trainers(
    df = df,
    config = hlc.config,
    opt_params = opt_params,
    sweep = True
)
if resume:
    trainer.run_sweep(sweep_id=CONFIG["sweep_id"])
else:
    trainer.run_sweep()
del trainer
_ = gc.collect()
print()
