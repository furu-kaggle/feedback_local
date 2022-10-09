
import os, gc, copy, time, random, string, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from textblob import TextBlob

from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except:
    print("not installed iterstrat")

from .train_datasets import FeedBackDataset, Collate

class helper:
    def __init__(self, config):
        self.config = config
        self.set_seed(config['seed'])

    def resolve_encodings_and_normalize(self, text: str) -> str:
        def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
            return error.object[error.start : error.end].encode("utf-8"), error.end


        def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
            return error.object[error.start : error.end].decode("cp1252"), error.end

        # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
        codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
        codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)
        """Resolve the encoding problems and normalize the abnormal characters."""
        text = (
            text.encode("raw_unicode_escape")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
            .encode("cp1252", errors="replace_encoding_with_utf8")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
        )
        text = unidecode(text)
        return text

    def set_seed(self, seed=42):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)

    def prepare_loaders_splite_trainstep(self, df, fold):
        collate_fn = Collate(self.config)
        df_valid = df[df.kfold == fold].sort_values("text_length").reset_index(drop=True)
        valid_dataset = FeedBackDataset(df_valid, config=self.config)
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.config['valid_batch_size'], 
            collate_fn = collate_fn, 
            num_workers=os.cpu_count(), 
            pin_memory=True, 
            shuffle=False,
            drop_last=False,
        )

        df_train = df[df.kfold != fold].reset_index(drop=True)
        train_dataset = FeedBackDataset(df_train, config=self.config)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['train_batch_size'], 
            collate_fn = collate_fn, 
            num_workers=os.cpu_count(), 
            pin_memory=True, 
            shuffle=True,
            drop_last=True,
        )

        return train_loader, valid_loader

    def get_df(self):
        df = pd.read_csv(self.config["train_df"])
        #df['full_text'] = df['full_text'].apply(lambda x : self.resolve_encodings_and_normalize(x).strip().lower())
        df["text_length"] = df.full_text.apply(lambda x: len(x.split()))

        mskf = MultilabelStratifiedKFold(n_splits=self.config['n_fold'], shuffle=True, random_state=self.config["fold_seed"])
        labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
        for fold, ( _, val_) in enumerate(mskf.split(df, df[labels].values)):
            df.loc[val_ , "kfold"] = int(fold)

        df["kfold"] = df["kfold"].astype(int)
        if "pesudo_df" in self.config:
            psdf = pd.read_csv(self.config["pesudo_df"])
            # dummy fold number
            psdf["fold"] = 5
            df = pd.concat([df,psdf]).reset_index(drop=True)

        if self.config["loss"] == "bce":
            for label in labels:
                df[label] = df[label]/5.0


        return df

    def get_test_df(self):
        df = pd.read_csv(self.config["test_df"])
        df['full_text'] = df['full_text'].apply(lambda x : self.resolve_encodings_and_normalize(x))
        df["text_length"] = df.full_text.apply(lambda x: len(x.split()))
        return df
