
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

class feedbacktestDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.text_id = df['text_id'].values
        self.text = df['full_text'].values
        self.config = config
        self.tokenizer = config["tokenizer"]
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        text_id = self.text_id[index]
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length = self.config["max_length"],
                        padding=False,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'text_id': text_id,
            'ids': ids,
            'mask': mask
        }

class testCollate:
    def __init__(self, config):
        self.tokenizer = config["tokenizer"]

    def __call__(self, batch):
        output = dict()
        for name in ["text_id", "ids","mask"]:
          output[name] = [sample[name] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
        output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]

        # convert to tensors
        for name in ["ids", "mask"]:
          output[name] = torch.tensor(output[name], dtype=torch.long)
        return output
