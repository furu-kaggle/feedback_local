
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import random

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

class FeedBackDataset(Dataset):
    def __init__(self, df, config):
        #if config["text_encode"]:
        #df['full_text'] = df['full_text'].apply(lambda x : self.resolve_encodings_and_normalize(x))
        self.df = df
        self.text = df['full_text'].values
        self.text_id = df['text_id'].values
        self.targets = df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions']].values
        self.textlength = df["text_length"].values
        self.config = config
        self.tokenizer = config["tokenizer"]

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
          
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        text_id = self.text_id[index]
        target = self.targets[index]
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
            'mask': mask,
            'target': target
        }

class Collate:
    def __init__(self, config):
        self.tokenizer = config["tokenizer"]
        self.dropout_prob = config["token_dropout_prob"]
        self.dropout_ratio = config["token_dropout_ratio"]

    def __call__(self, batch):
        output = dict()
        for name in ["ids","mask", "target","text_id"]:
          output[name] = [sample[name] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
        output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]

        # convert to tensors
        for name in ["ids", "mask"]:
          output[name] = torch.tensor(output[name], dtype=torch.long)
        output["target"] = torch.tensor(output["target"], dtype=torch.float)

        if (self.dropout_prob > 0)&(random.uniform(0,1) < self.dropout_prob):
            output["ids"] = self.torch_mask_tokens(output["ids"])

        return output
    
    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        probability_matrix = torch.full(inputs.shape, self.dropout_ratio)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.clone().tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()\

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs
