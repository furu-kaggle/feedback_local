from mimetypes import suffix_map
import torch
from torch.optim import lr_scheduler
from transformers import AdamW
#from torch.optim import Adam
import os, gc, copy, time, random, string, joblib, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import wandb
from .train_datasets import FeedBackDataset, Collate
from torch.utils.data import Dataset, DataLoader
from .models import FeedBackModel
from transformers import AutoTokenizer
from .swa import SWA
from .awp import AWP

def set_seed(seed=42):
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

class Trainers:

    def __init__(self, df, config, sweep=False, opt_params=None):
        self.config = config
        set_seed(config["seed"])
        self.df = df
        self.device = self.config["device"]
        self.fold = self.config["fold"]
        self.sweep = sweep
        if sweep:
            self.sweep_config = opt_params
            print("start sweep mode")
        else:
            wandb.init(
                    project="feedback3", 
                    group="baseline",
                    config = opt_params
            )
        self.opt_params = opt_params

    def fetch_scheduler(self):
        if self.config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.config['T_max'], 
                                                    eta_min=self.config['min_lr'])
        elif self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.config['T_0'], 
                                                                eta_min=self.config['min_lr'])
        elif self.config['scheduler'] == None:
            return None
            
        return scheduler
    
    def metric_fn(self, outputs, targets):
        colwise_mse = np.mean(np.square(targets - outputs), axis=0)
        loss = np.mean(np.sqrt(colwise_mse), axis=0)
        return loss

    def predict_fn(self, model, test_loader):
        model.eval()
        
        preds = []
        text_ids = []
        embs = []
        for step, data in enumerate(self.valid_loader):
            text_id = data['text_id']
            ids = data['ids'].to(self.device, dtype = torch.long)
            mask = data['mask'].to(self.device, dtype = torch.long)
            
            with autocast(enabled=True):
                outputs, emb = model.get_emb(ids, mask)
            preds.append(outputs.cpu().detach().numpy())
            embs.append(emb.cpu().detach().numpy())
            text_ids.append(text_id)
        
        preds = np.concatenate(preds)
        embs =  np.concatenate(embs)
        text_ids = np.concatenate(text_ids)
        gc.collect()
        pred_df = pd.DataFrame([text_ids,preds],index_col=["text_id","pred"]).T
        return pred_df

    @torch.no_grad()
    def valid_fn(self):
        self.model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        
        TEXT_IDS = []
        PREDS = []
        EMBS = []
        TARGETS = []

        for step, data in enumerate(self.valid_loader):
            text_id = data['text_id']
            ids = data['ids'].to(self.device, dtype = torch.long)
            mask = data['mask'].to(self.device, dtype = torch.long)
            targets = data['target'].to(self.device, dtype = torch.float)
            
            with autocast(enabled=False):
                outputs, emb = self.model.get_emb(ids, mask)
            TEXT_IDS.append(text_id)
            PREDS.append(outputs.cpu().detach().numpy())
            EMBS.append(emb.cpu().detach().numpy())
            TARGETS.append(targets.cpu().detach().numpy())
        
        TEXT_IDS = np.concatenate(TEXT_IDS)
        if self.config["loss"] == "bce":
            PREDS = np.concatenate(PREDS)*5.0
            TARGETS = np.concatenate(TARGETS)*5.0
        else:
            PREDS = np.concatenate(PREDS)
            TARGETS = np.concatenate(TARGETS)
        EMBS = np.concatenate(EMBS)
        valid_loss = self.metric_fn(PREDS, TARGETS)
        print("mcrmse score:",valid_loss)
        gc.collect()
        labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions']
        oof_df = pd.DataFrame([TEXT_IDS],index=["text_id"]).T
        for index,label in enumerate(labels):
            oof_df[f"{label}_pred"] = PREDS[:,index]
            oof_df[label] = TARGETS[:,index]
        for emb_index in range(EMBS.shape[1]):
            oof_df[f"emb_{emb_index}"] = EMBS[:,emb_index]
        
        return valid_loss, oof_df
  
    def train_one_epoch(self, epoch, best_epoch_loss):
        self.model.train()
      
        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, data in bar:
            text_id = data['text_id']
            data['ids'] = data['ids'].to(self.device, dtype = torch.long)
            data['mask'] = data['mask'].to(self.device, dtype = torch.long)
            data['target'] = data['target'].to(self.device, dtype = torch.float)

            with autocast():
                loss = self.model(**data)
                loss = loss / self.config['n_accumulate']
            self.scaler.scale(loss).backward()
            if self.config["adv_th"] > best_epoch_loss:
                self.awp.attack_backward(data,epoch,self.config) 

            if (step + 1) % self.config['n_accumulate'] == 0:
                if self.config["max_norm"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_norm"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
     
            running_loss += (loss.item() * self.config["train_batch_size"]) * self.config["n_accumulate"]
            dataset_size += self.config["train_batch_size"]
            
            epoch_loss = running_loss / dataset_size
            
            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                            bb_LR=self.optimizer.param_groups[0]['lr'])
            
            if (step % int(self.config["eval_step"]//self.config["train_batch_size"])==0)and((epoch-1)*len(self.train_loader) + step > int(self.config["eval_start"]//self.config["train_batch_size"])):
                val_epoch_loss, self.oof_df = self.valid_fn()
                self.model.train()
                
                if self.sweep:
                    criteria_loss = min(self.best_sweep_loss, best_epoch_loss)
                    if val_epoch_loss <= best_epoch_loss:
                        best_epoch_loss = val_epoch_loss
                else:
                    criteria_loss = best_epoch_loss

                # deep copy the model
                if val_epoch_loss <= criteria_loss:
                    print(f"Validation Loss Improved ({criteria_loss} ---> {val_epoch_loss})")
                    best_epoch_loss = val_epoch_loss
                    best_model = copy.deepcopy(self.model)
                    best_model.model.half()
                    self.best_model_wts = best_model.state_dict()
                    suffix_tag = "concat" if "pesudo_df_concat" in self.config else "pretrain"
                    PATH = f"{self.config['savedir']}/Loss-Fold{self.fold}_{suffix_tag}.bin"
                    torch.save(self.best_model_wts, PATH)
                    oof_path = f"{self.config['savedir']}/oof-Fold{self.fold}_{suffix_tag}.csv"
                    self.oof_df.to_csv(oof_path,index=False)
                    # Save a model file from the current directory
                    print(f"Model and oof dataframe Saved")
                    print()
                
                wandb.log({
                    "valid_loss":val_epoch_loss,
                    "best_valid_loss":best_epoch_loss,
                    "train_loss":epoch_loss
                })
        #self.optimizer.swap_swa_sgd()

            
        gc.collect()
        
        return epoch_loss, best_epoch_loss

    def run_training(self):
        if self.sweep:
            wandb.init()
            self.config.update(wandb.config)
        

        #import model and tokenzier
        self.model = FeedBackModel(self.config['model_name'], self.config)
        self.model.to(self.config['device'])
        if "pretrain_path" in self.config:
            self.model.load_state_dict(torch.load(self.config["pretrain_path"]))
        self.config["tokenizer"] = AutoTokenizer.from_pretrained(self.config['model_name'], use_fast=True)
        #self.config["tokenizer"].add_tokens(["\n"], special_tokens=True)
        self.config["tokenizer"].save_pretrained(f"{self.config['savedir']}/tokenizer_fold{self.config['fold']}")

        self.num_epochs = self.config["epochs"]
        #optimizer setting
        param_optimizer = list(self.model.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.config['weight_decay']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if "model" not in n], 'learning rate': self.config['head_lr'],'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate']
        )
        self.scaler = GradScaler()
        self.awp = AWP(
            self.model,
            self.optimizer,
            adv_lr=self.config["adv_lr"],
            adv_eps=self.config["adv_eps"],
            scaler=self.scaler
        )
        self.scheduler = self.fetch_scheduler()

        collate_fn = Collate(self.config)
        df_train = self.df[self.df.kfold != self.fold].reset_index(drop=True)
        train_dataset = FeedBackDataset(df_train, config=self.config)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['train_batch_size'], 
            collate_fn = collate_fn, 
            num_workers=os.cpu_count(), 
            pin_memory=False, 
            shuffle=True,
            drop_last=True,
        )

        #validate set
        collate_fn.dropout_drop = 0
        df_valid = self.df[self.df.kfold == self.fold].sort_values("text_length").reset_index(drop=True)
        valid_dataset = FeedBackDataset(df_valid, config=self.config)
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.config['valid_batch_size'], 
            collate_fn = collate_fn, 
            num_workers=os.cpu_count(), 
            pin_memory=False, 
            shuffle=False,
            drop_last=False,
        )

        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        if self.config["seed_average"]:
            pass
        else:
            self.best_epoch_loss = np.inf
        val_epoch_loss = self.valid_fn()
        if (self.sweep)&(self.best_sweep_loss > self.best_epoch_loss):
            self.best_sweep_loss = self.best_epoch_loss
        for epoch in range(1, self.num_epochs + 1): 
            gc.collect()
            train_epoch_loss, self.best_epoch_loss = self.train_one_epoch(epoch=epoch, best_epoch_loss=self.best_epoch_loss)
        if (self.sweep)&(self.best_sweep_loss > self.best_epoch_loss):
            self.best_sweep_loss = self.best_epoch_loss
        
        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        # save only best model for drive
        # if 'savedir_drive' in self.config:
        #     PATH = f"{self.config['savedir_drive']}/Loss-Fold{self.fold}_{self.best_epoch_loss}.bin"
        #     torch.save(self.best_model_wts, PATH)
        #     oof_path = f"{self.config['savedir_drive']}/oof-Fold{self.fold}_{self.best_epoch_loss}.csv"
        #     self.oof_df.to_csv(oof_path, index=False)
        #     token_path = f"{self.config['savedir_drive']}/tokenizer_fold{self.fold}_{self.best_epoch_loss}"
        #     self.config["tokenizer"].save_pretrained(token_path)
        
        print("Best Loss: {:.4f}".format(self.best_epoch_loss))
    
    def run_seed_average(self):
        #reset best loss
        self.best_epoch_loss = np.inf
        for seed in [41,455,3021]:
            set_seed(seed)
            self.run_training()

    def run_sweep(self,sweep_id=None):
        self.best_sweep_loss = np.inf
        if sweep_id is None:
            sweep_id = wandb.sweep(self.sweep_config)
        if self.config["seed_average"]:
            wandb.agent(sweep_id, self.run_seed_average)
        else:
            wandb.agent(sweep_id, self.run_training)
