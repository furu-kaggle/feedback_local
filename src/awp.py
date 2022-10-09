import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

#loss backwardの後に入れて調整する
#パラメータ調整に関しては adv_param adv_lr adv_eps adv_step eval_thなどで調整する
class AWP:
    """
    Args:
    adv_param (str): layernameを書く
    adv_lr (float): このパラメータは、最初の層の埋め込みのみを攻撃する場合、すべてのパラメータで 0.1に調整されます。
    adv_eps (float): パラメーターの動きの最大幅の制限、一般に（0,1）の間で設定
    start_epoch (int): 動き始めるエポック
    adv_step (int): 攻撃回数、通常1回の攻撃で比較的効果はあるが、複数回の攻撃には正確な adv_lr が必要
    """
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, data, epoch, config):
        # 開始条件が満たされたときに敵対的訓練を開始する
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()  
            with torch.cuda.amp.autocast():
                loss = self.model(**data)
                loss = loss / config['n_accumulate']
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6  # 定义一个极小值
         # emb_name パラメータは、モデルの埋め込みのパラメータ名に置き換える必要があります
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        # emb_name パラメータは、モデルの埋め込みのパラメータ名に置き換える必要があります
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # 保存原始参数
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        # emb_name パラメータは、モデルの埋め込みのパラメータ名に置き換える必要があります
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
