import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False

class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.tensor(q_t, dtype=torch.float))
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.tensor(w_ht, dtype=torch.float))

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm//2, num_layers = 2, batch_first=True, bidirectional=True)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = out[:, -1, :]
        return out

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = torch.stack(list(all_hidden_states), dim=0)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class FeedBackModel(nn.Module):
    def __init__(self, model_name, opt_params=None, num_labels=6, hidden_dim=128):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "hidden_dropout_prob": opt_params["backbone_dropout_prob"],
                "layer_norm_eps": 1e-7,
                "add_pooling_layer": False,
                "attention_probs_dropout_prob":opt_params["attention_probs_dropout_prob"],
                "num_labels": num_labels,
            }
        )
        
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        if opt_params["weight_pool"]:
            self.weighted_pooler = WeightedLayerPooling(
                num_hidden_layers=self.config.num_hidden_layers, 
                layer_start=self.config.num_hidden_layers - opt_params["weight_pool"]
            )
        if opt_params["loss"] == "bce":
            self.dropout = nn.Dropout(p=opt_params["head_dropout_prob"])
            self.drop1 = nn.Dropout(p=opt_params["stable_prob1"])
            self.drop2 = nn.Dropout(p=opt_params["stable_prob2"])
            self.drop3 = nn.Dropout(p=opt_params["stable_prob3"])
            self.drop4 = nn.Dropout(p=opt_params["stable_prob4"])
            self.drop5 = nn.Dropout(p=opt_params["stable_prob5"])

        self.pooler = MeanPooling()
        if opt_params["head_type"] == "att":
            self.hidden_dim = opt_params["hidden_dim"]
            self.head = nn.Sequential(
                AttentionPooling(
                    self.config.num_hidden_layers, 
                    self.config.hidden_size, 
                    self.hidden_dim
                ),
                nn.Linear(self.hidden_dim, self.config.num_labels)
            )
        elif opt_params["head_type"] == "lstm":
            self.hidden_dim = opt_params["hidden_dim"]
            self.head = nn.Sequential(
                LSTMPooling(
                    self.config.num_hidden_layers, 
                    self.config.hidden_size, 
                    self.hidden_dim
                ),
                nn.Linear(self.hidden_dim, self.config.num_labels)
            )
        else:
            self.fc = nn.Linear(self.config.hidden_size, self.config.num_labels)
        if opt_params["loss"] == "mse":
            self.loss = nn.MSELoss()
        elif opt_params["loss"] == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        #self.loss = nn.SmoothL1Loss()

        freeze(self.model.embeddings)
        if opt_params["head_type"]:
            self._init_weights(self.head[1])
        else:
            self._init_weights(self.fc)
            
        if opt_params["freeze_layernum"] > 0:
            num = opt_params["freeze_layernum"]
            freeze(self.model.encoder.layer[:num])

        if opt_params["reinit_layernum"] > 0:
            rnum = opt_params["reinit_layernum"]
            self._init_weights(self.model.encoder.layer[-rnum:])

        self.opt_params = opt_params

        self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def mcrmse_fn(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss

    def multihead_forward(self, out):
        #out = torch.cat([head(out) for head in self.heads],1)
        out = self.head(out)
        return out

    def get_emb(self, ids, mask, target=None,text_id=None):
        out = self.model(input_ids=ids,attention_mask=mask
                         ,output_hidden_states=True)

        if self.opt_params["head_type"]:
            input = out.hidden_states
            #out = torch.cat([head[0](input) for head in self.heads],1)
            #outputs = torch.cat([head(input) for head in self.heads],1)

            out = self.head[0](input)
            outputs = self.head(input)
        else:
            if self.opt_params["weight_pool"]:
                out = self.weighted_pooler(out.hidden_states)
                out = self.pooler(out,mask)
            else:
                out = self.pooler(out.last_hidden_state,mask)
            outputs = self.fc(out)
        if self.opt_params["loss"] == "bce":
            return outputs.sigmoid(), out
        else:
            return outputs.sigmoid(), out

    def forward(self, ids, mask, target=None,text_id=None):        
        out = self.model(input_ids=ids,attention_mask=mask
                         ,output_hidden_states=True)
        if self.opt_params["head_type"]:
            out = out.hidden_states
            outputs = self.multihead_forward(out)
        else:
            if self.opt_params["weight_pool"]:
                out = self.weighted_pooler(out.hidden_states)
                out = self.pooler(out,mask)
            else:
                out = self.pooler(out.last_hidden_state,mask)
            if self.opt_params["loss"] == "bce":
                out = self.dropout(out)
            outputs = self.fc(out)

        if target is not None:
            if self.opt_params["head_type"]:
                if self.opt_params["loss"] == "bce":
                    loss1 = self.loss(self.fc(self.drop1(out)), target) 
                    loss2 = self.loss(self.fc(self.drop2(out)), target) 
                    loss3 = self.loss(self.fc(self.drop3(out)), target) 
                    loss4 = self.loss(self.fc(self.drop4(out)), target) 
                    loss5 = self.loss(self.fc(self.drop5(out)), target) 
                    loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
                else:
                    loss = self.loss(self.multihead_forward(out), target) 
            else:
                if self.opt_params["loss"] == "bce":
                    loss1 = self.loss(self.fc(self.drop1(out)), target) 
                    loss2 = self.loss(self.fc(self.drop2(out)), target) 
                    loss3 = self.loss(self.fc(self.drop3(out)), target) 
                    loss4 = self.loss(self.fc(self.drop4(out)), target) 
                    loss5 = self.loss(self.fc(self.drop5(out)), target) 
                    loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
                else:
                    loss = self.loss(self.fc(out), target) 
            if self.opt_params["loss"] == "mse":
                loss = torch.sqrt(loss)
            return loss
        else:
            if self.opt_params["loss"] == "bce":
                return outputs.sigmoid()
            else:
                return outputs
