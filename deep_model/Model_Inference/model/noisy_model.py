from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForPreTraining
from model.banner_crf import CRF
# from datasets.noisy.noisy_vocab import LABEL_TO_ID
from utils.functions import get_vocab

VOCAB,LABEL_TO_ID,ID_TO_LABEL =  get_vocab()

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


class NERModel(nn.Module):
    def __init__(self, model_name, weights=None, weighted = True,\
        dropout_prob = 0.1, is_banner = True,top_rnns = True, \
        fine_tuning = True, device = 'cuda'):
        super().__init__()
        
        if weighted:
           self.crit_weights = torch.tensor(weights)
        
        self.num_class = len(LABEL_TO_ID) 
        self.banner = is_banner
        self.device = device
        config = AutoConfig.from_pretrained(model_name,  output_hidden_states=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.top_rnns=top_rnns
        if self.banner:
            self.model = AutoModelForPreTraining.from_pretrained(model_name, config = config )
            if self.top_rnns:
                self.rnn = nn.LSTM(bidirectional=True, num_layers=4, dropout=0.5, input_size=768, hidden_size=768//2, batch_first=True)
            self.classifier = nn.Sequential( 
                    nn.Linear(768, 512),
                    nn.Dropout(0.5),
                    nn.Linear(512, len(LABEL_TO_ID)+1))
            self.crf = CRF(len(LABEL_TO_ID)+1)
            self.finetuning = fine_tuning
            if weighted:
                self.loss_fnt = nn.CrossEntropyLoss(weight=self.crit_weights)
                print('Using weighted cross entropy loss')
            else:
                crit_weights = torch.ones(len(LABEL_TO_ID)+1)
                crit_weights[-1] = 0
                self.loss_fnt = nn.CrossEntropyLoss(weight = crit_weights)
        else:
            self.model = AutoModel.from_pretrained(model_name,)
            self.classifier = nn.Linear(config.hidden_size, len(LABEL_TO_ID))

        # self.device = args.device
            self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None,epoch=None):
        
        c = self.num_class
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if self.banner:
            if self.training and self.finetuning and (epoch>20):
                self.model.train()
                enc = self.model(input_ids, attention_mask).hidden_states[-1]
            else:
                self.model.eval()
                with torch.no_grad():
                    enc = self.model(input_ids, attention_mask).hidden_states[-1]
            if self.top_rnns:
                h, _ = self.rnn(enc)
            logits_focal = self.classifier(h)
            y_hat = self.crf.forward(logits_focal)

            if labels is not None:
                crf_loss = self.crf.loss(logits_focal, labels)
                # logits_focal = logits_focal.view(-1, logits_focal.shape[-1])
                loss1 = self.loss_fnt(logits_focal.view(-1, logits_focal.shape[-1]), labels.view(-1))
                loss = crf_loss+loss1
                outputs = (loss,logits_focal)
                return outputs
            else:
                logits = logits_focal.view(-1, logits_focal.shape[-1])
                # outputs = (logits,)
                return logits
            
        else:
            h, *_ = self.model(input_ids, attention_mask, return_dict=False)
            h = self.dropout(h)
            logits = self.classifier(h)
            logits = logits.view(-1, c)
            outputs = (logits,)
            if labels is not None:
                labels = labels.view(-1)
                loss = self.loss_fnt(logits, labels)
                outputs = (loss,) + outputs
            return outputs
    
class NLLModel(nn.Module):
    def __init__(self, model_name_or_path, device= 'cuda', weights = None, is_banner = True,n_model = 2, weighted = True, alpha_t= 0):
        super().__init__()
        
        self.banner = is_banner
        self.models = nn.ModuleList()
        self.loss_fnt = nn.CrossEntropyLoss()
        self.n_model = n_model
        self.alpha_t = alpha_t
        self.device = [i % torch.cuda.device_count() for i in range(n_model)]
        for i in range(self.n_model):
            if weighted:
                model = NERModel(model_name_or_path[i],weights=weights)
            else:
                model = NERModel(model_name_or_path[i],weighted=False)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, labels=None,epoch=None):
        if labels is None:
            outputs = []
            for i in range(self.n_model):
                outputs.append(self.models[i](input_ids=input_ids[i].to(self.device[i]),
                                  attention_mask=attention_mask[i].to(self.device[i]),epoch=epoch
                                  ))
            return outputs 
        else:
            num_models = len(self.models)
            outputs = []
            for i in range(num_models):
                output = self.models[i](
                    input_ids=input_ids[i].to(self.device[i]),
                    attention_mask=attention_mask[i].to(self.device[i]),
                    labels=labels[i].to(self.device[i]) if labels[i] is not None else None,epoch=epoch,
                )
                output = tuple([o.to(0) for o in output])
                outputs.append(output)
            model_output = outputs[0]
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [torch.masked_select(outputs[i][1].to(self.device[i]),labels[i].to(self.device[i]).\
                unsqueeze(dim=-1)!=self.num_class).\
                reshape(-1,self.num_class+1)\
                for i in range(len(outputs))]
            if self.banner:
                probs = [F.softmax(logit[:,:-1], dim=-1) for logit in logits]
            else:
                probs = [F.softmax(logit, dim=-1) for logit in logits]
                
            avg_prob = torch.stack(probs, dim=0).mean(0)

            mask = [(labels[i].view(-1) != self.num_class).to(logits[i]) for i in range(num_models)]
            reg_loss = sum([kl_div(avg_prob, probs[i]) for i in range(num_models)]) / num_models
            reg_loss = reg_loss.sum() / (mask[0].sum() + 1e-3)
            loss = loss + self.alpha_t * reg_loss
            model_output = (loss,) + model_output[1:]
        return model_output

def load_pretrained_model(model,weight_path,device):
    
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    
    return model