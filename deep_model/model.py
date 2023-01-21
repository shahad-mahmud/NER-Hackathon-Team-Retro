from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForPreTraining
from crf import CRF

def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)


class NERModel(nn.Module):
    def __init__(self, args,model_name=None,weights=None):
        super().__init__()
        self.args = args
        if args.weighted:
           self.crit_weights = weights
        self.banner = args.is_banner
        config = AutoConfig.from_pretrained(model_name,  output_hidden_states=True)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.top_rnns=args.top_rnns
        if self.banner:
            self.embeddings = nn.Embedding(32000, 256)
            self.lstm = nn.LSTM(bidirectional=False, input_size=256, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)
        
            self.model = AutoModelForPreTraining.from_pretrained(model_name, config = config )
            if self.top_rnns:
                self.rnn = nn.LSTM(bidirectional=True, num_layers=4, dropout=0.5, input_size=768+64, hidden_size=(768+64)//2, batch_first=True)
            self.classifier = nn.Sequential( 
                    nn.Linear(768+64, 512),
                    nn.Dropout(0.5),
                    nn.Linear(512, args.num_class+1))
            self.crf = CRF(args.num_class+1)
            self.finetuning = args.fine_tuning
            if args.weighted:
                self.loss_fnt = nn.CrossEntropyLoss(weight=self.crit_weights)
                print('Using weighted cross entropy loss')
            else:
                self.loss_fnt = nn.CrossEntropyLoss(ignore_index=9)
        else:
            self.model = AutoModel.from_pretrained(model_name,)
            self.classifier = nn.Linear(config.hidden_size, args.num_class)

        # self.device = args.device
            self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None,epoch=None):
        
        c = self.args.num_class
        if self.banner:
            if self.training and self.finetuning and (epoch>20):
                self.model.train()
                enc = self.model(input_ids, attention_mask).hidden_states[-1]
            else:
                self.model.eval()
                with torch.no_grad():
                    enc = self.model(input_ids, attention_mask).hidden_states[-1]
            
            # lstm embd
            lstm_embd = self.embeddings(input_ids)
            lstm_embd, _ = self.lstm(lstm_embd)
            
            enc = torch.concat([enc, lstm_embd], dim=2)
            
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
    def __init__(self, args, weights=None):
        super().__init__()
        self.args = args
        self.banner = args.is_banner
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        self.loss_fnt = nn.CrossEntropyLoss()
        self.n_model = args.n_model
        for i in range(self.n_model):
            if args.weighted:
                model = NERModel(args,args.model_name_or_path[i],weights=weights)
            else:
                model = NERModel(args,args.model_name_or_path[i])
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
                unsqueeze(dim=-1)!=self.args.num_class).\
                reshape(-1,self.args.num_class+1)\
                for i in range(len(outputs))]
            if self.banner:
                probs = [F.softmax(logit[:,:-1], dim=-1) for logit in logits]
            else:
                probs = [F.softmax(logit, dim=-1) for logit in logits]
                
            avg_prob = torch.stack(probs, dim=0).mean(0)

            mask = [(labels[i].view(-1) != self.args.num_class).to(logits[i]) for i in range(num_models)]
            reg_loss = sum([kl_div(avg_prob, probs[i]) for i in range(num_models)]) / num_models
            reg_loss = reg_loss.sum() / (mask[0].sum() + 1e-3)
            loss = loss + self.args.alpha_t * reg_loss
            model_output = (loss,) + model_output[1:]
        return model_output
