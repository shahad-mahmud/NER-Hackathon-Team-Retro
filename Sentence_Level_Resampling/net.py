import torch
import torch.nn as nn
#from pytorch_pretrained_bert import BertModel
from dataset import VOCAB
from crf import CRF
from transformers import AutoModelForPreTraining, AutoConfig

class Net(nn.Module):
    def __init__(self, top_rnns=True, vocab_size=None, device='cpu', finetuning=True):
        super().__init__()

        coonfig = AutoConfig.from_pretrained("csebuetnlp/banglabert", output_hidden_states=True)

        #self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert", config = coonfig )
        self.projection = nn.Linear(768, 50, False)

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=4, dropout=0.5, input_size=768+50, hidden_size=818//2, batch_first=True)
        self.fc = nn.Sequential( 
                nn.Linear(768+50, 512),
                nn.Dropout(0.5),
                nn.Linear(512, vocab_size)
        )
        

        self.device = device
        self.finetuning = finetuning
        self.crf = CRF(len(VOCAB))

    def forward(self, x, y=None, attention_mask = None, epoch=None):
        
        x=x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if y is not None:
            y=y.to(self.device)
        
        if self.training and self.finetuning and (epoch>20):
            self.bert.train()
            encoded_layers = self.bert(x,attention_mask=attention_mask)
            enc = encoded_layers.hidden_states[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers = self.bert(x,attention_mask=attention_mask)
                enc = encoded_layers.hidden_states[-1]
        
        # create the n-grams
        n_gram_batch = []
        for sample in enc:
            n_tokens = sample.shape[0]
            n_grams = [torch.mean(sample[0:2], dim=0)]
            for i in range(1, n_tokens-1):
                n_grams.append(torch.mean(sample[i-1:i+2], dim=0))
            n_grams.append(torch.mean(sample[n_tokens-1:], dim=0))
            
            n_gram_batch.append(torch.stack(n_grams))
        
        n_gram_batch = torch.stack(n_gram_batch)
        n_gram_batch = self.projection(n_gram_batch)
        enc = torch.concat([enc, n_gram_batch], dim=2)

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits_focal = self.fc(enc)
        y_hat = self.crf.forward(logits_focal)
        logits_crf = self.crf.loss(logits_focal, y)
        return logits_focal, logits_crf, y, y_hat