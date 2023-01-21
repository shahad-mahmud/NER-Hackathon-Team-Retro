import truecase,json
import re
from transformers import AutoTokenizer
import torch
# from datasets.noisy.noisy_vocab import LABEL_TO_ID
import pandas as pd
from ast import literal_eval
from utils.get_attr_from_file import get_attr_file
from utils.functions import get_vocab

VOCAB,LABEL_TO_ID,ID_TO_LABEL =  get_vocab()
print(LABEL_TO_ID)

def true_case(tokens):
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        if len(parts) != len(word_lst):
            return tokens
        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens


def process_instance(words, labels, tokenizers, n_model, max_seq_length=512,is_banner=False,index = None):
    tokens, token_labels,input_ids = [[]]*n_model, [[]]*n_model, [[]]*n_model
    # if is_banner:
    #     LABEL_TO_ID['<PAD>'] = 9
    # outputs={}
    for i in range(n_model):
        for word, label in zip(words, labels):
            tokenized = tokenizers[i].tokenize(word)
            
            if is_banner:
                token_label = [LABEL_TO_ID[label]] + [len(LABEL_TO_ID)] * (len(tokenized) - 1)
            else:
                token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)
            if len(tokenized) == 0:
                tokens[i] = tokens[i] + [word]
            else:
                tokens[i] = tokens[i] + tokenized
            token_labels[i] =  token_labels[i]+ token_label
        # print(len(token_labels),len(tokens))
        
        assert len(tokens[i]) == len(token_labels[i]) , print(words,len(words),len(labels),index)
        tokens[i], token_labels[i] = tokens[i][:max_seq_length - 2], token_labels[i][:max_seq_length - 2]
        input_ids[i] = tokenizers[i].convert_tokens_to_ids(tokens[i])
        input_ids[i] = tokenizers[i].build_inputs_with_special_tokens(input_ids[i])
        if is_banner:
            token_labels[i] = [len(LABEL_TO_ID)] + token_labels[i] + [len(LABEL_TO_ID)]
        else:
            token_labels[i] = [-1] + token_labels[i] + [-1]
        # outputs[f"input_ids_{i}"] = input_ids[i]
        # outputs[f"labels_{i}"] = token_labels[i]
        # outputs[f"tokens_{i}"] = tokens[i]
    # outputs['tags']=labels
    # outputs['n_models']=args.n_model
    # return outputs
    return {
        "input_ids": input_ids,
        "labels": token_labels,
        'tokens':tokens,
        'tags':labels,
        'n_models':n_model,
        'words':words
    }

def preprocess_data(data_path, tokenizer_names, type ,n_model,max_seq_length,is_banner):
    
    if type == 'txt':
        assert data_path.endswith('.txt') or data_path.endswith('.tsv')
    elif type == 'csv':
        assert data_path.endswith('.csv')        
    
    tokenizers=[]
    for tok in tokenizer_names:
        tokenizers.append(AutoTokenizer.from_pretrained(tok))

    words_list = []
    labels_list = []
    
    if type == 'csv':    
        def convert_null(val):
            return val.replace('null','None')
        inference_data = pd.read_csv(data_path,converters={"spans": lambda x:literal_eval(convert_null(x))})
        for i,row in inference_data.iterrows():
            # print(row['spans'])
            span_info = row['spans']
            words = row['text'].split()
            label_arr = ["O"]*len(words)
            for span in span_info:
                start,end = int(span['token_start']),int(span['token_end'])
                if span['label'] != 'OTHER':
                    label_arr[start:end] = ['B-'+span['label']] +['I-'+span['label'] for i in range(start,end-1)]
            
            words_list.append(words)
            labels_list.append(label_arr)
    
    elif type == 'txt':        
        with open(data_path,'r') as f:
            for line in f:
                line = line.strip()                    
                if len(line) > 0:
                    line_split = line.split('\t')
                    if len(line_split)>1:
                        sen,tags = line.split('\t')
                        words_list.append(sen.split())
                        labels_list.append(tags.split())
                    else:
                        words_list.append(line_split[0].split())
                        labels_list.append(["O"]*len(line_split[0].split()))
        
    
    train_features=[]
    
    for words,labels in zip(words_list,labels_list):
        
        train_features.append(process_instance(words,labels,tokenizers,n_model,max_seq_length,is_banner))
        
    return train_features 


def collate_func(batch):
    is_banner = True
    
    n_models = batch[0]['n_models']  
    input_ids,attention_mask,labels = [[]]*n_models,[[]]*n_models,[[]]*n_models
    # outputs = {}
    for i in range(n_models):
        max_len = max([len(f["input_ids"][i]) for f in batch])
        # tokens = [f[f"tokens"][i] for f in batch]
        input_ids[i] = [f[f"input_ids"][i] + [0] * (max_len - len(f[f"input_ids"][i])) for f in batch]
        attention_mask[i] = [[1.0] * len(f["input_ids"][i]) + [0.0] * (max_len - len(f[f"input_ids"][i])) for f in batch]
        
        input_ids[i] = torch.tensor(input_ids[i], dtype=torch.long)
        attention_mask[i] = torch.tensor(attention_mask[i], dtype=torch.float)    
        
        
        if batch[0]["labels"][0]:
            labels[i] = [f["labels"][i] + [len(LABEL_TO_ID)] * (max_len - len(f["labels"][i])) for f in batch]
            
        labels[i] = torch.tensor(labels[i], dtype=torch.long)
       
    if len(labels[0]) == 0:
        labels=None
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'tags':[f['tags'] for f in batch],
        'tokens':[f['tokens'] for f in batch],
        'words':[f['words'] for f in batch]
    }
    return output