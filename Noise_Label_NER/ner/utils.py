import torch
import random
import numpy as np
import json
from prepro import LABEL_TO_ID

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def collate_func(batch):
    with open('./config.json','r') as f:
        is_banner = json.load(f)['is_banner']
    
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
        
        if is_banner:
            if batch[0]["labels"][0]:
                labels[i] = [f["labels"][i] + [len(LABEL_TO_ID)] * (max_len - len(f["labels"][i])) for f in batch]
        else:
            if batch[0]["labels"][0]:
                labels[i] = [f["labels"][i] + [-1] * (max_len - len(f["labels"][i])) for f in batch]
            
        labels[i] = torch.tensor(labels[i], dtype=torch.long)
        # outputs[f"input_ids_{i}"] = input_ids
        # outputs[f"labels_{i}"] = labels
        # outputs[f"attention_mask_{i}"] = attention_mask
        # outputs[f"tokens_{i}"] = tokens
        
    # outputs['tags']=[f['tags'] for f in batch] 
    
    if len(labels[0]) == 0:
        labels=None
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'tags':[f['tags'] for f in batch],
        'tokens':[f['tokens'] for f in batch]
    }
    return output