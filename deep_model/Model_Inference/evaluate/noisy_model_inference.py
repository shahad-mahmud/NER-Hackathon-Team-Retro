CUDA_LAUNCH_BLOCKING="1"
import os
import numpy as np
import torch
# from datasets.noisy.noisy_vocab import LABEL_TO_ID, ID_TO_LABEL
from sklearn.metrics import classification_report,f1_score
from utils.functions import get_vocab

VOCAB,LABEL_TO_ID,ID_TO_LABEL =  get_vocab() 

def evaluate(model, dataloader, output_dir, show_performance = True, device = 'cuda', best_model_index=0,):
    model.to(device)
    preds, keys, words_list, probs = [], [], [], []
            
    for sen_id,batch in enumerate(dataloader):
        model.eval()
        # print(batch['labels'])
        batch_labels = batch['labels'][int(best_model_index)].cpu().numpy().flatten().tolist() 
        
        with torch.no_grad():
            logits = model(input_ids = batch['input_ids'],
                                attention_mask = batch['attention_mask'])
            # logits = model(**batch)[0]
        sen_preds=np.argmax(logits[int(best_model_index)][:,:-1].cpu().numpy(), axis=-1).tolist()
        sen_probs = torch.nn.Softmax(dim=-1)(logits[int(best_model_index)][:,:-1])
        sen_probs = torch.max(sen_probs,dim=-1).values.cpu().numpy()
        assert len(sen_preds) == len(batch_labels) == len(sen_probs)
        sen_preds = [pred for i,pred in enumerate(sen_preds) if batch_labels[i]!=len(LABEL_TO_ID)]
        sen_preds = [ID_TO_LABEL[l] for l in sen_preds]
        sen_probs = sen_probs[np.array(batch_labels)!=len(LABEL_TO_ID)]
        batch_labels = [label for label in batch_labels if label!=len(LABEL_TO_ID)]

        # print(sen_id,len(words_list),len(sen_preds))
        assert len(sen_preds) == len(batch_labels) == len(batch['words'][0]) == len(sen_probs)
        keys.append(batch_labels)
        words_list.append(batch['words'][0])
        probs.append(sen_probs)
        # print(sen_preds,words_list[sen_id])
        preds.append(sen_preds)
    
    with open(os.path.join(output_dir,'prediction.txt'),'w',encoding='utf8') as f:
        for tokens,tags in zip(words_list,preds):
            f.write(' '.join(tokens)+"\t"+" ".join(tags))
            f.write("\n")
            
    if show_performance:
        preds = [LABEL_TO_ID[p] for pred_list in preds for p in pred_list]
        keys = [k for key_list in keys for k in key_list]
        # print(len(preds),len(keys))
        
        assert len(preds) == len(keys)
        with open(os.path.join(output_dir,"prediction_report.txt"), 'w') as f:
            report = classification_report(keys,preds, target_names = list(LABEL_TO_ID.keys()), labels=list(LABEL_TO_ID.values()),  digits=4, zero_division=0)
            f1_macro = f1_score(keys,preds, labels=list(LABEL_TO_ID.values()),average = 'macro')
            # print(report)
            f.write(report)
            f.write('\nMacro F1\n')
            f.write(str(f1_macro))

    # print(preds,keys)
    return preds,probs