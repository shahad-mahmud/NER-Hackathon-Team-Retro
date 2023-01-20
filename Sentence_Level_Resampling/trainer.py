import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from crf import CRF
from dataset import tokenizer, VOCAB, tag2idx, idx2tag
from sklearn.metrics import classification_report,f1_score
import wandb
# import mlflow
# import mlflow.sklearn
import math

def train(model, iterator, optimizer, cost, criterion, epoch, args):
    loss_list = []
    model.train()
    n_steps_per_epoch = math.ceil(len(iterator.dataset) / args.train_batch_size)
    
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, attention_mask = batch
        _y = y 
        optimizer.zero_grad()
        logits_focal, logits_crf, y, _ = model(x, y,attention_mask, epoch)

        logits_focal = logits_focal.view(-1, logits_focal.shape[-1])
        y_focal = y.view(-1)
        
        temp, _ = cost(logits_focal, y_focal)
        loss1 = criterion(temp, y_focal)
        if isinstance(loss1, tuple):
            loss1 = loss1[0]
        
        loss = logits_crf+loss1
        loss_list.append(int(loss))
        loss.backward()

        optimizer.step()
        #log each step loss in wandb
        metrics = {'train_step_loss':loss}
        if i + 1 < n_steps_per_epoch:
            continue
          #wandb.log(metrics)

        
        if i%100 == 0:
            print(f'Step: {i} and train loss: {loss}')
        if i==0:
            print("==============Check Dataloader===============")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("attention mask:", attention_mask[0])
            print("=============================================")
        
    return loss_list
        

def eval(model, iterator, criterion, args, is_test = False, epoch=None):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat, Y_hat_viterbi = [], [], [], [], [], []
    

    crf = CRF(len(VOCAB))
    crf.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    #loss_list= []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens, attention_mask = batch

            logits_focal, logits_crf, y_, y_hat = model(x, y,attention_mask)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            

    all_preds = []
    
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        # prob = prob[np.array(is_heads)==1]
        for x in range(len(preds)):
            if preds[x] == '<PAD>':
                preds[x] = 'O'
        assert len(preds)==len(words.split())==len(tags.split()) , "Sentence: {}\n True Tags: {}\n Pred Tags: {}".format(words,tags,preds)
        
        all_preds.append(preds)
        # all_pred_probs.append(prob)
    
    if is_test:
        preds = [tag2idx[p] for pred_list in all_preds for p in pred_list[1:-1]]
        keys = [tag2idx[k] for key_list in Tags for k in key_list.split()[1:-1]]
        assert len(preds) == len(keys)
        target_names = [k for k in list(tag2idx.keys()) if k!='<PAD>']
        labels = [v for v in list(tag2idx.values()) if v!=0]
        report = classification_report(keys,preds, target_names = target_names, labels=labels,  digits=4, zero_division=0)
        f1_macro = f1_score(keys,preds, labels=labels,average = 'macro')
        
        return f1_macro,report
    
    
    with open(os.path.join(args.output_dir,"val_report.txt"), 'a') as f:
        if epoch is not None:
            headline = "============Evaluation at Epoch= "+str(epoch)+ " ============"
            preds = [tag2idx[p] for pred_list in all_preds for p in pred_list[1:-1]]
            keys = [tag2idx[k] for key_list in Tags for k in key_list.split()[1:-1]]
            assert len(preds) == len(keys)
            target_names = [k for k in list(tag2idx.keys()) if k!='<PAD>']
            labels = [v for v in list(tag2idx.values()) if v!=0]
            report = classification_report(keys,preds, target_names = target_names, labels=labels,  digits=4, zero_division=0)
            f1_macro = f1_score(keys,preds, labels=labels,average = 'macro')
            print(headline)
            print(report)
            f.write(headline)
            f.write(report)
        
    return f1_macro,report
    #return sum(loss_list)/len(loss_list)

