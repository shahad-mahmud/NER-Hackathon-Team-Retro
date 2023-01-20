from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import json
from trainer import train,eval
from cost import crit_weights_gen
from net import Net
from dataset import NerDataset, VOCAB, pad
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from resample import Resampling
import argparse
import csv
import json
import logging
import os
import random
import sys
import wandb
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset)

# from seqeval.metrics import classification_report
# from model.xlmr_for_token_classification import (XLMRForTokenClassification, 
#                                                 XLMRForTokenClassificationWithCRF, Discriminator)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .json file (or other data files) for the task.")
     # parser.add_argument("--pretrained_path", default=None, type=str, required=True,
     #                     help="pretrained BanglaBert model path")
    parser.add_argument("--method",
                         default=None,
                         type=str,
                         choices = ['bus','sen_sample'],
                         help="The name of the sampling method to sample train data.")

    parser.add_argument("--sen_method",
                         default='sCRD',
                         type=str,
                         choices = ['sc','sCR','sCRD','nsCRD'],
                         help="The name of the sentence sampling method to sample train data.")
    parser.add_argument('--pretrained_model_path',
                        required=True,
                        type=str,
                        help='path of the pretrained model path')

    parser.add_argument("--wandb_run_id",
                         required=True,
                         type=str,
                         help="The name of the wandb run you want to resume.")

    parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     
    parser.add_argument("--top_rnns",
                         action='store_false',
                         help="Whether to add a BI-LSTM layer on top of BERT.")
    
    parser.add_argument("--fine_tuning",
                         action='store_false',
                         help="Whether to finetune the BERT weights.")
    
    parser.add_argument("--do_eval",
                         action='store_false',
                         help="Whether to run evaliation on test set or not.")
    
    parser.add_argument("--train_batch_size",
                         default=128,
                         type=int,
                         help="Total batch size for training.")
    
    parser.add_argument("--eval_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
    parser.add_argument("--start_epoch",
                         default=31,
                         type=int,
                         help="Starting Epoch number")
    parser.add_argument("--num_train_epochs",
                         default=10,
                         type=int,
                         help="Total number of additional training epochs to perform.")
    parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    train_batch_size = args.train_batch_size
    lr = args.learning_rate
    n_epochs = args.num_train_epochs
    top_rnns = args.top_rnns
    finetuning = args.fine_tuning
    trainset = args.data_dir
    method = args.method
    run_id = args.wandb_run_id
    if method == 'sen_sample':
        sen_method = args.sen_method
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = args.output_dir
    model = Net(top_rnns, len(VOCAB), device, finetuning)
    checkpoint = torch.load(args.pretrained_model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    with open(trainset) as infile:
        data = json.load(infile)

    new = data['train']
    train_texts, train_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
    new = data['validation']
    valid_texts, valid_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
    new = data['test']
    test_texts, test_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
    if method:
        if method == 'sen_sample':
            train_texts,train_labels = Resampling(train_texts, train_labels).resamp(sen_method)
        else:
            train_texts,train_labels = Resampling(train_texts, train_labels).BUS()
    sents_train, tags_li_train = [], []
    for x in train_texts:
        sents_train.append(["[CLS]"] + x + ["[SEP]"])
    for y in train_labels:
        tags_li_train.append(["<PAD>"] + y + ["<PAD>"])

    sents_valid, tags_li_valid = [], []
    for x in valid_texts:
        sents_valid.append(["[CLS]"] + x + ["[SEP]"])
    for y in valid_labels:
        tags_li_valid.append(["<PAD>"] + y + ["<PAD>"])

    sents_test, tags_li_test = [], []
    for x in test_texts:
        sents_test.append(["[CLS]"] + x + ["[SEP]"])
    for y in test_labels:
        tags_li_test.append(["<PAD>"] + y + ["<PAD>"])


    train_dataset = NerDataset(sents_train, tags_li_train)
    eval_dataset = NerDataset(sents_valid, tags_li_valid)
    test_dataset = NerDataset(sents_test, tags_li_test)

    train_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                                 batch_size= args.train_batch_size,
                                 shuffle=True,
                                 collate_fn=pad,
                                 num_workers=0
                                 )
    eval_iter = torch.utils.data.DataLoader(dataset=eval_dataset,
                                 batch_size=args.train_batch_size,
                                 shuffle=False,
                                 collate_fn = pad,
                                 num_workers=0
                                 )
    test_iter = torch.utils.data.DataLoader(dataset=test_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 collate_fn = pad,
                                 num_workers=0
                                 )

    optimizer = optim.Adam([
                                    {"params": model.fc.parameters(), "lr": 0.0005},
                                    {"params": model.bert.parameters(), "lr": 5e-5},
                                    {"params": model.rnn.parameters(), "lr": 0.0005},
                                    {"params": model.crf.parameters(), "lr": 0.0005}
                                    ],)
    num_train_optimization_steps = int(
            len(train_dataset) / train_batch_size ) * n_epochs
    # data_dist = [7237, 15684, 714867, 759, 20815, 9662, 8512, 37529, 70025]
    # crit_weights = crit_weights_gen(0.2, 0.9, data_dist)
    # crit_weights[-1]= .4
    # #insert 0 cost for ignoring <PAD>
    # crit_weights.insert(0,0)
    # crit_weights = torch.tensor(crit_weights).to(device)
    # criterion = nn.CrossEntropyLoss(weight=crit_weights)
    criterion = nn.CrossEntropyLoss()

    #last_loss = 100000000000
    #patience = 100
    #triggertimes = 0

    train_loss = []
    print("***** Running training *****")
    print("  Num examples = ", len(train_dataset))
    print("  Batch size = ", train_batch_size)
    print("  Num steps = ", num_train_optimization_steps)
    
    best_val_f1 = 0.0
    prev_val_f1 = 0.0
    test_f1_score = 0.0
    
    run = wandb.init(project="NER_sentence_sampling",\
          entity='mahtab-team',config=vars(args),id=run_id,resume='must')
    # run = wandb.init(project="test",entity='mahtab-team',config=vars(args))

    for epoch in range(args.start_epoch, args.start_epoch+n_epochs+1):
        loss_list = train(model, train_iter, optimizer, criterion, epoch, args)
        train_loss.extend(loss_list)
        wandb.log({'train_epoch_loss':np.mean(loss_list)})
                  
        #torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
        print("Evaluating on validation set...\n")
        
        f1,report = eval(model, eval_iter, criterion, args,epoch= epoch)
        if f1 > best_val_f1:
            best_val_f1 = f1
            print("\nFound better f1=%.4f on validation set. Saving model\n" %(f1))
            torch.save(model.state_dict(), open(os.path.join(output_dir, 'model.pt'), 'wb'))
            prev_val_f1 = f1

        else :
            if f1 < prev_val_f1:
                print("\nF1 score worse than the previous epoch: {}\n".format(f1))
                prev_val_f1 = f1
            else:
                prev_val_f1 = f1
                  
        
    print(f'Best F1 Score on validation set: {best_val_f1}')
    if args.do_eval:
        # load best/ saved model
        state_dict = torch.load(open(os.path.join(output_dir, 'model.pt'), 'rb'))
        model.load_state_dict(state_dict)
        print("Loaded saved model")

        model.to(device)

        print("***** Running evaluation on test set *****")
        print("  Num examples =", len(eval_dataset))
        print("  Batch size =", args.eval_batch_size)

        test_f1_score,report = eval(model, test_iter, criterion, args, is_test=True)

        print("\n%s", report)
        output_eval_file = os.path.join(output_dir, "test_results.txt")
        print("dataset = {}".format(args.data_dir))
        print("model = {}".format(args.output_dir))
        with open(output_eval_file, "w") as writer:
            print("***** Writing results to file after resuming training*****")
            writer.write(report)
    #             logger.info("Done.")
                  
    print(" Save best model weights in wandb")
    model_artifact = wandb.Artifact(
      f"model_{best_val_f1:.3f}", description=f"model-validation f1 macro:{best_val_f1} test f1 macro {test_f1_score}",
      metadata=dict(wandb.config), type = 'BERT-CRF')
    PATH=os.path.join(output_dir,'model.pt')
    model_artifact.add_file(PATH)                  
    run.log_artifact(model_artifact)
    wandb.finish()             
    plt.plot(train_loss)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.show()
    plt.savefig(os.path.join(output_dir,"loss.png"))



if __name__ == "__main__":
    main()    
    