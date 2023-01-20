from __future__ import absolute_import, division, print_function
CUDA_LAUNCH_BLOCKING="1"
import argparse, json, logging, os, random, sys, torch
from aug_dataclass import ConllCorpus
from augment import get_category2mentions, get_label2tokens
import torch.nn as nn
from trainer import train,eval
from cost import crit_weights_gen
from net import Net, BertWithEmbeds
from collections import Counter
from dataset import NerDataset, VOCAB, pad, remove_duplicates, tag2idx, read_conll
from resample import Resampling
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import csv
import wandb
# import mlflow
# import mlflow.sklearn
import math
import numpy as np
import torch.nn.functional as F
# from transformers import AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset)
from augment import generate_sentences_by_shuffle_within_segments, generate_sentences_by_replace_mention, generate_sentences_by_replace_token, generate_sentences_by_synonym_replacement


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
                         metavar='PATH',
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .json file (or other data files) for the task.")
     # parser.add_argument("--pretrained_path", default=None, type=str, required=True,
     #                     help="pretrained BanglaBert model path")
    
    parser.add_argument("--method",
                         type=str,
                         choices = ['bus','sen_sample','dau'],
                         help="The name of the sampling method to sample train data.")

    parser.add_argument("--sen_method",
                         default='nsCRD',
                         type=str,
                         choices = ['sc','sCR','sCRD','nsCRD'],
                         help="The name of the sentence sampling method to sample train data.")

    parser.add_argument("--output_dir",
                         default=None,
                         metavar='PATH',
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
    # Other parameters
    # parser.add_argument("--cache_dir",
    #                     default="",
    #                     type=str,
    #                     help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--banner_weighted",
                         action='store_true',
                         help="Whether to use banner's weighted cross entropy loss.")

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
                         default=64,
                         type=int,
                         help="Total batch size for training.")
    
    parser.add_argument("--eval_batch_size",
                         default=64,
                         type=int,
                         help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                         default=50,
                         type=int,
                         help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
    parser.add_argument('--patience',
                         type=int,
                         default=10,
                         help="Number of consecutive decreasing or unchanged validation runs after which model is stopped running")
    
    # augmentation
    parser.add_argument("--task_name", default="development", type=str) 
    parser.add_argument("--augmentation", type=str, nargs="+", default=[])
    parser.add_argument("--p_power", default=1.0, type=float,
                        help="the exponent in p^x, used to smooth the distribution, "
                             "if it is 1, the original distribution is used; "
                             "if it is 0, it becomes uniform distribution")
    parser.add_argument("--replace_ratio", default=0.3, type=float)
    parser.add_argument("--num_generated_samples", default=1, type=int)


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
    train_examples = read_conll(os.path.join(trainset,'train.txt'))
    val_examples = read_conll(os.path.join(trainset,'val.txt'))
    test_examples = read_conll(os.path.join(trainset,'test.txt'))
                              
    train_texts, train_labels = [],[]

    if method == 'dau':
        corpus = ConllCorpus(args.task_name, data)
        category2mentions = get_category2mentions(corpus.train)
        label2tokens = get_label2tokens(corpus.train, args.p_power)
        print(f"# sentences before augmenting: {len(corpus.train)}")
        # print(f"# sentences in development set: {len(corpus.dev)}")

        train_data = corpus.train
        if len(args.augmentation) > 0:
            augmented_sentences = []
            for s in train_data:
                if "MR" in args.augmentation:
                    augmented_sentences += generate_sentences_by_replace_mention(s, category2mentions, args.replace_ratio,
                                                                                args.num_generated_samples)
                if "LwTR" in args.augmentation:
                    augmented_sentences += generate_sentences_by_replace_token(s, label2tokens, args.replace_ratio,
                                                                            args.num_generated_samples)
                if "SiS" in args.augmentation:
                    augmented_sentences += generate_sentences_by_shuffle_within_segments(s, args.replace_ratio,
                                                                                        args.num_generated_samples)
                if "SR" in args.augmentation:
                    augmented_sentences += generate_sentences_by_synonym_replacement(s, args.replace_ratio,
                                                                                    args.num_generated_samples)
            train_data += augmented_sentences
        else:
            print("No data augmentation used")
        
        for sentence in train_data:
            train_texts.append([t.text for t in sentence.tokens])
            train_labels.append([t.get_label('gold') for t in sentence.tokens])
        print("-" * 100)
        print(f"# sentences after augmentation: {len(train_data)}")
    else:  
        train_texts,train_labels = list(zip(*train_examples))
        # train_texts, train_labels = list(zip(*map(lambda d: (d['sentence'], d['iob_tags']), new)))
        train_texts,train_labels = remove_duplicates(train_texts,train_labels)
        if method == 'sen_sample':
            sen_method = args.sen_method
            train_texts,train_labels = Resampling(train_texts, train_labels).resamp(sen_method)
        elif method == 'bus':
            train_texts,train_labels = Resampling(train_texts, train_labels).BUS()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = args.output_dir
    # model = Net(top_rnns, len(VOCAB), device, finetuning)
    model = BertWithEmbeds(top_rnns, len(VOCAB), device, finetuning)
    #model.load_state_dict(torch.load('models/banner_model.pt'))
    model.to(device)
    valid_texts, valid_labels = list(zip(*val_examples))
    valid_texts,valid_labels = remove_duplicates(valid_texts,valid_labels)
    test_texts, test_labels = list(zip(*test_examples))
    test_texts,test_labels = remove_duplicates(test_texts,test_labels)  
                               
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

    optimizer = optim.Adam(model.parameters(), lr = lr)
    num_train_optimization_steps = int(
            len(train_dataset) / train_batch_size ) * n_epochs
    run = wandb.init(project="banner",entity='shahad001',config=vars(args))
    if args.banner_weighted:
        d = Counter([label for example in train_labels for label in example])
        data_dist = [d[key] for key in tag2idx.keys() if key!='<PAD>']
        # data_dist = [7237, 15684, 714867, 759, 20815, 9662, 8512, 37529, 70025]
        crit_weights = crit_weights_gen(0.5, 0.9, data_dist)
        #insert 0 cost for ignoring <PAD>
        crit_weights.insert(0,0)
        crit_weights = torch.tensor(crit_weights).to(device)
        print(crit_weights)
        criterion = nn.CrossEntropyLoss(weight = crit_weights,ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

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
    patience = 0
    
    for epoch in range(1, n_epochs+1):
        
        if epoch>10:
            optimizer = optim.Adam([
                                    {"params": model.fc.parameters(), "lr": 0.0005},
                                    {"params": model.bert.parameters(), "lr": 5e-5},
                                    {"params": model.rnn.parameters(), "lr": 0.0005},
                                    {"params": model.crf.parameters(), "lr": 0.0005}
                                    ],)
        loss_list = train(model, train_iter, optimizer, criterion, epoch, args)
        train_loss.extend(loss_list)
        wandb.log({'train_epoch_loss':np.mean(loss_list)})
                  
        #torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
        print("Evaluating on validation set...\n")
        
        f1,report = eval(model, eval_iter, criterion, args,epoch= epoch)
        wandb.log({'validation_f1':f1})
        if f1 > best_val_f1:
            best_val_f1 = f1
            print("\nFound better f1=%.4f on validation set. Saving model\n" %(f1))
            # print("\n%s\n" %(report))

            torch.save(model.state_dict(), open(os.path.join(output_dir, 'model.pt'), 'wb'))
            patience=0
            prev_val_f1 = f1

        else :
            if f1 < prev_val_f1:
                print("\nF1 score worse than the previous epoch: {}\n".format(f1))
                patience+=1
                prev_val_f1 = f1
            else:
                patience = 0
                prev_val_f1 = f1
                  
        if patience >= args.patience:
                print(f"No more patience. Existing best model score: {best_val_f1}")
                break
                  
        #if eval_loss > last_loss:
        #    trigger_times += 1

        #   if trigger_times >= patience:
        #        break
        #else:
        #    trigger_times = 0

        #last_loss = eval_loss

#         if epoch == 10 or epoch == 20:
#             fname = os.path.join(output_dir, str(epoch))
#             torch.save(model.state_dict(), f"{fname}.pt")
    
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
            print("***** Writing results to file *****")
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
    # experiment_name = 'Banner with annotated data v1'
    # mlflow.create_experiment(experiment_name,best_val_f1,test_f1_score,model_path=PATH,run_name =None,run_metrics=None,model=None,confusion_metrics_path = None, run_params = None)
    # mlflow.set_experiment(experiment_name)
    # with mlflow.start_run():
    #     mlflow.log_metric(loss_list, "loss")
    #     mlflow.sklearn.log_model(PATH,"model")
    #     mlflow.log_artifect(f"model-validation f1 macro:{best_val_f1} test f1 macro {test_f1_score}","best_val_f1")             
    plt.plot(train_loss)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.show()
    plt.savefig(os.path.join(output_dir,"loss.png"))



if __name__ == "__main__":
    main()    
    