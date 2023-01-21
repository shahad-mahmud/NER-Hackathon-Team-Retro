from sklearn.metrics import classification_report, f1_score
from cost import crit_weights_gen
from collections import Counter
from prepro import process_instance, LABEL_TO_ID, ID_TO_LABEL
from utils import set_seed, collate_func
import json
from model import NLLModel
from torch.optim import Adam
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import os
import argparse
CUDA_LAUNCH_BLOCKING = "1"


class ArgsClass:
    def __init__(self):
        self.data_dir = './pred_sen_list.txt'
        self.pretrained_path = "/mldata/NER/Noise_Label_NER/ner/noisy_banner_optim_annot_v2/banglabert_1_model.pt"
        self.is_banner = True
        self.weighted = True
        self.top_rnns = True
        self.fine_tuning = True
        self.model_name_or_path = [
            "csebuetnlp/banglabert", "csebuetnlp/banglabert"]
        self.max_seq_length = 512
        self.dropout_prob = 0.1
        self.batch_size = 1
        self.seed = 42
        self.num_class = len(LABEL_TO_ID)
        self.n_model = len(self.model_name_or_path)
        self.alpha = 50
        self.alpha_warmup_ratio = 0.1
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.file_or_sentence = 'sentence'
        self.sen = "২৮ সেপ্টেম্বর দুবাইতে অনুষ্ঠিত হবে এই ১৫ সেপ্টেম্বর শ্রীলঙ্কার বিপক্ষে এবং ২০ সেপ্টেম্বর আফগানিস্তানের বিপক্ষে খেলবে"


def evaluate(args, model, features, words_list):

    dataloader = DataLoader(features, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_func, drop_last=False)
    preds, keys = [], []

    for sen_id, batch in enumerate(dataloader):
        model.eval()
        # print(batch['labels'])
        batch_labels = batch['labels'][int(
            args.best_model_index)].cpu().numpy().flatten().tolist()

        with torch.no_grad():
            logits = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'])
            # logits = model(**batch)[0]
        sen_preds = np.argmax(
            logits[int(args.best_model_index)][:, :-1].cpu().numpy(), axis=-1).tolist()
        assert len(sen_preds) == len(batch_labels)
        sen_preds = [pred for i, pred in enumerate(
            sen_preds) if batch_labels[i] != len(LABEL_TO_ID)]
        sen_preds = [ID_TO_LABEL[l] for l in sen_preds]
        batch_labels = [
            label for label in batch_labels if label != len(LABEL_TO_ID)]
        keys.append(batch_labels)
        # print(sen_id,len(words_list),len(sen_preds))

        assert len(sen_preds) == len(batch_labels) == len(words_list[sen_id])
        # print(sen_preds,words_list[sen_id])
        preds.append(sen_preds)

    with open(os.path.join(args.output_dir, 'prediction.txt'), 'w', encoding='utf8') as f:
        for tokens, tags in zip(words_list, preds):
            f.write(' '.join(tokens)+"\t"+" ".join(tags))
            f.write("\n")
    if args.labels_provided:
        preds = [LABEL_TO_ID[p] for pred_list in preds for p in pred_list]
        keys = [k for key_list in keys for k in key_list]
        # print(len(preds),len(keys))

        assert len(preds) == len(keys)
        with open(os.path.join(args.output_dir, "prediction_report.txt"), 'w') as f:
            report = classification_report(keys, preds, target_names=list(
                LABEL_TO_ID.keys()), labels=list(LABEL_TO_ID.values()),  digits=4, zero_division=0)
            f1_macro = f1_score(keys, preds, labels=list(
                LABEL_TO_ID.values()), average='macro')
            print(report)
            f.write(report)

    # print(preds,keys)


def main(args):

    args.alpha_t = 0.0
    # wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    data_type = args.file_or_sentence
    # set_seed(args)
    # print(args)

    tokenizers = []
    for i in range(args.n_model):
        tokenizers.append(AutoTokenizer.from_pretrained(
            args.model_name_or_path[i]))
    words_list = []
    labels_list = []

    if data_type == 'sentence':
        if args.labels_provided:
            sen, tags = args.sen.split('\t')
            words_list.append(sen.split())
            labels_list.append(tags.split())
        else:
            words_list = [args.sen.split()]
            labels_list = [['O']*len(args.sen.split())]
    else:
        with open(args.data_dir, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    if args.labels_provided:
                        sen, tags = line.split('\t')
                        words_list.append(sen.split())
                        labels_list.append(tags.split())
                    else:
                        words_list.append(line.split())
                        labels_list.append(['O']*len(line.split()))

    # print(words_list,labels_list)
    train_features = []

    for words, labels in zip(words_list, labels_list):

        train_features.append(process_instance(
            words, labels, tokenizers, args, args.max_seq_length, args.is_banner))

    # first weights are for banner dataset
    # crit_weights=[0.862590592787226,0.813810527531545,0.5,0.9,0.784179828487281,0.8485866081482979,0.8552276730286144,0.6876591690006641, 0.5,0]
    crit_weights = [0.5, 0.6126589945487584, 0.6241671714112659, 0.6263476680799516, 0.8200484554815264, 0.8202907328891581, 0.6829194427619625, 0.8533615990308905, 0.8241671714112659, 0.842580254391278,
                    0.5822531798909751, 0.8147183525136281, 0.8307086614173229, 0.9, 0.7407026044821321, 0.7270139309509388, 0.8766202301635373, 0.8762568140520897, 0.8681405208964265, 0.8915202907328892, 0.5, 0]
    crit_weights = torch.tensor(crit_weights).to(device)
    # print(crit_weights)
    model = NLLModel(args, crit_weights)
    PATH = torch.load(args.pretrained_path)
    model.load_state_dict(PATH)
    evaluate(args, model, train_features, words_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_or_sentence', '-fs',
                        default='sentence',
                        choices=['sentence', 'file'],
                        required=True,
                        type=str,
                        help='Whether to take input a file containing one sentence per line or single sentence')
    parser.add_argument('--best_model_index',
                        default='0',
                        type=str,
                        help='The best model index among noisy label models')

    parser.add_argument('--sen',
                        default=None,
                        type=str,
                        help='pass a single sentence')
    parser.add_argument("--data_dir",
                        default='./pred_sen_list.txt',
                        metavar='PATH',
                        type=str,
                        help="The input data dir if a file is given")

    parser.add_argument("--output_dir",
                        default='/mldata/NER/Noise_Label_NER/ner/noisy_banner_optim_annot_v2',
                        metavar='PATH',
                        type=str,
                        help="The output file name for predictions")

    parser.add_argument("--is_banner",
                        action='store_true',
                        help="Whether to use banner model or vanilla bert model like in noisy label paper.")

    parser.add_argument("--weighted",
                        action='store_true',
                        help="Whether to use weighted cross entropy or not.")

    parser.add_argument("--labels_provided",
                        action='store_true',
                        help="Whether labels are provided with sentence")

    parser.add_argument("--top_rnns",
                        action='store_false',
                        help="Whether to add a BI-LSTM layer on top of BERT in Banner Model.")

    parser.add_argument("--fine_tuning",
                        action='store_false',
                        help="Whether to finetune the BERT weights in Banner Model.")

    parser.add_argument("--model_name_or_path", type=str, nargs="+",
                        default=["csebuetnlp/banglabert", "csebuetnlp/banglabert"])
    parser.add_argument(
        "--pretrained_path", default="./noisy_output_banner_optim_cleaned/model.pt", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=len(LABEL_TO_ID))

    # parser.add_argument("--project_name", type=str, default="Noisy-Label-NER")
    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    args = parser.parse_args()

    # args = ArgsClass()
    main(args)
