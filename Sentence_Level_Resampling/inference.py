from net import Net
from Sentence_Level_Resampling.utils.dataset import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
# from trainer import eval
import torch
import numpy as np
import argparse
import os
from sklearn.metrics import classification_report,f1_score


##########################
def eval(model, iterator, epoch=None):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat, Y_hat_viterbi = [], [], [], [], [], []


    # crf = CRF(len(VOCAB))
    # crf.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    #loss_list= []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            
            words, x, is_heads, tags, y, seqlens, attention_mask = batch
            if i==0:
                print(words[0])
            logits_focal, logits_crf, y_, y_hat = model(x, y,attention_mask)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            
    all_preds = []
    # result_dir = os.path.join(os.getcwd(),"val_results.txt")
    # with open(result_dir, 'w+') as fout:
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        for x in range(len(preds)):
            if preds[x] == '<PAD>':
                preds[x] = 'O'
        all_preds.append(preds)
        # assert len(preds)==len(words.split())==len(tags.split()), "Sentence: {}\n True Tags: {}\n Pred Tags: {}".format(words,tags,preds)
        # for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
        #     fout.write("{} {} {}\n".format(w,t,p))
        # fout.write("\n")

    # y_true =  np.array([tag2idx[line.split()[1]] for line in open("results.txt", 'r').read().splitlines() if len(line) > 0])
    # y_pred =  np.array([tag2idx[line.split()[2]] for line in open("results.txt", 'r').read().splitlines() if len(line) > 0])

    # if epoch is not None:
    #     report = classification_report(y_true,y_pred, labels = [1,2,3,4,5,6,7,8,9], digits=4, zero_division=0)
    #     print(f"============Evaluation at Epoch={epoch}============")
    #     print(report)
    # os.remove("results.txt")

    return all_preds


##########################
def run_ner_infer(sent,model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    top_rnns=True
    model = Net(top_rnns, len(VOCAB), device, finetuning=True)
    if device == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    elif device == 'cuda':
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    train_texts,train_labels=[],[]
    for s in sent:
        train_texts.append(s.split())
        train_labels.append(['O']*len(s.split()))
    sents_train, tags_li_train = [], []
    for x in train_texts:
        sents_train.append(["[CLS]"] + x + ["[SEP]"])
    for y in train_labels:
        tags_li_train.append(["<PAD>"] + y + ["<PAD>"])

    train_dataset = NerDataset(sents_train, tags_li_train)

    infer_iter = torch.utils.data.DataLoader(dataset=train_dataset,
                             batch_size=32,
                             shuffle=False,
                             collate_fn = pad,
                             num_workers=0
                             )
    pred = eval(model, infer_iter)
    # for x in range(len(pred[0])):
    #     if pred[0][x] == '<PAD>':
    #         pred[0][x] = 'O'
    # return sent_infer[0][1:-1],pred[0][1:-1]
    return pred


def main(sent,model_path):
    # sent = ' '.join(sent)
    return run_ner_infer(sent,model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NER Inference')
    parser.add_argument("--file", required=True, type=str, help="Enter bangla sentences file")
    parser.add_argument("--output_file", required=True, type=str, help="Enter bangla sentences file")
    parser.add_argument("--model_path",
                         default="/mldata/NER/Sentence_Level_Resampling/banner_annotatedv2/model.pt",
                         metavar='PATH',
                         type=str,
                         help="pretrained model path")
    args = parser.parse_args()
    sen_list=[]
    with open(args.file,'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    sen_list.append(line)
                    
    # print(sen_list)              
    all_preds = main(sen_list,args.model_path)
    with open(args.output_file,'w',encoding='utf8') as f:
        for sen,tags in zip(sen_list,all_preds):
            tags = ' '.join(tags)
            f.write(sen+'\t'+tags[1:-2])
            f.write('\n')
            