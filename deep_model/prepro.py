import truecase,json
import re
from transformers import AutoTokenizer
import os
VOCAB = ['B-CORP',
 'B-CW',
 'B-GRP',
 'B-LOC',
 'B-PER',
 'B-PROD',
 'I-CORP',
 'I-CW',
 'I-GRP',
 'I-LOC',
 'I-PER',
 'I-PROD',
 'O']

# VOCAB = ('<PAD>', 'I-LOC', 'B-ORG', 'O', 'I-OBJ', 'I-PER', 'B-OBJ', 'I-ORG', 'B-LOC', 'B-PER')
LABEL_TO_ID = {tag: idx for idx, tag in enumerate(VOCAB)}
ID_TO_LABEL = {idx: tag for idx, tag in enumerate(VOCAB)}

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


def process_instance(words, labels, tokenizers, args, max_seq_length=512,is_banner=False,index = None):
    tokens, token_labels,input_ids = [[]]*args.n_model, [[]]*args.n_model, [[]]*args.n_model
    # if is_banner:
    #     LABEL_TO_ID['<PAD>'] = 9
    # outputs={}
    for i in range(args.n_model):
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
        'n_models':args.n_model
    }


def read_conll(file_in):
    words, labels = [], []
    examples = []
    # raw_examples = []
    # is_title = False
    with open(file_in, "r",encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            
            if len(line) > 0:
                parts  = line.split('_')
                word,label = parts[0].strip(),parts[-1].strip()
                
                words.append(word)
                labels.append(label)
            else:
                if len(words) > 0:
                    assert len(words) == len(labels)
                    # raw_examples.append([words,labels])
                    examples.append([words, labels])
                    words, labels = [], []
    # VOCAB = list(set([l for ins in examples for l in ins[1]]))
    # VOCAB = sorted(VOCAB)
    return examples

def read_banner(file_in, args, is_banner = False, datatype='train', max_seq_length=512):
    examples = []
    raw_examples = read_conll(os.path.join(file_in,datatype+'.txt'))
    if is_banner:
        with open('./config.json','w') as f:
            data = {'is_banner':True}
            json.dump(data,f)
    else:
        with open('./config.json','w') as f:
            data = {'is_banner':False}
            json.dump(data,f)
    words_list, labels_list = list(zip(*raw_examples))
    # words_list = words_list[:1000]
    # labels_list = labels_list[:1000]
    tokenizers=[]
    for i in range(args.n_model):
        tokenizers.append(AutoTokenizer.from_pretrained(args.model_name_or_path[i]))
    for i,(words,labels) in enumerate(zip(words_list,labels_list)):
        # raw_examples.append([words,labels])
        # print(words,labels)
        
        examples.append(process_instance(words, labels, tokenizers, args, max_seq_length, is_banner,i))
    return examples
