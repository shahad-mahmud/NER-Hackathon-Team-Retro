import numpy as np
import torch
from torch.utils import data
from transformers import AutoTokenizer
from normalizer import normalize
from tqdm import tqdm

#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
VOCAB = ['<PAD>', 'B-CORP',
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
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

# tag2idx = {'<PAD>':0,'B-Person': 1,
#  'I-Person': 2,
#  'B-Organization': 3,
#  'I-Organization': 4,
#  'B-Location': 5,
#  'I-Location': 6,
#  'B-GPE': 7,
#  'I-GPE': 8,
#  'B-Event': 9,
#  'I-Event': 10,
#  'B-Number': 11,
#  'I-Number': 12,
#  'B-Unit': 13,
#  'I-Unit': 14,
#  'B-D&T': 15,
#  'I-D&T': 16,
#  'B-T&T': 17,
#  'I-T&T': 18,
#  'B-Misc': 19,
#  'I-Misc': 20,
#  'O':21}
# idx2tag = {idx:tag for tag,idx in tag2idx.items()}
# VOCAB = list(tag2idx.keys())


class NerDataset(data.Dataset):
    def __init__(self, sents, tags_li):
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]

        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in (
                "[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        seqlen = len(y)

        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):

    def f(x): return [sample[x] for sample in batch]
    #x = f(1)
    #y = f(-2)
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    attention_mask = [[1.0] * len(labels) + [0.0]
                      * (maxlen - len(labels)) for labels in f(-2)]

    def f(x, seqlen): return [sample[x] + [0] *
                              (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens, torch.tensor(attention_mask, dtype=torch.float)


def remove_duplicates(train_texts, train_labels):
    unq_sen = {}
    for sen, label in zip(train_texts, train_labels):
        sen = ' '.join(sen)
        if sen not in unq_sen:
            unq_sen[sen] = label
    return [key.split() for key in unq_sen.keys()], [val for val in unq_sen.values()]


def prepare_samples(texts, labels):
    sentences, tags = [], []
    for x in texts:
        sentences.append(["[CLS]"] + x + ["[SEP]"])
    for y in labels:
        tags.append(["<PAD>"] + y + ["<PAD>"])

    return sentences, tags


def read_conll(file_in):
    words, labels = [], []
    examples = []
    # raw_examples = []
    # is_title = False
    with open(file_in, "r", encoding='utf-8') as fh:
        for line in tqdm(fh, desc=f'Reading {file_in}'):
            line = line.strip()

            if len(line) > 0:
                parts = line.split('_')
                word, label = parts[0].strip(), parts[-1].strip()

                word = normalize(word)

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


def read_test_data(file_path, with_tag: bool = False):
    samples = [[]]
    if with_tag:
        tags = [[]]

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            if not line and samples[-1]:
                samples.append([])
                if with_tag:
                    tags.append([])
                continue

            line = line.split()

            word = normalize(line[0])
            samples[-1].append(word)

            if with_tag:
                tags[-1].append(line[-1])

    if with_tag:
        return samples, tags
    return samples
