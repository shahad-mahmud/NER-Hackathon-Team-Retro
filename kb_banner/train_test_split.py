import collections
from deep_utils import stratify_train_test_split_multi_label
import numpy as np
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
    VOCAB = list(set([l for ins in examples for l in ins[1]]))
    VOCAB = sorted(VOCAB)
    return examples,VOCAB

def train_test_split(train_file_path):
    
    examples,VOCAB = read_conll(train_file_path)
    train_ex,train_l = list(zip(*examples))
    LABEL_TO_ID  = {k:i for i,k in enumerate(VOCAB)}
    all_sen_tags = []
    for words,tags in examples:
      count_arr = [0]*len(LABEL_TO_ID)
      for tag,count in collections.Counter(tags).items():
        if tag not in LABEL_TO_ID:
          print(tag)
        count_arr[int(LABEL_TO_ID[tag])] = count
      all_sen_tags.append(count_arr)
    

    x_train, x_test, y_train, y_test = stratify_train_test_split_multi_label(list(range(len(examples))), np.array(all_sen_tags), test_size=0.1, closest_ratio=False)
    all_train = [examples[i] for i in x_train]
    all_val = [examples[i] for i in x_test]
    return all_train,all_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file_path",
                         default='/root/nlp_hackathon_bd_2023/data/processed',
                         metavar='PATH',
                         type=str,
                         help="The train file dir")
    parser.add_argument("--output_dir",
                         default='/root/nlp_hackathon_bd_2023/data/new_preprocessed',
                         metavar='PATH',
                         type=str,
                         help="The train file dir")
    args  = parser.parse_args()
    all_train,all_val = train_test_split(args.train_file_path)
    with open(os.path.join(args.output_dir,'train.txt'),'w',encoding='utf-8') as f:
        for words,tags in all_train:
            for word,tag in zip(words,tags):
                f.write(word+' _'+' _ '+tag+'\n')
            f.write('\n')
    with open(os.path.join(args.output_dir,'val.txt'),'w',encoding='utf-8') as f:
        for words,tags in all_val:
            for word,tag in zip(words,tags):
                f.write(word+' _'+' _ '+tag+'\n')
            f.write('\n')
                    