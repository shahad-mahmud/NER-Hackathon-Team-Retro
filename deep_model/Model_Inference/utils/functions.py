from utils.get_attr_from_file import get_attr_file
import pickle

def import_vocab(tagset,model):
    
    if tagset=='banner':
        VOCAB = get_attr_file('data.banner.banner_vocab.VOCAB')
            
    elif tagset=='annotated':
        
        VOCAB = get_attr_file('data.annotated_data.annotated_vocab.VOCAB')
        
    elif tagset=='validated':
        
        VOCAB = get_attr_file('data.validated_data.validated_vocab.VOCAB')
    
    elif tagset=='hackathon':
        
        VOCAB = get_attr_file('data.hackathon_data.hackathon_vocab.VOCAB')

    if model == 'banner':
        VOCAB = ['<PAD>'] + VOCAB
    with open('data/VOCAB.pickle','wb') as f:
        pickle.dump(VOCAB,f)
    
    
def get_vocab():
    with open('data/VOCAB.pickle','rb') as f:
        VOCAB = pickle.load(f)  
    LABEL_TO_ID = {tag: idx for idx, tag in enumerate(VOCAB)}
    ID_TO_LABEL = {idx: tag for idx, tag in enumerate(VOCAB)}
    
    return VOCAB,LABEL_TO_ID,ID_TO_LABEL
    
def read_conll(file_in,output_file):
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
    
    with open(output_file,'w',encoding='utf8') as data_output_file:
        for words,labels in examples:
            data_output_file.write(' '.join(words)+'\t'+' '.join(labels))
            data_output_file.write('\n')
    return examples


def read_conll_without_label(file_in,output_file):
    words, labels = [], []
    examples = []
    # raw_examples = []
    # is_title = False
    with open(file_in, "r",encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if len(line) > 0:
                word  = line
                words.append(word)
                labels.append("O")
            else:
                if len(words) > 0:
                    assert len(words) == len(labels)
                    # raw_examples.append([words,labels])
                    examples.append([words, labels])
                    words, labels = [], []
    
    with open(output_file,'w',encoding='utf8') as data_output_file:
        for words,labels in examples:
            data_output_file.write(' '.join(words)+'\t'+' '.join(labels))
            data_output_file.write('\n')
    return examples


def convert_sub_format(file_in,file_out):
    all_labels = []
    with open(file_in, "r",encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            _,labels = line.split('\t')
            labels = labels.split()
            all_labels.append(labels)
    with open(file_out, "a+",encoding='utf-8') as fo:
        for label_list in all_labels:
            for l in label_list:
                fo.write(l)
                fo.write('\n')
            fo.write('\n')
     