import src
from bnlp import POS
from bnlp.corpus import stopwords

pos = POS()
model_path = 'data/models/bn_pos.pkl'

def line_to_features(line):
    features = []
    suffix_chars, prefix_chars = 4, 3
    
    poses = pos.tag(model_path, line)
    # poses = [('a','a')] * len(line)
    # print(poses)

    for i, word in enumerate(line):
        feats = {
            'emphasis': 0.8,
            'word': word,
            'suffix': word[-suffix_chars:],
            'prefix': word[:prefix_chars],
            'word_len': len(word),
            'pos': poses[i][1]
        }

        # put less emphasis on punctuations
        if word in ['!', '"', "'", '(', ')', ',', '-', '.', ':', ':-', ';', '<', '=', '>', '?', '[', ']', '{', '}', 'ʼ', '।', '॥', '–', '—', '‘', '’', '“', '”', '…', '′', '″', '√', '/', '_', '|']:
            feats['emphasis'] = 0.0
        
        if src.utils.is_digit(word):
            # not setting zero as there are 354 samples where a number contains NE tag
            feats['emphasis'] = 0.3
        
        if word in stopwords:
            feats['emphasis'] = 0.3

        if i > 0:
            feats['last_word'] = line[i-1]
            feats['last_suffix'] = line[i-1][-suffix_chars:]
            feats['last_prefix'] = line[i-1][:prefix_chars]
            feats['last_word_len'] = len(line[i-1])
            feats['last_pos'] = poses[i-1][1]
            
            if line[i-1] == 'তিনি':
                feats['emphasis'] += 0.15
            if line[i-1] == 'হল':
                feats['emphasis'] += 0.05


        if i < len(line)-1:
            feats['next_word'] = line[i+1]
            feats['next_suffix'] = line[i+1][-suffix_chars:]
            feats['next_prefix'] = line[i+1][:prefix_chars]
            feats['next_word_len'] = len(line[i+1])
            feats['next_pos'] = poses[i+1][1]

        features.append(feats)

    return features

def show_feature_impacts(states):
    for (attribution, label), weight in states:
        print("%0.6f %-8s %s" % (weight, label, attribution))
