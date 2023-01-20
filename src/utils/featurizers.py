def line_to_features(line):
    features = []
    suffix_chars, prefix_chars = 3, 3
    
    for i, word in enumerate(line):
        feats = {
            'bias': 1.0,
            'word': word,
            'suffix': word[-suffix_chars:],
            'prefix': word[:prefix_chars]
        }
        
        if i==0:
            # feats['bos'] = True
            pass
        else:
            feats['last_word'] = line[i-1]
            feats['last_suffix'] = line[i-1][-suffix_chars:]
            feats['last_prefix'] = line[i-1][:prefix_chars]
        
        if i==len(line)-1:
            # feats['eos'] = True
            pass
        else:
            feats['next_word'] = line[i+1]
            feats['next_suffix'] = line[i+1][-suffix_chars:]
            feats['next_prefix'] = line[i+1][:prefix_chars]
        
        features.append(feats)
    
    return features