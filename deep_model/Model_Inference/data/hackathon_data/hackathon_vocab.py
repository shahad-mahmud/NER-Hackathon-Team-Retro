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

LABEL_TO_ID = {tag: idx for idx, tag in enumerate(VOCAB)}
ID_TO_LABEL = {idx: tag for idx, tag in enumerate(VOCAB)}

