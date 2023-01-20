import os
import src
import yaml
import argparse

import sklearn_crfsuite
from sklearn import metrics
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    
    args = parser.parse_args()
    configs = yaml.safe_load(open(args.config_file))
    
    train_data = src.utils.read_ner_data(configs['train_data_path'])
    valid_data = src.utils.read_ner_data(configs['valid_data_path'])
    test_data = src.utils.read_ner_data(configs['test_data_path'])
    
    train_features = [src.utils.line_to_features(line) for line in tqdm(train_data[0], dynamic_ncols=True)]
    test_features = [src.utils.line_to_features(line) for line in tqdm(test_data[0], dynamic_ncols=True)]

    train_labels = train_data[1]
    test_labels = test_data[1]
    
    model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    model.fit(train_features, train_labels)
    preds = model.predict(test_features)
    
    labels = list(model.classes_)
    test_labels = src.utils.flatten(test_labels)
    preds = src.utils.flatten(preds)
    
    score = metrics.f1_score(test_labels, preds, average='macro', labels=labels)
    print('Macro f1:', score)
    
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    
    report = metrics.classification_report(test_labels, preds, labels = sorted_labels, digits=3)
    print(f"\nClassification report:\n{report}")