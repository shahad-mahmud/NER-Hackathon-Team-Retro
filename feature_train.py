import os
import src
import yaml
import pickle
import argparse

import sklearn_crfsuite
from sklearn import metrics
from tqdm import tqdm
from typing import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    
    args = parser.parse_args()
    configs = yaml.safe_load(open(args.config_file))
    
    train_data = src.utils.read_ner_data(configs['train_data_path'])
    valid_data = src.utils.read_ner_data(configs['valid_data_path'])
    # test_data = src.utils.read_ner_data(configs['test_data_path'])
    test_data = src.utils.read_test_ner_data(configs['test_data_path'])
    
    train_features = [src.utils.line_to_features(line) for line in tqdm(train_data[0], dynamic_ncols=True)]
    valid_features = [src.utils.line_to_features(line) for line in tqdm(valid_data[0], dynamic_ncols=True)]
    # test_features = [src.utils.line_to_features(line) for line in tqdm(test_data[0], dynamic_ncols=True)]
    test_features = test_data

    train_labels = train_data[1]
    valid_labels = valid_data[1]
    # test_labels = test_data[1]
    
    model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.15,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    print('Training the model...')
    model.fit(train_features, train_labels)
    pickle.dump(model, open('data/results/crf.model', 'wb'))
    
    print('Training completed. Predicting on validation set...')
    valid_predictions = model.predict(valid_features)
    
    labels = list(model.classes_)
    valid_labels = src.utils.flatten(valid_labels)
    valid_predictions = src.utils.flatten(valid_predictions)
    
    score = metrics.f1_score(valid_labels, valid_predictions, average='macro', labels=labels)
    print('Macro f1 on validation data:', score)
    
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    
    report = metrics.classification_report(valid_labels, valid_predictions, labels = sorted_labels, digits=3)
    print(f"\nClassification report on validation data:\n{report}")
    
    # understanding feature impacts
    print('Top positive features:')
    src.utils.show_feature_impacts(Counter(model.state_features_).most_common(15))
    
    print('\nTop negative features:')
    src.utils.show_feature_impacts(Counter(model.state_features_).most_common()[-15:])
    
    # predict on test data
    test_predictions = model.predict(test_features)
    with open('data/results/feature_model.txt', 'w') as res_file:
        for line in test_predictions:
            for tag in line:
                res_file.write(f'{tag}\n')
            res_file.write('\n')
    print('Predictions on test data saved on "data/results/feature_model.txt" file.')