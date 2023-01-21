import src
import pickle
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file', help='Test file path')
    parser.add_argument('--model', default='data/results/crf.model', help='Test file path')
    
    args = parser.parse_args()
    
    test_data = src.utils.read_test_ner_data(args.test_file)
    test_data = [src.utils.line_to_features(line) for line in tqdm(test_data, dynamic_ncols=True, desc='Featurizing')]

    model = pickle.load(open(args.model, 'rb'))
    predictions = model.predict(test_data)
    
    with open('data/results/feature_model.txt', 'w') as res_file:
        for line in predictions:
            for tag in line:
                res_file.write(f'{tag}\n')
            res_file.write('\n')
    print('Predictions on test data saved on "data/results/feature_model.txt" file.')