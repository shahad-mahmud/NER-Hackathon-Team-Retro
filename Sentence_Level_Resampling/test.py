import net
import torch
import argparse

from Sentence_Level_Resampling.utils.dataset import NerDataset, pad, VOCAB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('model_weights')
    parser.add_argument('-o', '--output_file', default='.')
    parser.add_argument('--tags_given', action='store_true')
    
    args = parser.parse_args()
    