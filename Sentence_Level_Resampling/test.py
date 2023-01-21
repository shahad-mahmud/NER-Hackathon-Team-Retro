import net
import utils
import torch
import argparse

from utils.dataset import NerDataset, pad, VOCAB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('model_weights')
    parser.add_argument('-o', '--output_file', default='.')
    parser.add_argument('--tags_given', action='store_true')
    parser.add_argument('--top_rnns', type=bool, default=True,)
    
    args = parser.parse_args()
    
    if args.tags_given:
        samples, tags = utils.read_test_data(args.file)
    else:
        samples = utils.read_test_data(args.file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = net.Net(args.top_rnns, len(utils.VOCAB), device, False)
    model = net.BertWithEmbeds(args.top_rnns, len(utils.VOCAB), device, False)
    model.load_state_dict(torch.load(args.model_weights), map_location=device)
    model.to(device)
    
    prepared_samples, _ = utils.prepare_samples(samples, [])
    