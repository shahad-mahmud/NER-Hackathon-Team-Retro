from __future__ import absolute_import, division, print_function
import os
import net
import torch
import pickle
import random
import logging
import utils
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from trainer import train, eval
from resample import Resampling
from collections import Counter
from cost import crit_weights_gen

CUDA_LAUNCH_BLOCKING = "1"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = utils.get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    utils.check_dir(args.output_dir)
    
    train_examples = utils.read_conll(os.path.join(args.data_dir, 'train.txt'))
    val_examples = utils.read_conll(os.path.join(args.data_dir, 'val.txt'))
    test_examples = utils.read_conll(os.path.join(args.data_dir, 'test.txt'))

    train_texts, train_labels = list(zip(*train_examples))
    train_texts, train_labels = utils.remove_duplicates(train_texts, train_labels)
    
    valid_texts, valid_labels = list(zip(*val_examples))
    valid_texts, valid_labels = utils.remove_duplicates(valid_texts, valid_labels)
    
    test_texts, test_labels = list(zip(*test_examples))
    test_texts, test_labels = utils.remove_duplicates(test_texts, test_labels)
    
    if args.method == 'sen_sample':
        train_texts, train_labels = Resampling(train_texts, train_labels).resamp(args.sen_method)
    elif args.method == 'bus':
        train_texts, train_labels = Resampling(train_texts, train_labels).BUS()
    
    aug_dict = pickle.load(open('../data/raw/augmented_dict_0.pickle', 'rb'))

    sentences_train, tags_li_train = utils.prepare_samples(train_texts, train_labels, aug_dict)
    sentences_valid, tags_li_valid = utils.prepare_samples(valid_texts, valid_labels)
    sentences_test, tags_li_test = utils.prepare_samples(test_texts, test_labels)

    train_dataset = utils.NerDataset(sentences_train, tags_li_train)
    eval_dataset = utils.NerDataset(sentences_valid, tags_li_valid)
    test_dataset = utils.NerDataset(sentences_test, tags_li_test)

    train_iter = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=utils.pad,
        num_workers=0
    )
    eval_iter = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=utils.pad,
        num_workers=0
    )
    test_iter = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=utils.pad,
        num_workers=0
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model = net.Net(args.top_rnns, len(utils.VOCAB), device, args.fine_tuning)
    model = net.BertWithEmbeds(args.top_rnns, len(utils.VOCAB), device, args.fine_tuning)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    num_train_optimization_steps = int(len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    
    if args.banner_weighted:
        d = Counter([label for example in train_labels for label in example])
        data_dist = [d[key] for key in utils.tag2idx.keys() if key != '<PAD>']

        crit_weights = crit_weights_gen(0.5, 0.9, data_dist)
        # insert 0 cost for ignoring <PAD>
        crit_weights.insert(0, 0)
        crit_weights = torch.tensor(crit_weights).to(device)
        print(crit_weights)
        criterion = nn.CrossEntropyLoss(weight=crit_weights, ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    import curricular_face
    cost = curricular_face.CurricularFace(len(utils.VOCAB), len(utils.VOCAB))
    cost.to(device)


    print("***** Starting training *****")
    print("  Num examples = ", len(train_dataset))
    print("  Batch size = ", args.train_batch_size)
    print("  Num steps = ", num_train_optimization_steps)

    best_val_f1 = 0.0
    prev_val_f1 = 0.0
    test_f1_score = 0.0
    patience = 0
    
    run = wandb.init(project="banner", entity='shahad001', config=vars(args))

    train_loss = []
    for epoch in range(1, args.num_train_epochs+1):
        if epoch > 10:
            optimizer = optim.Adam([
                {"params": model.fc.parameters(), "lr": 0.0005},
                {"params": model.bert.parameters(), "lr": 5e-5},
                {"params": model.rnn.parameters(), "lr": 0.0005},
                {"params": model.crf.parameters(), "lr": 0.0005},
                {"params": cost.parameters(), "lr": 0.0005}
            ],)
        loss_list = train(model, train_iter, optimizer, cost, criterion, epoch, args)
        train_loss.extend(loss_list)
        
        wandb.log({'train_epoch_loss': np.mean(loss_list)})

        #torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
        print("Evaluating on validation set...\n")

        f1, report = eval(model, eval_iter, criterion, args, epoch=epoch)
        
        wandb.log({'validation_f1': f1})
        if f1 > best_val_f1:
            best_val_f1 = f1
            print("\nFound better f1=%.4f on validation set. Saving model\n" % (f1))

            torch.save(model.state_dict(), open(
                os.path.join(args.output_dir, 'model.pt'), 'wb'))
            patience = 0
            prev_val_f1 = f1
        else:
            if f1 < prev_val_f1:
                print("\nF1 score worse than the previous epoch: {}\n".format(f1))
                patience += 1
                prev_val_f1 = f1
            else:
                patience = 0
                prev_val_f1 = f1

        if patience >= args.patience:
            print(
                f"No more patience. Existing best model score: {best_val_f1}")
            break

    print(f'Best F1 Score on validation set: {best_val_f1}')
    if args.do_eval:
        # load best/ saved model
        state_dict = torch.load(
            open(os.path.join(args.output_dir, 'model.pt'), 'rb'))
        model.load_state_dict(state_dict)
        print("Loaded saved model")

        model.to(device)

        print("***** Running evaluation on test set *****")
        print("  Num examples =", len(eval_dataset))
        print("  Batch size =", args.eval_batch_size)

        test_f1_score, report, gt, pred = eval(
            model, test_iter, criterion, args, is_test=True)

        print("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        print("dataset = {}".format(args.data_dir))
        print("model = {}".format(args.output_dir))
        with open(output_eval_file, "w") as writer:
            print("***** Writing results to file *****")
            writer.write(report)

        # save ground truth and predictions
        gt_file_path = os.path.join(args.output_dir, "eval_gt_labels.txt")
        pred_file_path = os.path.join(args.output_dir, "eval_pred_labels.txt")

        with open(gt_file_path) as gt_file, open(pred_file_path) as pred_file:
            for g, p in zip(gt, pred):
                gt_file.write(f"{g}\n")
                pred_file.write(f"{p}\n")

    print(" Save best model weights in wandb")
    model_artifact = wandb.Artifact(
        f"model_{best_val_f1:.3f}", description=f"model-validation f1 macro:{best_val_f1} test f1 macro {test_f1_score}",
        metadata=dict(wandb.config), type='BERT-CRF')
    PATH = os.path.join(args.output_dir, 'model.pt')
    model_artifact.add_file(PATH)
    run.log_artifact(model_artifact)

    wandb.finish()
    plt.plot(train_loss)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.show()
    plt.savefig(os.path.join(args.output_dir, "loss.png"))


if __name__ == "__main__":
    main()
