import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        metavar='PATH',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json file (or other data files) for the task.")

    parser.add_argument("--method",
                        type=str,
                        choices=['bus', 'sen_sample', 'dau'],
                        help="The name of the sampling method to sample train data.")

    parser.add_argument("--sen_method",
                        default='nsCRD',
                        type=str,
                        choices=['sc', 'sCR', 'sCRD', 'nsCRD'],
                        help="The name of the sentence sampling method to sample train data.")

    parser.add_argument("--output_dir",
                        default=None,
                        metavar='PATH',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--banner_weighted",
                        action='store_true',
                        help="Whether to use banner's weighted cross entropy loss.")

    parser.add_argument("--top_rnns",
                        action='store_false',
                        help="Whether to add a BI-LSTM layer on top of BERT.")

    parser.add_argument("--fine_tuning",
                        action='store_false',
                        help="Whether to finetune the BERT weights.")

    parser.add_argument("--do_eval",
                        action='store_false',
                        help="Whether to run evaliation on test set or not.")

    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help="Number of consecutive decreasing or unchanged validation runs after which model is stopped running")

    # augmentation
    parser.add_argument("--task_name", default="development", type=str)
    parser.add_argument("--augmentation", type=str, nargs="+", default=[])
    parser.add_argument("--p_power", default=1.0, type=float,
                        help="the exponent in p^x, used to smooth the distribution, "
                             "if it is 1, the original distribution is used; "
                             "if it is 0, it becomes uniform distribution")
    parser.add_argument("--replace_ratio", default=0.3, type=float)
    parser.add_argument("--num_generated_samples", default=1, type=int)

    return parser.parse_args()
