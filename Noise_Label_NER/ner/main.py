CUDA_LAUNCH_BLOCKING="1"
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from model import NLLModel
from utils import set_seed, collate_func
from prepro import read_banner, LABEL_TO_ID
from torch.cuda.amp import autocast, GradScaler
# import seqeval.metrics
from sklearn.metrics import classification_report,f1_score
import wandb


ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()
    
    print("***** Running training *****")
    print("  Num examples = ", len(train_features))
    print("  Batch size = ", args.batch_size)
    print("  Num steps = ", total_steps)
    
    best_val_f1 = 0.0
    prev_val_f1 = -1.0
    test_f1_score = 0.0
    val_f1 = 0.0
    patience = 0
    

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        loss_list = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
            
            with autocast():
                outputs = model(input_ids = batch['input_ids'].to(args.device),
                                attention_mask = batch['attention_mask'].to(args.device),
                                labels = batch['labels'].to(args.device),epoch = epoch)
            loss = outputs[0] / args.gradient_accumulation_steps
            loss_list.append(loss.item())
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'train_step_loss': loss.item()}, step=num_steps)
            if step%100 == 0:
                print(f'Step: {step} and train loss: {loss}')
            if step==0:
                print("==============Check Dataloader===============")
                print("tokens:", batch['tokens'][0])
                print("tags:", batch['tags'][0])
                print("x:", batch['input_ids'][0])
                print("labels:", batch['labels'][0])
                print("=============================================")
            if step == len(train_dataloader) - 1:
                for tag, features in benchmarks:
                    results = evaluate(args, model, features, epoch, tag=tag)
                    if tag == 'dev':
                        val_f1 = results[f'{tag}_f1']     
                    wandb.log(results, step=num_steps)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("\nFound better f1=%.4f on validation set. Saving model\n" %(val_f1))
            # print("\n%s\n" %(report))

            torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
            patience=0
            prev_val_f1 = val_f1

        else :
            if val_f1 < prev_val_f1:
                print("\nF1 score worse than the previous epoch: {}\n".format(val_f1))
                patience+=1
                prev_val_f1 = val_f1
            else:
                patience = 0
                prev_val_f1 = val_f1
                
        if patience >= args.patience:
                print(f"No more patience. Existing best model score: {best_val_f1}")
                break

        loss_epoch = np.mean(loss_list)
        wandb.log({'train_loss_epoch':loss_epoch})
    print(f'Best F1 Score on validation set: {best_val_f1}')
    state_dict = torch.load(open(os.path.join(args.output_dir, 'model.pt'), 'rb'))
    model.load_state_dict(state_dict)
    print("Loaded saved model")

    model.to(args.device)
    for tag, features in benchmarks:
        evaluate(args, model, features,tag=tag, stage = 'final')
    
    print(" Save best model weights in wandb")
    model_artifact = wandb.Artifact(
      f"model_{best_val_f1:.3f}", description=f"model-validation f1 macro:{best_val_f1} test f1 macro {test_f1_score}",
      metadata=dict(wandb.config), type = 'BERT-CRF')
    PATH=os.path.join(args.output_dir,'model.pt')
    model_artifact.add_file(PATH)                  
    wandb.log_artifact(model_artifact)
    wandb.finish()             


def evaluate(args, model, features, epoch=None, stage = None, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, drop_last=False)
    preds, keys = [], []
    for batch in dataloader:
        model.eval()
        # batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()
        batch['labels'] = None
        with torch.no_grad():
            logits = model(input_ids = batch['input_ids'].to(args.device),
                                attention_mask = batch['attention_mask'].to(args.device))[0]
            # logits = model(**batch)[0]
            if args.is_banner:
                preds += np.argmax(logits[:,:-1].cpu().numpy(), axis=-1).tolist()
            else:
                preds += np.argmax(logits.cpu().numpy(), axis=-1).tolist()
    if args.is_banner:
        preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != len(LABEL_TO_ID)]))
    else:
        preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))
    print(set(preds))
    # preds = [ID_TO_LABEL[pred] for pred in preds]
    # keys = [ID_TO_LABEL[key] for key in keys]
    model.zero_grad()
    # f1 = seqeval.metrics.f1_score([keys], [preds])
    
    with open(os.path.join(args.output_dir,f"{tag}_report_{stage if stage else ''}.txt"), 'a') as f:
        if epoch is not None and tag=='dev':
            headline = f"============Evaluation at Epoch= {epoch} on {tag} set ============"
            print(headline)
            f.write(headline)
        else:
            headline = f"============Evaluation on {tag} set ============"
            print(headline)
            f.write(headline)

        report = classification_report(keys,preds, target_names = list(LABEL_TO_ID.keys()), digits=4, zero_division=0)
        f1_macro = f1_score(keys,preds, average = 'macro')
        print(report)
        f.write(report)
        output = {
                tag + "_f1": f1_macro
            }
        return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                         default='./Bangla-NER-Splitted-Dataset.json',
                         metavar='PATH',
                         type=str,
                         help="The input data dir")
    parser.add_argument("--output_dir",
                         default='./noisy_output_bert_crf_v2',
                         metavar='PATH',
                         type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--patience',
                         type=int,
                         default=3,
                         help="Number of consecutive decreasing or unchanged validation runs after which model is stopped running")
    
    parser.add_argument("--is_banner",
                         action='store_true',
                         help="Whether to use banner model or vanilla bert model like in noisy label paper.")
    
    parser.add_argument("--weighted",
                         action='store_true',
                         help="Whether to use weighted cross entropy or not.")

    parser.add_argument("--top_rnns",
                         action='store_false',
                         help="Whether to add a BI-LSTM layer on top of BERT in Banner Model.")    
    
    parser.add_argument("--fine_tuning",
                         action='store_false',
                         help="Whether to finetune the BERT weights in Banner Model.")
    
    parser.add_argument("--model_name_or_path", default="csebuetnlp/banglabert", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=50.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=9)

    parser.add_argument("--project_name", type=str, default="Noisy-Label-NER")
    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = NLLModel(args)
    data_file = args.data_dir
    # data_file = os.path.join(args.data_dir, "Bangla-NER-Splitted-Dataset.json")
    # train_file = os.path.join(args.data_dir, "conll_train.txt")
    # dev_file = os.path.join(args.data_dir, "conll_dev.txt")
    # test_file = os.path.join(args.data_dir, "conll_test.txt")
    # testre_file = os.path.join(args.data_dir, "conllpp_test.txt")
    if args.is_banner:
        train_features = read_banner(data_file, tokenizer, is_banner = args.is_banner,datatype='train',max_seq_length=args.max_seq_length)
        dev_features = read_banner(data_file, tokenizer, is_banner = args.is_banner,datatype='validation',max_seq_length=args.max_seq_length)
        test_features = read_banner(data_file, tokenizer, is_banner = args.is_banner,datatype='test',max_seq_length=args.max_seq_length)
    else:
        train_features = read_banner(data_file, tokenizer, datatype='train',max_seq_length=args.max_seq_length)
        dev_features = read_banner(data_file, tokenizer, datatype='validation',max_seq_length=args.max_seq_length)
        test_features = read_banner(data_file, tokenizer, datatype='test',max_seq_length=args.max_seq_length)
    
    # train_features = read_conll(train_file, tokenizer, max_seq_length=args.max_seq_length)
    # dev_features = read_conll(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    # test_features = read_conll(test_file, tokenizer, max_seq_length=args.max_seq_length)
    # testre_features = read_conll(testre_file, tokenizer, max_seq_length=args.max_seq_length)

    # benchmarks = (
    #     ("dev", dev_features),
    #     ("test", test_features),
    #     ("test_rev", testre_features)
    # )
    benchmarks = (
        ("dev", dev_features),
        ("test", test_features)
    )
    wandb.init(project=args.project_name,entity='mahtab-team',config=vars(args))
    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
