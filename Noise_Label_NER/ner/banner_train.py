CUDA_LAUNCH_BLOCKING="1"
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import NLLModel
from utils import set_seed, collate_func
from prepro import read_banner, LABEL_TO_ID, ID_TO_LABEL
from sklearn.metrics import classification_report,f1_score
import wandb
from collections import Counter
from cost import crit_weights_gen
import shutil

def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    print("***** Running training *****")
    print("  Num examples = ", len(train_features))
    print("  Batch size = ", args.batch_size)
    print("  Num steps = ", total_steps)
    
    best_val_f1,prev_val_f1 = [0]*args.n_model,[-1]*args.n_model
    test_f1_score = 0.0
    patience = [0]*args.n_model
    

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        if epoch>10:
            num_models = len(model.models)
            parameters = []
            for i in range(num_models):
                parameters.extend([{"params": model.models[i].classifier.parameters(), "lr": 0.0005},
                                        {"params": model.models[i].model.parameters(), "lr": 5e-5},
                                        {"params": model.models[i].rnn.parameters(), "lr": 0.0005},
                                        {"params": model.models[i].crf.parameters(), "lr": 0.0005}])
            optimizer = Adam(parameters,)
        loss_list = []
        val_f1 = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
                
            outputs = model(input_ids = batch['input_ids'],
                                attention_mask = batch['attention_mask'],
                                labels = batch['labels'],epoch = epoch)
            loss = outputs[0] / args.gradient_accumulation_steps
            loss_list.append(loss.item())
            loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
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
                        for i in range(args.n_model):
                            val_f1.append(results[f"{tag}_f1_{args.model_name_or_path[i]}_{i}"])     
                    wandb.log(results, step=num_steps)
        
        for i in range(args.n_model):
            model_save_path = os.path.join(args.output_dir, f"{args.model_name_or_path[i].split('/')[-1]}_{i}_model.pt")
            if val_f1[i] > best_val_f1[i]:
                best_val_f1[i] = val_f1[i]
                print("\nFound better f1=%.4f on validation set. Saving model\n" %(val_f1[i]))
                # print("\n%s\n" %(report))

                torch.save(model.state_dict(), open(model_save_path, 'wb'))
                patience[i]=0
                prev_val_f1[i] = val_f1[i]

            else :
                if val_f1[i] < prev_val_f1[i]:
                    print("\nF1 score worse than the previous epoch: {}\n".format(val_f1[i]))
                    patience[i]+=1
                    prev_val_f1[i] = val_f1[i]
                else:
                    patience[i] = 0
                    prev_val_f1[i] = val_f1[i]
                    
        if np.all(patience >= [args.patience]*args.n_model):
            for i in range(args.n_model):
                print(f"No more patience. Existing best model score for {args.model_name_or_path[i]}: {best_val_f1[i]}")
                break

        loss_epoch = np.mean(loss_list)
        wandb.log({'train_loss_epoch':loss_epoch})
        
    print(f'Best F1 Score on validation set: {np.max(best_val_f1)}')
    best_model_index = np.argmax(best_val_f1)
    best_model_save_path = os.path.join(args.output_dir, f"{args.model_name_or_path[best_model_index].split('/')[-1]}_{best_model_index}_model.pt")             
    state_dict = torch.load(open(best_model_save_path, 'rb'))
    model.load_state_dict(state_dict)
    
    print("Loaded saved model")
    model.to(args.device)
    for tag, features in benchmarks:
        if tag=='test':
            output = evaluate(args, model, features,tag=tag, stage = 'final')
            test_f1_score = np.max(list(output.values()))
        else:
            evaluate(args, model, features,tag=tag, stage = 'final')
    wandb.run.summary["best_test_f1_macro"] = test_f1_score
    wandb.run.summary["best_model"] = args.model_name_or_path[best_model_index] +"_"+str(best_model_index)
    
    print(" Save best model weights in wandb")
    
    model_artifact = wandb.Artifact(
      f"model_{test_f1_score}", description=f"model-validation f1 macro:{best_val_f1[best_model_index]} test f1 macro {test_f1_score}",
      metadata=dict(wandb.config), type = 'BERT-CRF')
    model_artifact.add_file(best_model_save_path)                  
    wandb.log_artifact(model_artifact)
    wandb.finish()             
    shutil.copyfile(best_model_save_path,os.path.join(args.model_dir,'model.pt'))

def evaluate(args, model, features, epoch=None, stage = None, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, drop_last=False)
    preds, keys = [[]]*args.n_model, [[]]*args.n_model
    
    for batch in dataloader:
        model.eval()
        for i in range(args.n_model):
            keys[i] = keys[i] + batch['labels'][i].cpu().numpy().flatten().tolist()  
        batch['labels'] = None
        with torch.no_grad():
            logits = model(input_ids = batch['input_ids'],
                                attention_mask = batch['attention_mask'])
            # logits = model(**batch)[0]
            for i in range(args.n_model):
                if args.is_banner:
                    preds[i] = preds[i] + np.argmax(logits[i][:,:-1].cpu().numpy(), axis=-1).tolist()
                else:
                    preds[i] = preds[i] + np.argmax(logits[i].cpu().numpy(), axis=-1).tolist()
                    
    if args.is_banner:
        for i in range(args.n_model):
            preds[i], keys[i] = list(zip(*[[pred, key] for pred, key in zip(preds[i], keys[i]) if key != len(LABEL_TO_ID)]))
    else:
        for i in range(args.n_model):
            preds[i], keys[i] = list(zip(*[[pred, key] for pred, key in zip(preds[i], keys[i]) if key != -1]))
   
    model.zero_grad()
    
    with open(os.path.join(args.output_dir,f"{tag}_report_{stage if stage else ''}.txt"), 'a') as f:
        if epoch is not None and tag=='dev':
            headline = f"============Evaluation at Epoch= {epoch} on {tag} set ============"
            print(headline)
            f.write(headline)
        else:
            headline = f"============Evaluation on {tag} set ============"
            print(headline)
            f.write(headline)
            

        output={}
        for i in range(args.n_model):
            model_headline=f'============Evaluation by {args.model_name_or_path[i]}============'
            print(model_headline)
            f.write(model_headline)
            report = classification_report(keys[i],preds[i], target_names = list(LABEL_TO_ID.keys()), labels=list(LABEL_TO_ID.values()),  digits=4, zero_division=0)
            f1_macro = f1_score(keys[i],preds[i], labels=list(LABEL_TO_ID.values()),average = 'macro')
            print(report)
            f.write(report)
            output[tag + "_f1"+"_"+args.model_name_or_path[i]+"_"+str(i)]=f1_macro
        return output


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir",
                         default='/root/nlp_hackathon_bd_2023/data/processed',
                         metavar='PATH',
                         type=str,
                         help="The input data dir")
    parser.add_argument("--output_dir",
                         default='./noisy_train_v1',
                         metavar='PATH',
                         type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_dir",
                         default='./noisy_model',
                         metavar='PATH',
                         type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument('--patience',
                         type=int,
                         default=3,
                         help="Number of consecutive decreasing or unchanged validation runs after which model is stopped running")
    
    parser.add_argument("--is_banner",
                         default=True,
                         type = bool,
                         help="Whether to use banner model or vanilla bert model like in noisy label paper.")
    
    parser.add_argument("--weighted",
                         default=True,
                         type = bool,
                         help="Whether to use weighted cross entropy or not.")

    parser.add_argument("--top_rnns",
                         default=True,
                         type = bool,
                         help="Whether to add a BI-LSTM layer on top of BERT in Banner Model.")
    
    parser.add_argument("--fine_tuning",
                         default=True,
                         type = bool,
                         help="Whether to finetune the BERT weights in Banner Model.")
    
    parser.add_argument("--model_name_or_path", type=str, nargs="+", default=["csebuetnlp/banglabert","bert-base-multilingual-cased"])
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=len(LABEL_TO_ID))

    parser.add_argument("--project_name", type=str, default="ner-hackathon-noisy")
    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # wandb.init(project=args.project_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    
    data_file = args.data_dir
   
    if args.is_banner:
        train_features = read_banner(data_file, args, is_banner = args.is_banner,datatype='train',max_seq_length=args.max_seq_length)
        
        dev_features = read_banner(data_file, args, is_banner = args.is_banner,datatype='val',max_seq_length=args.max_seq_length)
        
        test_features = read_banner(data_file, args, is_banner = args.is_banner,datatype='test',max_seq_length=args.max_seq_length)
        
    else:
        train_features = read_banner(data_file, args, datatype='train',max_seq_length=args.max_seq_length)
        dev_features = read_banner(data_file, args, datatype='validation',max_seq_length=args.max_seq_length)
        test_features = read_banner(data_file, args, datatype='test',max_seq_length=args.max_seq_length)
    
    wandb.init(project="banner", entity='shahad001',config=vars(args))
    d = Counter([label for example in train_features for label in example['labels'][0] ])
    data_dist = [d[key] for key in ID_TO_LABEL.keys()]
    crit_weights = crit_weights_gen(0.5, 0.9, data_dist)
    crit_weights.insert(len(crit_weights),0)
    # print(crit_weights)
    crit_weights = torch.tensor(crit_weights).to(device)
    model = NLLModel(args,crit_weights)

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features)
    )
    
    train(args, model, train_features, benchmarks)

if __name__ == "__main__":
    main()
