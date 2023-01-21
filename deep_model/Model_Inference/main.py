CUDA_LAUNCH_BLOCKING="1"
import logging
from utils import get_attr_file
import torch
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import pickle
from analysis.analysis import *
import numpy as np
import pandas as pd
import collections,json
from utils.functions import import_vocab

from utils.functions import get_vocab
VOCAB,LABEL_TO_ID,ID_TO_LABEL =  get_vocab()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_file):
    
    with open(config_file,'r') as f:
            config = yaml.safe_load(f)

    dataset_config = config['dataset_config']
    test_file =dataset_config['test_file']
    tokenizer = dataset_config['tokenizer']
    tagset = config['tagset']
    model_type = config['model']
    file_type = config['file_type']
    import_vocab(tagset,model_type)
    args = [test_file,tokenizer,file_type]
    print('dataset built started')
    for func_name,func_args in dataset_config['dataset_funcs'].items():
        func_name = get_attr_file(func_name)
        args = func_name(*args,**func_args)
            
    dataset = args

    
    print('dataset build finished')
    if dataset_config['collate_func']!='None':
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=get_attr_file(dataset_config['collate_func']))
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    print('model build started')
    model_config = config['model_config']
    weight_path = model_config['weight_path']
    args = []
    if 'model_funcs' in model_config:
        for func_name,func_args in model_config['model_funcs'].items():
            func_name = get_attr_file(func_name)
            args = func_name(*args,**func_args)        
        model = args
    print('model build finished')
    print('loading model weights')
    if 'load_model_func' in model_config:
        if args:
            args = [args]+[weight_path,config['device']]
        else:
            args = [weight_path,config['device']]
        
        for func_name,func_args in model_config['load_model_func'].items():
            func_name = get_attr_file(func_name)
            args = func_name(*args,**func_args)
        model = args
    print('model weights loaded')
    
    print('prediction started')
    inference_config = config['inference_config']
    output_dir =os.path.join(inference_config['output_dir'],os.path.basename(test_file).split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    show_performance = inference_config['show_performance']
    device = config['device']
    args = [model,dataloader,output_dir,show_performance,device]
    for func_name,func_args in inference_config['inference_funcs'].items():
        func_name = get_attr_file(func_name)
        args = func_name(*args,**func_args)
    print('prediction finished')
    
    #analysis
    show_analysis = config['show_analysis']
        
    if show_analysis:
        print('analysis started')
        analysis_config = config['analysis_config']
        show_prob_dist = analysis_config['show_prob_distrib']
        show_intersection_mistakes = analysis_config['show_intersection_mistakes']
        show_train_majority_mistakes = analysis_config['show_train_majority_mistakes']


        if show_prob_dist:
            
            if args:
                preds,pred_probs = args
                with open(os.path.join(output_dir,'pre_probs.pkl'),'wb') as f:
                    pickle.dump(pred_probs,f)    
            else:
                preds = []
                with open(os.path.join(output_dir,'prediction.txt'),'r',encoding='utf8') as f:
                    for line in f:
                        sen,tags = line.split('\t')
                        tags = tags.split()
                        preds.append(tags)
                with open(os.path.join(output_dir,'pre_probs.pkl'),'rb') as f:
                    pred_probs = pickle.load(f)
        else:
            preds = args

        entits,per_ins,invalid_ents = entities_info(test_file,set='test',type=file_type)
        
        if len(invalid_ents) >0:
            with open(os.path.join(output_dir,'invalid_test.json'),'w',encoding='utf8') as f:
                json.dump(invalid_ents,f,ensure_ascii=False)

        pred_ents,pred_per_ins,invalid_preds = entities_info(os.path.join(output_dir,'prediction.txt'),set='test')

        if len(invalid_preds) >0:
            with open(os.path.join(output_dir,'invalid_pred.json'),'w',encoding='utf8') as f:
                json.dump(invalid_preds,f,ensure_ascii=False)
        
        with open(os.path.join(output_dir,'analysis_summary.txt'),'a+') as f:
            f.write(f'Total Test sentences: {len(per_ins)}')
            f.write(f'\nTotal Invalid Real Entity tags: {len(invalid_ents)}')
            f.write(f'\nTotal number of unqiue entities: {len(entits)}')
            all_tags = [t for p in per_ins for t in per_ins[p]['tags']]
            distrib_tags = dict(collections.Counter(all_tags))
            f.write('\nTags distribution')
            f.write('\n'+str(distrib_tags))
            f.write('\n'+'#'*100)
            f.write(f'\nTotal Invalid Predicted Entity tags: {len(invalid_preds)}')
                
        if show_prob_dist:
            save_prob_dif_hist(pred_ents,output_dir,pred_probs)   
            

        misclassified_info = misclassified_tags(list(entits.keys()),entits,pred_per_ins)
        total_misc = misclssfication_report(misclassified_info,output_dir)
        mis_output_dir = os.path.join(output_dir,'all_misclassifications')
        misclassified_df = save_entity_df(misclassified_info,mis_output_dir,'all_misc.csv')
        
        mis_output_dir_grouped = os.path.join(output_dir,'all_misclassifications','grouped')
        misclassified_df_grouped = group_save_entity_df(misclassified_info,mis_output_dir_grouped)

        if show_prob_dist:
            save_prob_dif_hist(misclassified_info,mis_output_dir,pred_probs)
            high_prob_misses, low_prob_misses = prob_distrib_analysis(misclassified_info,pred_probs,mis_output_dir)
            with open(os.path.join(output_dir,'analysis_summary.txt'),'a+') as f:
                f.write(f'\nAmong {total_misc} misclassifed tags, {high_prob_misses.shape[0]} was high prob(>0.85) misses,\
    and {low_prob_misses.shape[0]} was low prob(<0.85) misses\n')
                f.write('\n'+'#'*100)

        if show_intersection_mistakes:
            train_file = analysis_config['train_file']
            entits,per_ins,invalid_ents = entities_info(train_file,entity_info = entits, per_instance_entity=per_ins,type=file_type)

            inter_ents,complement_ents = intersecting_complement_tags(entits)     
            with open(os.path.join(output_dir,'analysis_summary.txt'),'a+') as f:
                f.write(f'\n\nTotal intersecting tags between train and test set: {len(inter_ents)}')
                f.write(f'\nTotal non intersecting tags between train and test set: {len(complement_ents)}')
                f.write('\n'+'#'*100)
                f.write('\n Misclassifications in Intersecting Tags analysis:')
            misclassified_info_inter = misclassified_tags(inter_ents,entits,pred_per_ins)
            total_misc_inter = misclssfication_report(misclassified_info_inter,output_dir)
            
            mis_inter_output_dir = os.path.join(output_dir,'intersecting_tags_misclass')
            misclassified_df_inter = save_entity_df(misclassified_info_inter,mis_inter_output_dir,'all_misc.csv')

            mis_inter_output_dir_grouped = os.path.join(mis_inter_output_dir,'grouped')
            misclassified_df_inter_groupe = group_save_entity_df(misclassified_info_inter,mis_inter_output_dir_grouped)
            if show_train_majority_mistakes:
                with open(os.path.join(output_dir,'analysis_summary.txt'),'a+') as f:
                    f.write('\n Misclassifications in Intersecting Tags due to choosing train majority tag analysis:')
                misclassified_info_inter_major,misclassified_info_inter = majority_misclassification(entits,misclassified_info_inter,mis_inter_output_dir,'majority_misclass.csv')
                total_misc_inter_major = misclssfication_report(misclassified_info_inter_major,output_dir)
            
        print('analysis finished')
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--config",type=str,required=True,metavar='PATH',help='The model inference config file location')
    args = parser.parse_args()
    main(args.config)