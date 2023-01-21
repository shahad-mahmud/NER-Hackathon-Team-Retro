import collections
from typing import Tuple, List
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pylab as plt
from ast import literal_eval


def bio_decode(char_label_list: List[Tuple[str, str]]) -> List:
    
    idx = 0
    length = len(char_label_list)
    # print(char_label_list)
    invalid_tags = []
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]
        
        # merge chars
        
        if current_label == "O":
            entity = char_label_list[idx][0]
            tags.append((entity, [label[0]], idx, idx))
            idx+=1
        
        elif current_label == "B":
            end = idx + 1
            while end < length and char_label_list[end][1][0] == "I":
                end += 1
            entity = " ".join(char_label_list[i][0] for i in range(idx, end))
            ent_label = [char_label_list[i][1] for i in range(idx, end)]
            tags.append((entity, ent_label, idx, end-1))
            idx = end
        else:
            # print('Invalid tags',char_label_list)
            entity = char_label_list[idx][0]
            tags.append((entity, [label], idx, idx))
            invalid_tags.append((entity, [label], idx, idx))
            idx+=1
            # raise Exception("Invalid Inputs")
    return tags,invalid_tags

def entities_info(file_name,entity_info=None,per_instance_entity=None,set='train',type='txt'):
    if type == 'txt':
        assert file_name.endswith('.txt') or file_name.endswith('.tsv')
    elif type == 'csv':
        assert file_name.endswith('.csv')   
    
    if not entity_info:
        entity_info = collections.defaultdict(dict)
    if not per_instance_entity:
        per_instance_entity = collections.defaultdict(list)
        
    invalid_ent_info = collections.defaultdict(dict)
    
    words_list,labels_list = [],[]
    if type == 'csv':    
        def convert_null(val):
            return val.replace('null','None')
        inference_data = pd.read_csv(file_name,converters={"spans": lambda x:literal_eval(convert_null(x))})
        for i,row in inference_data.iterrows():
            # print(row['spans'])
            # origin_count += 1
            span_info = row['spans']
            words = row['text'].split()
            label_arr = ["O"]*len(words)
            for span in span_info:
                start,end = int(span['token_start']),int(span['token_end'])
                if span['label'] != 'OTHER':
                    label_arr[start:end] = ['B-'+span['label']] +['I-'+span['label'] for i in range(start,end-1)]
            
            words_list.append(words)
            labels_list.append(label_arr)
    
    elif type == 'txt':
    
        with open(file_name) as fin:
            for i,line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                # origin_count += 1
                src, labels = line.split('\t')
                words_list.append(src.split())
                labels_list.append(labels.split())
    
    for i,(sen,tags) in enumerate(zip(words_list,labels_list)):
    
        ent_tags,inv_tags = bio_decode(char_label_list=[(char, label) for char, label in zip(sen, tags)])
        
        for ent_name,label,start,end in ent_tags:
            entity_info[ent_name][f'{set}_{i}']={'span':(start,end),'tag':label,'words':sen,'tags':tags}
        for ent_name,label,start,end in inv_tags:
            invalid_ent_info[ent_name][f'{set}_{i}']={'span':(start,end),'tag':label,'words':sen,'tags':tags}
        
        per_instance_entity[f'{set}_{i}'] = {'words':sen,'tags':tags}
        
    return entity_info,per_instance_entity,invalid_ent_info


def intersecting_complement_tags(entities_info):
    train_entities,test_entities = [],[]
    for ent_name in entities_info:
        for ent_idx in entities_info[ent_name]:
            if 'train' in ent_idx:
                train_entities.append(ent_name)
            elif 'test' in ent_idx:
                test_entities.append(ent_name)
    inter_ents = set(train_entities).intersection(set(test_entities))
    complement_ents = set(test_entities) - inter_ents
    return inter_ents,complement_ents

def misclassified_tags(entities,entities_info,per_instance_pred_entity):
    # inc = collections.defaultdict(list)
    misclassified_info = collections.defaultdict(dict)
    misclassified_distrib = collections.defaultdict(list)
    for ent in entities:
        for ent_index in entities_info[ent]:
            if 'test' in ent_index:
                real_ent_tag = entities_info[ent][ent_index]['tag']
                real_tags = entities_info[ent][ent_index]['tags']
                pred_tags = per_instance_pred_entity[ent_index]['tags']
                tag_span = entities_info[ent][ent_index]['span']
                pred_ent_tag = pred_tags[tag_span[0]:tag_span[1]+1]
                if  real_ent_tag!= pred_ent_tag:
                    for ent_tag,pred_tag in zip(real_ent_tag,pred_ent_tag):
                        if ent_tag != pred_tag:
                            misclassified_distrib[ent_tag].append(pred_tag)
                    misclassified_info[ent][ent_index]={'span':tag_span,'real_tag':real_ent_tag,'pred_tag':pred_ent_tag,'words':entities_info[ent][ent_index]['words'],\
                                                        'real_tags':real_tags,'pred_tags':pred_tags}        

                    
    return misclassified_info

def misclssfication_report(misclassified_info,output_dir):
    misclassified_distrib = collections.defaultdict(list)
    for ent in misclassified_info:
        for ent_index in misclassified_info[ent]:
            for ent_tag,pred_tag in zip(misclassified_info[ent][ent_index]['real_tag'],misclassified_info[ent][ent_index]['pred_tag']):
                if ent_tag != pred_tag:
                    misclassified_distrib[ent_tag].append(pred_tag)
                    
    with open(os.path.join(output_dir,'analysis_summary.txt'),'a+') as f:
        count = sum([len(v) for v in misclassified_distrib.values()])    
        f.write(f'\nTotal Misclassfied entities: {count}')
        f.write(f'\nTotal misclassifications by tags:')
        for k,v in misclassified_distrib.items():
            f.write(f'\n{k}:{len(v)}')
        f.write(f'\n\nMisclassificaiton tags analysis:')
        for k,v in misclassified_distrib.items():
            f.write(f'\n{k} was misclassified to theses different types of tags:')
            f.write('\n'+str(dict(collections.Counter(v)))+'\n')
        f.write('\n'+'#'*100)
    return count

         
def save_entity_df(entities_info,output_dir,file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    entity_info_df = collections.defaultdict(list)
    for ent_name in entities_info:
        for ent_index in entities_info[ent_name]:
            entity_info_df['entity'].append(ent_name)
            entity_info_df['ent_index'].append(ent_index)
            for k in entities_info[ent_name][ent_index]:
                entity_info_df[k].append(entities_info[ent_name][ent_index][k])
                
    
    all_df = pd.DataFrame.from_dict(entity_info_df)
    all_df.to_csv(os.path.join(output_dir,file_name),index=None)
    return all_df
    
def group_save_entity_df(entities_info,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    entity_info_df = collections.defaultdict(list)
    unique_ent_labels = []
    for ent_name in entities_info:
        for ent_index in entities_info[ent_name]:
            real_label = entities_info[ent_name][ent_index]['real_tag']
            if '-' in real_label[0]:
                real_label = real_label[0][2:]
                unique_ent_labels.append(real_label)
            else:
                real_label = real_label[0]
                    
            entity_info_df[real_label+'_entity'].append(ent_name)
            entity_info_df[real_label+'_ent_index'].append(ent_index)
            for k in entities_info[ent_name][ent_index]:    
                entity_info_df[real_label+'_'+k].append(entities_info[ent_name][ent_index][k])
            
    unique_ent_labels = list(set(unique_ent_labels))
    ent_df_group = collections.defaultdict()
    for label in unique_ent_labels:
        entity_info_df_group = collections.defaultdict(list)
        for k in entity_info_df:
            if label in k:
                entity_info_df_group[k] = entity_info_df[k].copy()
        ent_df_group[label] = pd.DataFrame.from_dict(entity_info_df_group)
    
    for k in ent_df_group:
        ent_df_group[k].to_csv(os.path.join(output_dir,f'mis_df_{k}.csv'),index=None)
    
    return ent_df_group        

def label_dist(ent_name,entities_info,set_name='train'):
    labels = []
    for ent_idx in entities_info[ent_name]:
        if set_name in ent_idx:
            ent_tags = entities_info[ent_name][ent_idx]['tag']
            if '-' in ent_tags[0]:
                ent_tags = ent_tags[0][2:]
            else:
                ent_tags = ent_tags[0]
            labels.append(ent_tags)
    if len(labels)>0:
        label_count = dict(collections.Counter(labels))
        return label_count


def majority_misclassification(entites_info,misclassified_info,output_dir,file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inc_major = collections.defaultdict(dict)
    # train_label_dist = collections.defaultdict()
    for ent_name in misclassified_info:
        train_label_dist = label_dist(ent_name,entites_info)
        if train_label_dist:
            major_label = max(train_label_dist.items(),key = lambda x:x[1])[0]
            major_label = ["B-"+major_label] +["I-"+major_label for i in range(len(ent_name.split())-1)]
            for ent_idx_inc in misclassified_info[ent_name]:
                pred_ent_tag = misclassified_info[ent_name][ent_idx_inc]['pred_tag']
                train_label_dist_str = ''
                for k,v in train_label_dist.items():
                    train_label_dist_str += f"{k}:{v}\n"
                if pred_ent_tag == major_label:
                    inc_major[ent_name][ent_idx_inc] = misclassified_info[ent_name][ent_idx_inc].copy()
                    inc_major[ent_name][ent_idx_inc]['train_major'] = major_label
                    inc_major[ent_name][ent_idx_inc]['train_dist'] = train_label_dist_str
                    
                misclassified_info[ent_name][ent_idx_inc]['train_dist'] = train_label_dist_str
                
    save_entity_df(inc_major,output_dir,file_name)
    save_entity_df(misclassified_info,output_dir,'all_misc.csv')
    return inc_major,misclassified_info

def prob_distrib_analysis(entities_info,pred_probs,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prob_inc_high = collections.defaultdict(dict)
    prob_inc_low = collections.defaultdict(dict)
    for ent_name in entities_info:
        for ent_index in entities_info[ent_name]:
            # print(ent_index)
            int_index = int(ent_index.split('_')[1])
            span = entities_info[ent_name][ent_index]['span']
            pred_prob = pred_probs[int_index][int(span[0]):int(span[1])+1]
            entities_info[ent_name][ent_index]['prob'] = pred_prob
            for i,prob_val in enumerate(pred_prob):
                
                if prob_val>0.85:
                    start,end = int(span[0])+i,int(span[0])+i
                    prob_inc_high[ent_name][ent_index] = entities_info[ent_name][ent_index].copy()
                    prob_inc_high[ent_name][ent_index]['span'] = (start,end) 
                    if 'real_tag' in entities_info[ent_name][ent_index]:
                        prob_inc_high[ent_name][ent_index]['real_tag'] = [entities_info[ent_name][ent_index]['real_tag'][i]]
                        prob_inc_high[ent_name][ent_index]['pred_tag'] = [entities_info[ent_name][ent_index]['pred_tag'][i]]
                    else:
                        prob_inc_high[ent_name][ent_index]['tag'] = [entities_info[ent_name][ent_index]['tag'][i]]
                        
                    prob_inc_high[ent_name][ent_index]['prob'] = prob_val
                    prob_inc_high[ent_name][ent_index]['prob_ent'] = ent_name.split()[i]
                    for k in entities_info[ent_name][ent_index]:
                        if k not in ['span','tag','prob']:
                            prob_inc_high[ent_name][ent_index][k] = entities_info[ent_name][ent_index][k].copy()
                    
                else :
                    start,end = int(span[0])+i,int(span[0])+i
                    prob_inc_low[ent_name][ent_index] = entities_info[ent_name][ent_index].copy()
                    prob_inc_low[ent_name][ent_index]['span'] = (start,end) 
                    if 'real_tag' in entities_info[ent_name][ent_index]:
                        prob_inc_low[ent_name][ent_index]['real_tag'] = [entities_info[ent_name][ent_index]['real_tag'][i]]
                        prob_inc_low[ent_name][ent_index]['pred_tag'] = [entities_info[ent_name][ent_index]['pred_tag'][i]]
                    else:
                        prob_inc_low[ent_name][ent_index]['tag'] = [entities_info[ent_name][ent_index]['tag'][i]]
                        
                    prob_inc_low[ent_name][ent_index]['prob'] = prob_val
                    prob_inc_low[ent_name][ent_index]['prob_ent'] = ent_name.split()[i]
                    for k in entities_info[ent_name][ent_index]:
                        if k not in ['span','tag','prob']:
                            prob_inc_low[ent_name][ent_index][k] = entities_info[ent_name][ent_index][k].copy()
    
    prob_inc_low_df = save_entity_df(prob_inc_low,output_dir,'low_prob_misses.csv')
    prob_inc_high_df = save_entity_df(prob_inc_high,output_dir,'high_prob_misses.csv')
    # save_prob_dif_hist(entities_info,output_dir)            
    return prob_inc_high_df,prob_inc_low_df
    
def save_prob_dif_hist(entities_info,output_dir,pred_probs=None): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)           
    hist_inc_new = collections.defaultdict(list)
    for ent_name in entities_info:
        for ent_index in entities_info[ent_name]:
            if 'prob' in entities_info[ent_name][ent_index]:
                tag_name = 'pred_tag' if 'pred_tag' in entities_info[ent_name][ent_index] else 'tag'
                for p_tag,prob in zip(entities_info[ent_name][ent_index][tag_name],entities_info[ent_name][ent_index]['prob']):
                    hist_inc_new['tag'].append(p_tag)
                    hist_inc_new['prob'].append(prob)
            else:
                int_index = int(ent_index.split('_')[1])
                span = entities_info[ent_name][ent_index]['span']
                pred_prob = pred_probs[int_index][int(span[0]):int(span[1])+1]
                tag_name = 'pred_tag' if 'pred_tag' in entities_info[ent_name][ent_index] else 'tag'
                for p_tag,prob in zip(entities_info[ent_name][ent_index][tag_name],pred_prob):
                    hist_inc_new['tag'].append(p_tag)
                    hist_inc_new['prob'].append(prob)
                
    fig,axes = plt.subplots(1, 1,figsize = (10,8))
    axes = sns.histplot(hist_inc_new,x='prob',hue='tag',log_scale = (False,True),bins = 10,ax=axes)
    fig.savefig(os.path.join(output_dir,'prob_dist.png'))