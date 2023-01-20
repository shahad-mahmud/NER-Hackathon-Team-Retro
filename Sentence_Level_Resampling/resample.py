# -*- coding: utf-8 -*-


from collections import Counter
import re
from math import log,sqrt, ceil
import emoji
import numpy as np


class Resampling():
    
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels
        
    def get_stats(self):
        
        # Get stats of the class distribution of the dataset
        labels = [label for item in self.labels for label in item]
        num_tokens = len(labels)
        ent = [label[2:] for label in labels if label != 'O']
        count_ent = Counter(ent)
        for key in count_ent:
            #Use frequency instead of count
            count_ent[key] = count_ent[key]/num_tokens
        return count_ent
    
    def resamp(self, method):
        
        # Select method by setting hyperparameters listed below:
        # sc: the smoothed resampling incorporating count
        # sCR: the smoothed resampling incorporating Count & Rareness
        # sCRD: the smoothed resampling incorporating Count, Rareness, and Density
        # nsCRD: the normalized and smoothed  resampling  incorporating Count, Rareness, and Density
        
        if method not in ['sc','sCR','sCRD','nsCRD']:
            raise ValueError("Unidentified Resampling Method")
        
        sampled_sents, sampled_labels = [],[]
        x,y =  self.sents, self.labels
        stats = self.get_stats()
        
        for sen in range(len(x)):
            
            # Resampling time can at least be 1, which means sentence without 
            # entity will be reserved in the dataset  
            rsp_time = 1
            sen_len = len(y[sen])
            ents = Counter([label[2:] for label in y[sen] if label != 'O'])
                 # Pass if there's no entity in a sentence
            if ents:
                for ent in ents.keys():
                    # Resampling method selection and resampling time calculation, 
                    # see section 'Resampling Functions' in our paper for details.
                    if method == 'sc':
                        rsp_time += ents[ent]
                    if method == 'sCR' or method == 'sCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += ents[ent]*weight
                    if method == 'nsCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += sqrt(ents[ent])*weight
                if method == 'sCR':
                    rsp_time = sqrt(rsp_time)
                if method == 'sCRD' or method == 'nsCRD':
                    rsp_time = rsp_time/sqrt(sen_len)
                # Ceiling to ensure the integrity of resamling time
                rsp_time = ceil(rsp_time) 
            for t in range(rsp_time):
                sampled_sents.append(x[sen])
                sampled_labels.append(y[sen])
        return sampled_sents, sampled_labels
                            
    def BUS(self):
        
        # Implementation of Balanced UnderSampling (BUS) mentioned in paper 
        # Balanced undersampling: a novel sentence-based undersampling method 
        # to improve recognition of named entities in chemical and biomedical text
        # Appl Intell (2018) Akkasi et al .
        
        # R parameter is set to 3, as what metioned in this paper.
        
        # Thank Jing Hou for pointing out a previous bug in this part
        
        
        sampled_sents, sampled_labels = [],[]
        x,y = self.sents,self.labels
        for sen in range(len(x)):
            num_sampled = len([label for label in y[sen] if label != 'O'])
            if num_sampled == 0:
                continue
            thres = 3*num_sampled
            mask = [1 if label != 'O' else 0 for label in y[sen] ]
            while num_sampled < thres and num_sampled < len(y[sen]):
                index=np.where(np.array(mask) == 1)[0]
                for i in index:
                    if i != len(mask)-1:
                        if mask[i+1] == 0:
                            mask[i+1] = 1
                            num_sampled += 1
                for i in index:
                    if i != 0:
                        if mask[i-1] == 0:
                            mask[i-1] = 1
                            num_sampled += 1
            sent, label = [],[]
            for i in range(len(y[sen])):        
                if mask[i] == 1:
                    sent.append(x[sen][i])
                    label.append(y[sen][i])
            sampled_sents.append(sent)
            sampled_labels.append(label)
        return sampled_sents,sampled_labels
            
