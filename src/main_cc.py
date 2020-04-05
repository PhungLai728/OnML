"""
An implementation of Ontology-based Interpretable Deep Learning
Author: Phung Lai, CIS, NJIT
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import time
import re
import pandas as pd
import gensim
import sklearn.ensemble
import sklearn.metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
st = PorterStemmer()
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import string
import random
from collections import Counter
import itertools
from keras import backend as K
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM, Lambda, ThresholdedReLU
from keras.layers import GlobalMaxPooling1D, AlphaDropout
from keras.layers.merge import Concatenate
from keras.losses import categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.utils.np_utils import to_categorical 
from keras.models import model_from_json
from keras.models import load_model
import itertools
import bisect 
# from nltk.corpus import stopwords 
from numpy import linalg as LA
import lime_text_3
from sklearn.pipeline import make_pipeline
from lime_text_3 import LimeTextExplainer



def prime2word(z_str_original, z_prime):
    z_str = []
    for i in range(len(z_str_original)):
        if z_prime[i] == 1:
            z_str.append(z_str_original[i])
    return z_str

def check_concepts(tweet):
    word_tokens = tweet.split()
    in_onto_words = []
    for i in range(len(word_tokens)):
        if word_tokens[i] in ontology[0]:
            in_onto_words.append(word_tokens[i])
    return in_onto_words

def normal_sampling(z_prime, i, sample_normal):
    rnd_normal = np.random.rand()
    if rnd_normal > sample_normal:
        z_prime[i] = 0 
    return z_prime

def filter_concept(a_c,pos_i,concept_assign_dict):
    pos = w_type(pos_i)
    sent = []
    for i in a_c:
        concept = concept_assign_dict[i]
        if pos in concept:
            sent.append(1)
        else:
            sent.append(0)
    true_concept = [a_c[i] for i in range(len(a_c)) if sent[i] ==1]
    return true_concept

def sublist(list_, sub_list):
    if(all(x in list_ for x in sub_list)): 
        return True
    else:
        return False

def classify_label(vectors, classifier):
    ''' Classify inputs and return labels '''
    return classifier.predict(vectors)

def classify_prob(vectors, classifier, target_class=None):
    ''' Classify inputs and return probability '''
    if target_class == None:
        return classifier.predict_proba(vectors)
    else:
        idx = list(classifier.classes_).index(target_class)
        return classifier.predict_proba(vectors)[:,idx]

def concatenate_list_data(num, list):
    result= 'Important score: ' +  str(num) + ' ---- Explanation: '
    for element in list:
        result += str(element) + ''
    return result

def normalize(x):
    return x / np.sum(np.abs(x))

def combine_tuple(tuples_list,pos,t1,t2,in_onto_concepts,w_tk,concept_assign_dict,index):
    pos_t1 = pos[t1[index]][1]
    pos_t2 = pos[t2[index]][1]
    a1_c = in_onto_concepts[w_tk[t1[index]]]
    a2_c = in_onto_concepts[w_tk[t2[index]]]
    a1_c_filter = a1_c
    a2_c_filter = a2_c
    # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
    # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
    if a1_c_filter == a2_c_filter: # A & B same concept
        t2.append(t1[index])
        combine = sorted(t2)
        if combine not in tuples_list:
            tuples_list.append(combine)
    else:
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    return tuples_list



def change_format(data, label):
    idx = np.arange(0 , 1)
    x = [data[ i] for i in idx]
    y = [label[ i] for i in idx]
    return x, y

def rem_sublist(tuples_list):
    # Remove sub-tuples
    tuples_list_merge = deepcopy(tuples_list)
    for i in range(len(tuples_list)):
        for j in range(len(tuples_list)):
            if i != j:
                if sublist(tuples_list[i], tuples_list[j]) == True: # i is list, j is sublist
                    if tuples_list[j] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[j])
                if sublist(tuples_list[j], tuples_list[i]) == True:# j is list, i is sublist
                    if tuples_list[i] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[i])
    tuples_list = deepcopy(tuples_list_merge)
    return tuples_list

def rem_sublist_2(tuples_list,t_pos_all):
    # Remove sub-tuples
    t_pos_all_ = deepcopy(t_pos_all)
    tuples_list_merge = deepcopy(tuples_list)
    for i in range(len(tuples_list)):
        for j in range(len(tuples_list)):
            if i != j:
                if sublist(tuples_list[i], tuples_list[j]) == True: # i is list, j is sublist
                    if tuples_list[j] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[j])
                        t_pos_all_.remove(t_pos_all[j])
                if sublist(tuples_list[j], tuples_list[i]) == True:# j is list, i is sublist
                    if tuples_list[i] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[i])
                        t_pos_all_.remove(t_pos_all[i])
    tuples_list = deepcopy(tuples_list_merge)
    t_pos_all = deepcopy(t_pos_all_)
    # tt = [t_pos_all_[ti] for i in range(len(tuples_list))]
    # t_pos_all.extend(tt)
    return tuples_list,t_pos_all

def combine_triple_2(tuples_list,t1,t2,intersect,in_onto_concepts,w_tk,concept_assign_dict,pos):
    p_t1_s = [i3 for i3 in range(len(t1)) if t1[i3] in intersect]
    p_t2_s = [i3 for i3 in range(len(t1)) if t2[i3] in intersect]
    if (p_t1_s == p_t2_s) and (p_t1_s == [0,1]): 
        # Case 1: same the begining, e.g. ABC & ABD
        pos_t1 = pos[t1[len(p_t1_s)]][1]
        pos_t2 = pos[t2[len(p_t2_s)]][1]
        a1_c = in_onto_concepts[w_tk[t1[len(p_t1_s)]]]
        a2_c = in_onto_concepts[w_tk[t2[len(p_t2_s)]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter: # C & D same concept
            if t1[len(p_t1_s)] < t2[len(p_t1_s)]: # index of C < index of D in the sentence 
                t1.append(t2[2])
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else:
                t2.append(t1[2])
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == p_t2_s) and (p_t1_s == [1,2]): 
        # Case 2: same the end, e.g. ABC & DBC
        pos_t1 = pos[t1[0]][1] # p_t1_s[0] - 1
        pos_t2 = pos[t2[0]][1]
        a1_c = in_onto_concepts[w_tk[t1[0]]]
        a2_c = in_onto_concepts[w_tk[t2[0]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter: # A & D same concept
            if t1[0] < t2[0]: # index of A < index of D in the sentence 
                t1 = [t1[0]] + t2 
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else:
                t2 = [t2[0]] + t1
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == p_t2_s) and (p_t1_s == [0,2]): 
        # Case 2: same the begining and the end, e.g. ABC & ADC
        pos_t1 = pos[t1[1]][1] 
        pos_t2 = pos[t2[1]][1]
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter: # B & D same concept
            if t1[1] < t2[1]: # index of B < index of D in the sentence (ABDC)
                t1 = [t1[0]] + [t1[1]] + [t2[1]] + [t2[2]]
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else: # index of D < index of B in the sentence (ADBC)
                t2 = [t2[0]] + [t2[1]] + [t1[1]] + [t1[2]]
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t2_s == [0,1]) and (p_t1_s == [1,2]):
        # Case 3: ABC & BCD 
        t1.append(t2[-1])
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    elif (p_t1_s == [0,1]) and (p_t2_s == [1,2]):
        # Case 3: BCD & ABC 
        t2.append(t1[-1])
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    elif (p_t1_s == [1,2]) and (p_t2_s == [0,2]):
        # Case 4.1: ABC & BDC => ABDC 
        pos_t1 = pos[t1[1]][1] #B
        pos_t2 = pos[t2[1]][1] #D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[2]][1] #C
        pos_t20 = pos[t2[1]][1] #D
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) or (a1_c_filter0 == a2_c_filter0) : # B & D or C & D same concept
            t1 = [t1[0]] + t2 # ABDC
            if t1 not in tuples_list:
                tuples_list.append(t1)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0,2]) and (p_t2_s == [1,2]):
        # Case 4.2: BDC & ABC => ABDC 
        pos_t1 = pos[t1[1]][1] #B
        pos_t2 = pos[t2[1]][1] #D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[1]][1] #D
        pos_t20 = pos[t2[2]][1] #C
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[2]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0) : # B & D or C & D same concept
            t2 = [t2[0]] + t1 # ABDC
            if t2 not in tuples_list:
                tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0,1]) and (p_t2_s == [0,2]):
        # Case 4.3: ABC & ADB => ADBC 
        pos_t1 = pos[t1[1]][1] #B
        pos_t2 = pos[t2[1]][1] #D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[0]][1] #A
        pos_t20 = pos[t2[1]][1] #D
        a1_c0 = in_onto_concepts[w_tk[t1[0]]]
        a2_c0 = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0) : # BD or AD same concept
            t2.append(t1[-1]) # ABDC
            if t2 not in tuples_list:
                tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0,2]) and (p_t2_s == [0,1]):
        # Case 4.4: ADB & ABC => ADBC 
        pos_t1 = pos[t1[1]][1] #B
        pos_t2 = pos[t2[1]][1] #D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[1]][1] #D
        pos_t20 = pos[t2[0]][1] #A
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[0]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0) : # BD or AD same concept
            t1.append(t2[-1]) # ABDC
            if t1 not in tuples_list:
                tuples_list.append(t1)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    return tuples_list

def combine_triple_1(tuples_list,t1,t2,intersect):
    # Only consider case: ABC & CDE => ABCDE
    p_t1_s = [i3 for i3 in range(len(t1)) if t1[i3] in intersect]
    p_t2_s = [i3 for i3 in range(len(t1)) if t2[i3] in intersect]
    if (p_t1_s == 2) and (p_t2_s == 0): # ABC & CDE
        t1.append(t2[1:])
        if (t1 not in tuples_list):
            tuples_list.append(t1)
    elif (p_t1_s == 0) and (p_t2_s == 2): # CDE & ABC
        t2.append(t1[1:])
        if (t2 not in tuples_list):
            tuples_list.append(t2)
    else: # Dont merge other cases
        if (t1 not in tuples_list):
            tuples_list.append(t1)
        if (t2 not in tuples_list):
            tuples_list.append(t2)
    return tuples_list

def raw_tweet_change_punc(raw_tweet):
    change_tweet = raw_tweet.replace('!','.') # Change ! to .
    change_tweet = change_tweet.replace('?','.') # Change ? to .
    return change_tweet

def rem_punc(tweet):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_tweet = regex.sub(' ', tweet)
    return clean_tweet

def is_verb(tag):
    if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','verb']:
        return 1
    else: 
        return 0

def is_adj(tag):
    if tag in ['JJ', 'JJR', 'JJS','adj']:
        return 1
    else: 
        return 0

def is_prep(tag):
    if tag in ['IN', 'TO', 'CC']:
        return 1
    else: 
        return 0

def w_type(tag):
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']: # noun
        return 'noun'
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: # verb
        return 'verb'
    elif tag in ['JJ', 'JJR', 'JJS']: # adjective
        return 'adj' 
    else:
        return 0
    
def ontology_based_weights(x_vectors, z_vectors, ontology, vocab_list,
                           kernel_width=25,
                           metric='cosine', cosine_scale=100.0):
    '''
    Compute the weights (kernel function) based on distance between z and x,
    Differs in ontology will increase distance
    Using their vector represenation
    '''
    distances = None
    if metric == 'cosine':
        distances = paired_cosine_distances(x_vectors, z_vectors) * 100.0
    elif metric == 'euclidean':
        distances = paired_euclidean_distances(x_vectors, z_vectors)
    else:
        raise NotImplementedError('Metric not implemented')
    return np.sqrt(np.exp(- (np.square(distances)) / np.square(kernel_width)))

def build_local_vocab_bow_modify(target_tweet):  
    ''' Build vocab out of target_tweet 
    Compare to un-modified version, here we accept repetition 
    Because of local fidelity characteristic, the same word but appears in different position can have different score/treated differently
    since depend on the position they are and their neighbour words '''
    local_vocab_list = []
    local_vocab_dict = {}
    words = target_tweet.strip().split()
    for word in words:
        idx = len(local_vocab_list)# word_id
        local_vocab_dict[word] = idx
        local_vocab_list.append(word)
    return local_vocab_list, local_vocab_dict

occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
def sampling_process(len_tweet,word_tokens,pos,start_id,position,local_fid,ontology,in_onto_concepts,isolated_concept,abstract,sample_onto,sample_normal,min_words_per_tweet,concept_assign_dict):
    tuples = []
    while True: # Run until a qualified sample found (which has more than 1 word)
        z_prime = [1] * len_tweet
        if len_tweet > local_fid:
            segment = [word_tokens[i] for i in range(start_id, start_id+local_fid+1)]
        else:
            segment = deepcopy(word_tokens) 
        n = len(segment)
        # --------***** Inside the segment *****-------- #
        # Find all tuples and sample them in the segment
        rnd = np.random.rand()
        for i in range(0,n-1):
            for j in range(i+1, n):
                a1 = segment[i]
                a2 = segment[j]
                real_i = start_id + i
                real_j = start_id + j 
                a1 = st.stem(a1)
                a2 = st.stem(a2)
                if (a1 in ontology[0]) and (a2 in ontology[0]): # Case 1: Two words appear in the ontology
                    a1_c = in_onto_concepts[a1]
                    a2_c = in_onto_concepts[a2]
                    if (len(a1_c) == 1) and (a1_c == a2_c): # Case 1.1: 2 words in the same main concept are not in a tuple
                        # print('Not in a tuple')
                        z_prime = normal_sampling(z_prime, real_i, sample_normal)
                        z_prime = normal_sampling(z_prime, real_j, sample_normal)
                    else:  # Case 1.2: words belong to different concepts in the ontology. Have to consider different cases
                        a1_c = [x for x in a1_c if x not in isolated_concept] # Avoid the case of isolated concepts, e.g., MedicalCondition 
                        a2_c = [x for x in a2_c if x not in isolated_concept]
                        # Put sent_based_concept here !!!!!!!!!!
                        idx_pos_i = real_i + position
                        idx_pos_j = real_j + position
                        pos_i = pos[idx_pos_i][1]
                        pos_j = pos[idx_pos_j][1]
                        a1_c_filter = a1_c
                        a2_c_filter = a2_c
                        # a1_c_filter = filter_concept(a1_c,pos_i,concept_assign_dict)
                        # a2_c_filter = filter_concept(a2_c,pos_j,concept_assign_dict)
                        if (len(a1_c_filter) == 1) and (a1_c_filter == a2_c_filter): # Case 1.1: 2 words in the same main concept are not in a tuple
                            # print('Not in a tuple')
                            z_prime = normal_sampling(z_prime, real_i, sample_normal)
                            z_prime = normal_sampling(z_prime, real_j, sample_normal)
                        else:
                            edge_a1_c = [abstract[k] for k in a1_c_filter]
                            edge_a2_c = [abstract[k] for k in a2_c_filter]
                            edge_a1_c = set(sum(edge_a1_c, [])) # To concatenate all edges (list) & remove redundance
                            edge_a2_c = set(sum(edge_a2_c, []))
                            if len(list(edge_a1_c & edge_a2_c)) > 0: # Case 1.2.1: A tuple is FOUND! when there is at least 1 sharing edge (a direct connection) between concepts
                                # print('In a tuple')
                                tuples.append([real_i, real_j])
                                if rnd > sample_onto: # inactive/remove words
                                    z_prime[real_i] = 0 
                                    z_prime[real_j] = 0
                            else: # Case 1.2.2: Words appear in the ontology, but belong to undirect connection (no sharing edges between them) 
                                # print('Not in a tuple')
                                z_prime = normal_sampling(z_prime, real_i, sample_normal)
                                z_prime = normal_sampling(z_prime, real_j, sample_normal)
                else: # Case 2: At least 1 word is not in the ontology
                    # print('Word not in ontology')
                    z_prime = normal_sampling(z_prime, real_i, sample_normal)
                    z_prime = normal_sampling(z_prime, real_j, sample_normal)
        # --------***** Outside the segment *****-------- #
        if len_tweet > local_fid:
            range_in = list(range(start_id, start_id+local_fid+1))
            range_out = [x for x in range(len(word_tokens)) if x not in range_in]
            for i in range_out:
                z_prime = normal_sampling(z_prime, i, sample_normal)
        if sum(z_prime) >= min_words_per_tweet:
            z_str = prime2word(word_tokens, z_prime)
            tuples = [list(x) for x in set(tuple(x) for x in tuples)]
            tmp = []
            for p in tuples:
                tmp.append([m+position for m in p])
            tuples = deepcopy(tmp)
            break
    return z_str, z_prime, tuples

def ontology_based_sample_z(target_tweet,pos,ontology, anstract_concepts, classifier,stopword,cause_list,join_list,concept_assign_dict,vocab_matrix,vocab_cc,max_length,
                            local_fid, no_circle,no_repeat,sample_normal, sample_onto,min_words_per_tweet):
    ''' Based on ontology, sample z from x '''
    # Conjunction words replacement. Fake punctuation
    rem_conj_tweet = deepcopy(target_tweet)
    tmp = nltk.word_tokenize(rem_conj_tweet)
    list_occ = []
    for i in range(len(cause_list)):
        occ = list(occurrences(cause_list[i], tmp)) 
        if len(occ) > 0:
            list_occ.append(list(occurrences(cause_list[i], tmp)))
    if len(list_occ) > 0:
        for i in  list_occ:
            tmp[i[0]] = '.'
    rem_conj_tweet = ' '.join(tmp)
        # if cause_list[i] in tmp:
        #     rem_conj_tweet = rem_conj_tweet.replace(cause_list[i],'.') 
    rem_conj_tweet_remPunc = rem_punc(rem_conj_tweet)
    # Stemming tweet which removes conjuction (because, so, as...)
    tmp = nltk.word_tokenize(rem_conj_tweet)
    rem_conj_tweet_stem = [st.stem(word) for word in tmp] # Stemming 
    rem_conj_tweet_stem = ' '.join(rem_conj_tweet_stem) 
    rem_conj_tweet_stem_remPunc = rem_punc(rem_conj_tweet_stem)
    # Stemming
    tmp = nltk.word_tokenize(target_tweet)
    target_tweet_stem = [st.stem(word) for word in tmp] # Stemming 
    target_tweet_stem = ' '.join(target_tweet_stem) 
    # Vectorize tweets (strings) for classification
    target_tweet_stem_remPunc = rem_punc(target_tweet_stem)
    # target_vector = tfidf_vectorizer.transform([rem_conj_tweet_stem_remPunc])
    sen_tk =  nltk.word_tokenize(rem_conj_tweet_remPunc)
    x_tmp = [vocab_cc[k] for k in sen_tk]
    target_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
    target_predict = model.predict(target_vector, verbose=0)
    # Classify processed tweets (vectors)
    target_label = np.argmax(target_predict)

    target_tweet_remPunc = rem_punc(target_tweet)
    local_vocab_list, local_vocab_dict = build_local_vocab_bow_modify(rem_conj_tweet_remPunc)

    # ------- Traversal checking concepts on ontology ------- #
    # Find all ontology-based words in the target tweet & define their main concepts
    word_tokens_stem = target_tweet_stem_remPunc.split() ########### STEMMING WORD TOKENS
    in_onto_concepts_stem = {}
    word_tokens = target_tweet_remPunc.split()
    in_onto_concepts = {}
   
    onto_list = list(ontology[0])
    concept_list = list(ontology[1])
    for i in range(len(word_tokens_stem)):
        if word_tokens_stem[i] in ontology[0]:
            all_inx = list(occurrences(word_tokens_stem[i], onto_list))
            tmp = []
            for j in all_inx:
                tmp.append(concept_list[j])
            concepts = list(set(tmp))
            in_onto_concepts_stem[word_tokens_stem[i]] = concepts # dictionary
            in_onto_concepts[word_tokens[i]] = concepts # dictionary

    # Find all edges/relationship between a concept with others. 
    # The intuition is that 2 concepts share the same edge will be a tuple
    abstract = {}
    u = list(set(anstract_concepts[0]))
    isolated_concept = []
    for i in range(len(u)):
        ui = u[i]
        all_inx = list(occurrences(ui, anstract_concepts[0]))
        tmp = [anstract_concepts[1][all_inx[k]] for k in range(len(all_inx))]
        if tmp == ['iso']:
            isolated_concept.append(i)
        else:
            abstract[i] = tmp

    # ----------- *********** SAMPLING PROCESS BASED ON ONTOLOGY ************* ----------- #
    # Number of sampled tweets based on length of the tweet.
    # So, if it is a short tweet, no_repeat bigger to create more samples. 
    # If it is a long tweet, no_repeat smaller but still enough samples for learning explainer
    or_tk = nltk.word_tokenize(target_tweet)
    rem_tk = nltk.word_tokenize(rem_conj_tweet)
    rem_pos = [i for i in range(len(or_tk)) if or_tk[i] != rem_tk[i]]

    sentences = rem_conj_tweet.split(' . ')
    sentences = [rem_punc(i) for i in sentences if len(i) != 0]
    word_tokens_list = []
    position_list = []
    count = 0
    if len(rem_pos) > 0:
        if rem_pos[0] == 0:
            count = 1
    for s_idx in range(len(sentences)):
        tmp  = sentences[s_idx].split()
        if len(tmp) > 0:
            position_list.append(count)
            word_tokens_list.append(tmp)
            count += len(tmp)
            if count in rem_pos:
                count += 1

    # Remove . in pos tagging
    pos_remPunc = []
    pos_each = []
    pos_tmp = deepcopy(pos)
    if pos_tmp[len(pos_tmp)-1] != '.':
        pos_tmp.append(('.', '.'))
    len_ = []
    for i in range(len(pos_tmp)):
        if pos_tmp[i][0] != '.':
            pos_each.append(pos_tmp[i])
        else:
            if len(pos_each) != 0:
                pos_remPunc.append(pos_each)
                len_.append(len(pos_each))
                pos_each = []
    # Find anchors
    start_anchors_w = ['not','no','illegal','against','without']
    in_anchor_w = []
    for i in range(len(sentences)):
        # print(i)
        s_tk = sentences[i].split()
        if '.' in s_tk:
            s_tk.remove('.')
        add_anchor = []
        for a in start_anchors_w:
            tmp = list(occurrences(a, s_tk))
            sorted_tmp = sorted(tmp)
            sorted_tmp.append(len(s_tk)-1)
            if len(sorted_tmp) > 1: 
                for t_i in range(len(sorted_tmp)):
                    add_anchor = []
                    t = [a]
                    t1 = sorted_tmp[t_i]
                    if t1 != len(s_tk)-1: # not the last word in the sentence
                        t2 = sorted_tmp[t_i+1]
                        if t2 == len(s_tk)-1:
                            t4 = len(s_tk)
                        else:
                            t4 = deepcopy(t2)
                        for k in range(t1+1, t4): # t1+1 => t2
                            t.append(s_tk[k])
                            t3 = deepcopy(t)
                            end_pos = pos_remPunc[i][k][1]
                            if len(t3) < local_fid and is_prep(end_pos) == 0:
                                add_anchor.append(t3)
                        # Calculate the score and chose one anchor 
                        if len(add_anchor) > 1:
                            x_tmp = [vocab_cc[k] for k in sen_tk]
                            ori_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                            ori_prob = np.max(classifier.predict(ori_vector, verbose=0), axis = 1)
                            differ_prob_tmp = []
                            all_idx = list(range(len(sen_tk)))
                            for remove_idx in add_anchor:
                                rest_words = [j for j in sen_tk if j not in remove_idx]
                                x_tmp = [vocab_cc[j] for j in rest_words]
                                rest_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                                prob = np.max(classifier.predict(rest_vector, verbose=0), axis = 1)
                                differ_prob_tmp.append(abs(prob - ori_prob).item())
                            max_ind = np.argmax(differ_prob_tmp)
                            chosen = add_anchor[max_ind]
                            in_anchor_w.append(' '.join(chosen))
                        else:
                            if len(add_anchor) > 0:
                                chosen = deepcopy(add_anchor[0])  
                                in_anchor_w.append(' '.join(chosen))

    # Find sentences containing anchors
    anchor_position = []
    for i in range(len(sentences)):
        s_tk = sentences[i].split()
        if len(s_tk) > 0:
            for a in range(len(in_anchor_w)):
                a_tk = in_anchor_w[a].split()
                # no_a = len(a_tk)
                if len(a_tk) > 1:
                    for j in range(len(s_tk)- len(a_tk)+1):
                        buffer = [s_tk[k] for k in range(j, j+len(a_tk))]
                        buffer_pos = [k for k in range(j, j+len(a_tk))]
                        buffer_pos = [k + position_list[i]  for k in buffer_pos]
                        # print(buffer)
                        if buffer == a_tk:
                            anchor_position.append([in_anchor_w[a], buffer_pos, i])
                            break

    len_tweet = len(word_tokens)

    z_strs_list = []
    z_primes_list = []

    len_ = min(len_tweet,2)
    for iter in range(no_repeat * len_):
        # This creates 1 sampled sentence 
        z_strs_chunk = []
        z_primes_chunk = []
        tuples_chunk= []
        for s_idx in range(len(word_tokens_list)):
            word_tokens = word_tokens_list[s_idx]
            len_chunk = len(word_tokens)
            if len_chunk < 2: # Obviously no tuples found (1) # Short chunk
                z_prime = [1] * len_chunk
                z_prime = normal_sampling(z_prime, 0, sample_normal)
                z_str = prime2word(word_tokens, z_prime)
                tuples = []
                z_strs_chunk.append([z_str])
                z_primes_chunk.append([z_prime])
                tuples_chunk.append([tuples])
            elif len_chunk > local_fid: # (local_fid+1) or more # Long enough chunk
                z_strs_one = []
                z_primes_one = []
                tuples_one = []
                for start_id in range(0,len_chunk-local_fid,3):
                    z_str, z_prime, tuples = sampling_process(len_chunk,word_tokens,pos,start_id,position_list[s_idx],local_fid,ontology,in_onto_concepts_stem,isolated_concept,abstract,sample_onto,sample_normal,min_words_per_tweet,concept_assign_dict)
                    z_strs_one.append(z_str)
                    z_primes_one.append(z_prime)
                    tuples_one.append(tuples)

                z_strs_chunk.append(z_strs_one)
                z_primes_chunk.append(z_primes_one)
                tuples_chunk.append(tuples_one)
            else: # 2 to local_fid (2-3) # Short chunk
                z_str, z_prime, tuples = sampling_process(len_chunk,word_tokens,pos,0,position_list[s_idx],local_fid,ontology,in_onto_concepts_stem,isolated_concept,abstract,sample_onto,sample_normal,min_words_per_tweet,concept_assign_dict)
                z_strs_chunk.append([z_str])
                z_primes_chunk.append([z_prime])
                tuples_chunk.append([tuples])

        # Combination list
        tmp = list(itertools.product(*z_strs_chunk))
        for str_list in tmp: 
            tmp2 = []
            for i in str_list:
                tmp2.extend(i)
            z_strs_list.append(tmp2)
        tmp = list(itertools.product(*z_primes_chunk))
        for str_list in tmp: 
            tmp2 = []
            for i in str_list:
                tmp2.extend(i)
            z_primes_list.append(tmp2)
        tuples_list = []
        for str_list in tuples_chunk: 
            tmp = []
            for i in str_list:
                if (len(i) != 0) and (i not in tmp):
                    tmp.append(i)
            tuples_list.append(tmp)
        
    return z_strs_list, z_primes_list,in_onto_concepts_stem,in_onto_concepts,abstract,isolated_concept,tuples_list, target_label,target_vector, local_vocab_list, local_vocab_dict,target_tweet_remPunc, target_tweet_stem_remPunc,rem_conj_tweet,rem_conj_tweet_remPunc, rem_pos, anchor_position, position_list# np.asarray(z_tweets)
        # # Add original target tweet at the end
        # z_strs_list.append(word_tokens) 
        # z_primes_list.append([1] * len_tweet) 
def norm_min_max(a):
    norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    return norm

def write_txt(all_explanation,sorted_w_and_i,sorted_norm_w_and_i,target_idx,target_tweet,target_label,actual_label,result_path,rules_OLLIE, rules,ollie_anchor_allPos_,anchor_only,onto_only,ensemble_ollie_osil_anc,w_LIME ):

    with open(result_path, 'a') as outf:
        outf.write('Tweet index: ' + str(target_idx) + '\n\n')
        outf.write('Processed tweet: ' + str(target_tweet) + '\n\n')
        # outf.write('Raw tweet: ' + str(target_raw_tweet) + '\n \n')
        outf.write('Prediction: ' + str(target_label) + ', Actual: ' + str(actual_label) + '\n\n')
        # print(sorted_w_and_i)
        # outf.write('Raw weight: ' + str(sorted_w_and_i) + '\n\n')
        # '''print(sorted_softmax_w_and_i)
        # outf.write(str(sorted_softmax_w_and_i) + '\n')'''


        # print(sorted_norm_w_and_i)
        # outf.write('Normalized: ' + str(sorted_norm_w_and_i) + '\n \n')

        if len(rules_OLLIE) != 0:
            outf.write('OLLIE rules: \n')
            for explain in rules_OLLIE:
                outf.write('   ' + str(explain) + '\n \n')

        if len(onto_only ) != 0:
            outf.write('Ontology-based: \n\n')
            for explain in onto_only:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ontology-based: No explanation found! \n\n')

        if len(anchor_only ) != 0:
            outf.write('Anchors: \n\n')
            for explain in anchor_only:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Anchors: No explanation found! \n\n')
        
        # if rules_OLLIE[0][1] != 'No extractions found.':
        #     outf.write('OLLIE rules: \n\n')
        #     for explain in rules_OLLIE:
        #         outf.write('   ' + str(explain) + '\n\n')
        # else:
        #     outf.write('OLLIE rules: No explanation found! \n\n')

        if len(w_LIME ) != 0:
            outf.write('LIME: \n\n')
            for explain in w_LIME:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('LIME: No explanation found! \n\n')

        if len(rules) != 0:
            outf.write('Ontology-based - Anchors: \n\n')
            for explain in rules:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            rules_ = []
            prob = [rules[k][1] for k in range(len(rules))]
            # prob = norm_min_max(prob)
            prob_onto = normalize(prob)
            for k in range(len(rules)):
                rules_.append([rules[k][0], prob_onto[k]])
            for explain in rules_:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ontology-based - Anchors: No explanation found! \n\n')
            prob_onto = []

        

        # if len(rules_OSIL_added1  ) != 0:
        #     outf.write('Ontology-based added OLLIE (Method 1): \n\n')
        #     for explain in rules_OSIL_added1:
        #         outf.write('   ' + str(explain) + '\n\n')

        # if len(ollie_anchor_ ) != 0:
        #     outf.write('OLLIE - Anchors: \n\n')
        #     for explain in ollie_anchor_:
        #         outf.write('   ' + str(explain) + '\n\n')

        if len(ollie_anchor_allPos_ ) != 0:
            outf.write('OLLIE - Anchors: \n\n')
            for explain in ollie_anchor_allPos_:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            ollie_anchor_allPos = []
            prob = [ollie_anchor_allPos_[k][1] for k in range(len(ollie_anchor_allPos_))]
            # prob = norm_min_max(prob)
            prob_ollie = normalize(prob)
            for k in range(len(ollie_anchor_allPos_)):
                ollie_anchor_allPos.append([ollie_anchor_allPos_[k][0], prob_ollie[k]])
            for explain in ollie_anchor_allPos:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('OLLIE - Anchors: No explanation found! \n\n')
            prob_ollie = []

        if len(ensemble_ollie_osil_anc) != 0:
            outf.write('Ensemble results: \n\n')
            for explain in ensemble_ollie_osil_anc:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            ensemble_ollie_osil_anc_ = []
            prob = [ensemble_ollie_osil_anc[k][1] for k in range(len(ensemble_ollie_osil_anc))]
            # prob = norm_min_max(prob)
            prob_ensem = normalize(prob)
            for k in range(len(ensemble_ollie_osil_anc)):
                ensemble_ollie_osil_anc_.append([ensemble_ollie_osil_anc[k][0], prob_ensem[k]])
            for explain in ensemble_ollie_osil_anc_:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ensemble: No explanation found! \n\n')
            prob_ensem=[]

        outf.write('\n ============= ********* =============\n\n\n')
    return rules_, ollie_anchor_allPos, ensemble_ollie_osil_anc_

def score_funct(new_vect, ori_vect):
    score = LA.norm(new_vect - ori_vect)
    return score 

def ontology_reasoning(target_idx, target_tweet_ori, target_tweet, target_tweet_stem,target_label, actual_label,coarse_tweet,rem_conj_tweet,rem_conj_tweet_remPunc,learned_w_dict, local_vocab_dict, result_path,model,ontology,in_onto_concepts_stem,in_onto_concepts,abstract,isolated_concept,tuples_list_all,rem_posistion,concept_assign_dict,pos,issue_text,cc_property,anchor_position,position_list,record,accuracy,rn_vect):
    # occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
    local_weights = []
    words = []
    words_token = rem_conj_tweet_remPunc.split()
    for word in words_token:
        idx = local_vocab_dict[word]
        local_weights.append(learned_w_dict[idx])
        words.append(word)
    w_and_i = [(local_weights[i], words[i]) for i in range(len(words))]
    norm_local_weights = normalize(local_weights)
    norm_w_and_i = [(norm_local_weights[i], words[i]) for i in range(len(words))]

    domain = []
    rel = []
    range_ = []
    for i in range(len(cc_property)):
        domain.append(cc_property[i][0])
        rel.append(cc_property[i][1])
        range_.append(cc_property[i][2])

    # ---------- Ontology-based rule generation ---------- #
    # Explanation for prediction:
    # Real = Pred: explanation
    # Real != Pred: no explanation
    if target_label != actual_label:
        print('Wrong prediction. No explain for now!')
    else: 
        true_truple = [] 
        anchor_idx = [anchor_position[i][2] for i in range(len(anchor_position))] 
        tuples_position = []
        for i in range(len(tuples_list_all)):
            if (tuples_list_all[i] not in true_truple) and (tuples_list_all[i] != []):
                true_truple.append([tuples_list_all[i], i]) 
                if i not in tuples_position:
                    tuples_position.append(i)
        print(true_truple)
        len_ = len(true_truple)
        print(len_)
        tuples_list_all = deepcopy(true_truple)
        coarse_tweet_rem = rem_punc(coarse_tweet)
        coarse_tweet_tk = coarse_tweet_rem.split()
        if len(tuples_list_all) == 0: # No tuples found in the tweet
            print('no tuple is found. no explain')
        else:
            # ----------- Print normal words and tuples with their score ----------- #
            sorted_w_and_i = sorted(w_and_i, key=lambda x:x[0], reverse=True)
            sorted_norm_w_and_i = sorted(norm_w_and_i, key=lambda x:x[0], reverse=True)
            ########### Merge pairs of tuples ###########
            w_tk = nltk.word_tokenize(target_tweet)
            tuples_list = []
            t_pos_all_ = []
            for tuple_idx_1 in tuples_list_all:
                # print(tuple_idx_1)
                tuple_idx = tuple_idx_1[0]
                t_pos = tuple_idx_1[1]
                t_pos_all_.append(t_pos)
                # print(tuple_idx)
                if len(tuple_idx) == 1: # only one tuple
                    t_idx = deepcopy(tuple_idx[0][0])
                    if t_idx not in tuples_list:
                        tuples_list.append([t_idx])
                else: # consider whether or not we can merge tuples 
                    tuples_list_tmp = []
                    flat_list = [item for sub_ in tuple_idx for item in sub_]
                    flat_list = [list(x) for x in set(tuple(x) for x in flat_list)] # remove repetition
                    tuple_idx = deepcopy(flat_list)
                    for i1 in range(len(tuple_idx)-1):
                        for i2 in range(i1+1, len(tuple_idx)):
                            # print('i1: ', i1)
                            # print('i2: ', i2)
                            t1 = deepcopy(tuple_idx[i1])
                            t2 = deepcopy(tuple_idx[i2])
                            intersect = list(set(t1) & set(t2))
                            if len(intersect) > 0:
                                p_t1_s = [i3 for i3 in range(len(t1)) if t1[i3] in intersect]
                                p_t2_s = [i3 for i3 in range(len(t1)) if t2[i3] in intersect]
                                if (p_t1_s == p_t2_s) and (p_t1_s == [0]): # Case 1: AB & AC => A {B and/or C} if possible
                                    tuples_list_tmp = combine_tuple(tuples_list_tmp,pos,t1,t2,in_onto_concepts,w_tk,concept_assign_dict,1)
                                elif (p_t1_s == p_t2_s) and (p_t1_s == [1]): # Case 2: AC & BC => {A and/or B} C if possible
                                    tuples_list_tmp = combine_tuple(tuples_list_tmp,pos,t1,t2,in_onto_concepts,w_tk,concept_assign_dict,0)
                                else: # Case 3: AB & BC => ABC
                                    if p_t1_s == [0]: # t2 is AB, t1 is BC 
                                        t2.append(t1[1])
                                        combine = sorted(t2)
                                        if combine not in tuples_list_tmp:
                                            tuples_list_tmp.append(combine)
                                    else:  # t1 is AB, t2 is BC
                                        t1.append(t2[1])
                                        combine = sorted(t1)
                                        if combine not in tuples_list_tmp:
                                            tuples_list_tmp.append(combine)
                            else: # Nothing to combine tuples
                                if (t1 not in tuples_list): 
                                    tuples_list_tmp.append(t1)
                                if (t2 not in tuples_list):
                                    tuples_list_tmp.append(t2)
                    tuples_list_tmp = [list(x) for x in set(tuple(x) for x in tuples_list_tmp)] # remove repetition               
                    tuples_list_tmp = rem_sublist(tuples_list_tmp)
                    tuples_list.append(tuples_list_tmp)
            
            ########### Combine triples to a long explanation ###########
            tuples_list_all = deepcopy(tuples_list)
            tuples_list = []
            t_pos_all_
            t_pos_all = []
            # Here, we can FLAT LIST  after merging triples from each sentence
            for ti in range(len(tuples_list_all)):
                tuple_idx = deepcopy(tuples_list_all[ti])
                if len(tuple_idx) == 1:
                    tuples_list.append(tuple_idx[0])
                    t_pos_all.append(t_pos_all_[ti])
                else:
                    # tt = [t_pos_all_[ti] for i in range(len(tuple_idx))]
                    # t_pos_all.extend(tt)
                    len_old = len(tuples_list)
                    for i1 in range(len(tuple_idx)-1):
                        for i2 in range(i1+1, len(tuple_idx)):
                            t1 = deepcopy(tuple_idx[i1])
                            t2 = deepcopy(tuple_idx[i2])
                            intersect = list(set(t1) & set(t2))
                            if len(intersect) == 2:
                                tuples_list = combine_triple_2(tuples_list,t1,t2,intersect,in_onto_concepts,w_tk,concept_assign_dict,pos)
                            elif len(intersect) == 1:
                                tuples_list = combine_triple_1(tuples_list,t1,t2,intersect)
                            else: # Nothing to combine tuples
                                if (t1 in tuples_list) and (t2 in tuples_list):
                                    break
                                else: 
                                    if (t1 not in tuples_list):
                                        tuples_list.append(t1)
                                    if (t2 not in tuples_list):
                                        tuples_list.append(t2)
                    len_off = len(tuples_list) - len_old                    
                    tt = [t_pos_all_[ti] for i in range(len_off)]
                    t_pos_all.extend(tt)
            tuples_list, t_pos_all = rem_sublist_2(tuples_list, t_pos_all)

            ########### Add reasoning words to tuple words ###########
            # Remove the rule that have Verb at the end
            tuples_list_tmp = deepcopy(tuples_list)
            tuples_vis_list  = []
            tuples_list = []
            t_pos_all_ = deepcopy(t_pos_all)
            t_pos_all = []
            for t in range(len(tuples_list_tmp)):
                t_id = tuples_list_tmp[t]
                combine_vis = []
                if len(t_id) >2:
                    for i in range(len(t_id)-1):
                        pos_t1 = pos[t_id[i]][1]
                        pos_t2 = pos[t_id[i+1]][1]
                        a1_c = in_onto_concepts[w_tk[t_id[i]]]
                        a2_c = in_onto_concepts[w_tk[t_id[i+1]]]
                        a1_c_filter = a1_c
                        a2_c_filter = a2_c
                        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
                        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
                        if [w_tk[t_id[i]]] != [w_tk[t_id[i+1]]]:
                            if a1_c_filter == a2_c_filter : # A & B same concept
                                vis = [w_tk[t_id[i]], 'and/or']
                            else:
                                vis = [w_tk[t_id[i]]]
                        else:
                            vis = [w_tk[t_id[i]]]
                        combine_vis.extend(vis)
                    if is_verb(pos[t_id[-1]][1]) == 0: # Not a verb
                        if w_tk[ t_id[-1]] not in combine_vis:
                            combine_vis.extend([w_tk[ t_id[-1]]])
                            # tuples_list.append(t_id)
                            # tuples_vis_list.append(combine_vis)
                        else:
                            t_id.remove(t_id[-1])
                        tuples_list.append(t_id)
                        tuples_vis_list.append(combine_vis)
                        t_pos_all.append(t_pos_all_[t])
                        # if t not in t_pos_all_1:
                        #     t_pos_all_1.append(t_pos_all[t])
                else: # 2 words in the tuple 
                    tuples_list.append(t_id)
                    tuples_vis_list.append([w_tk[t_id[0]], w_tk[t_id[1]]])
                    t_pos_all.append(t_pos_all_[t])
                    # if t not in t_pos_all_1:
                    #     t_pos_all_1.append(t_pos_all[t])

            tuples_vis_list_original = deepcopy(tuples_vis_list)
            ########### Merge tuples with conjunction linking words ###########
            if (len(rem_posistion) > 0) and (len(tuples_list) > 1):
                tuples_list_merge_conj = []
                tuples_list_vis_merge_conj = []
                for t_idx in range(len(tuples_list)-1):
                    for r_idx in rem_posistion:
                        if (max(tuples_list[t_idx]) < r_idx) and (min(tuples_list[t_idx+1]) > r_idx):
                            tmp1 = deepcopy(tuples_list[t_idx])
                            tmp1.extend([r_idx])
                            tmp1.extend(tuples_list[t_idx+1])
                            tuples_list_merge_conj.append(tmp1)

                            tmp2 = deepcopy(tuples_vis_list[t_idx])
                            tmp2.extend([w_tk[r_idx]])
                            tmp2.extend(tuples_vis_list[t_idx+1])
                            tuples_list_vis_merge_conj.append(tmp2)
                            
                            tuples_list = deepcopy(tuples_list_merge_conj)
                            tuples_vis_list = deepcopy(tuples_list_vis_merge_conj)

            sen_tk =  nltk.word_tokenize(rem_conj_tweet_remPunc)
            # x_tmp = [vocab_cc[k] for k in sen_tk]
            # ori_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            # ori_prob = np.max(model.predict(ori_vector, verbose=0), axis = 1)

            x_tmp = [vocab_cc[k] for k in coarse_tweet_tk if k in vocab_cc]
            ori_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            # ori_prob = np.max(model.predict(ori_vector, verbose=0), axis = 1)
            ori_prob_vec = model.predict(ori_vector, verbose=0)
            # ori_label = np.argmax(model.predict(ori_vector, verbose=0))

            tuples_list_ = deepcopy(tuples_list)
            tuples_vis_list_ = deepcopy(tuples_vis_list)
            tuples_list = []
            tuples_vis_list = []
            list_tuple = list(set(t_pos_all))
            t_pos_all_ = []
            for p in list_tuple:
                p_tuple = list(occurrences(p, t_pos_all))
                tt = []
                tt_w = []
                for t in p_tuple:
                    tt.extend(tuples_list_[t])
                    tt_w.extend(tuples_vis_list_[t])
                tuples_list.append(tt)
                tuples_vis_list.append(tt_w)
                t_pos_all_.append(p)

            t_pos_all = deepcopy(t_pos_all_)
            # tuples_list_0 = deepcopy(tuples_list)

            osil_pos = deepcopy(t_pos_all)
            list_anchor = list(set(anchor_idx))
            for p_anchor in range(len(list_anchor)):
                anc = list_anchor[p_anchor]
                anc_t = list(occurrences(anc, t_pos_all))
                if anc in t_pos_all:
                    p_tuple = list(occurrences(anc, anchor_idx))
                    for p in p_tuple:
                        tuples_list[anc_t[0]].extend(anchor_position[p][1])
                        tuples_vis_list[anc_t[0]].extend(anchor_position[p][0].split())
                else:
                    p_anc = list(occurrences(anc, anchor_idx))
                    tt = []
                    tt_w = []
                    for pa in p_anc:
                        tt.extend(anchor_position[pa][1])
                        tt_w.extend(anchor_position[pa][0].split())
                    tuples_list.append(tt)
                    tuples_vis_list.append(tt_w)
                    osil_pos.append(anc)



            # list_anchor = list(set(anchor_idx))
            # t_ = deepcopy(t_pos_all)
            # t_.extend(anchor_idx)
            # list_anchor_osil = list(set(t_))
            # tuples_list_tmp = deepcopy(tuples_list)
            # tuples_vis_list_tmp = deepcopy(tuples_vis_list)
            # tuples_list = []
            # tuples_vis_list = []
            # for p_anchor in range(len(list_anchor_osil)):
            #     anc = list_anchor_osil[p_anchor]
            #     anc_t = list(occurrences(anc, t_pos_all))
            #     anc_a = list(occurrences(anc, anchor_idx))
            #     if len(anc_t) > 0 and len(anc_a) == 0: # No anchor for this ontology-based
            #         for a in anc_t:
            #             osil_pos.append(t_pos_all[a])
            #             tuples_list.append(tuples_list_tmp(t_pos_all[a]))
            #             tuples_vis_list.append(tuples_vis_list_tmp(t_pos_all[a]))
            #     elif len(anc_t) == 0 and len(anc_a) > 0: # Anchor without ontology-based
            #         # osil_pos.append(anc)
            #         for a in anc_a:
            #             osil_pos.append(anchor_idx[a])
            #         p_anc = list(occurrences(anc, anchor_idx))
            #         tt = []
            #         tt_w = []
            #         for pa in p_anc:
            #             tt.extend(anchor_position[pa][1])
            #             tt_w.extend(anchor_position[pa][0].split())
            #         tuples_list.append(tt)
            #         tuples_vis_list.append(tt_w)
            #     else: # both anchor + osil
            #         osil_pos.append(anc)
            #         p_tuple = list(occurrences(anc, anchor_idx))
            #         for p in p_tuple:
            #             tuples_list[anc_t[0]].extend(anchor_position[p][1])
            #             tuples_vis_list[anc_t[0]].extend(anchor_position[p][0].split())
            #     # else:
            

            # Start Adding begin-end words
            tuples_list_tmp = deepcopy(tuples_list)
            tuples_vis_list_tmp = deepcopy(tuples_vis_list)
            tuples_list = []
            tuples_vis_list = []
            for anc in range(len(tuples_list_tmp)):
                min_ = min(tuples_list_tmp[anc])
                max_ = max(tuples_list_tmp[anc])
                tt = [k for k in range(min_,max_+1)]
                tuples_list.append(tt)
                tt_w = [sen_tk[k] for k in tt]
                tuples_vis_list.append(tt_w)

            differ_prob_tmp = []
            words_token = target_tweet.split()
            # all_idx = list(range(len(words_token)))
            
            for remove_idx in tuples_list: # already change tp correct calculation for probability
                rest_idx = [i for i in range(len(coarse_tweet_tk)) if i not in remove_idx and coarse_tweet_tk[i] in vocab_cc]
                rest_words = [coarse_tweet_tk[i] for i in range(len(coarse_tweet_tk)) if i in rest_idx and coarse_tweet_tk[i] in vocab_cc]
                x_tmp = [vocab_cc[k] for k in rest_words]
                rest_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                all_prob_vec = model.predict(rest_vector, verbose=0)
                diff_p = score_funct(all_prob_vec, ori_prob_vec)
                differ_prob_tmp.append(diff_p)
                # prob =  all_prob[0][target_label]
                # prob = np.max(model.predict(rest_vector, verbose=0), axis = 1)
                # pred_ = np.argmax(model.predict(rest_vector, verbose=0))
                # diff_p = modify_score(prob, ori_prob)
                
            # Remove duplicate prob difference which refers to same set of words in the tuple
            # Reduce by same probability
            differ_prob_reduce = list(set(differ_prob_tmp))
            tuples_list_tmp = deepcopy(tuples_list)
            tuples_vis_list_tmp = deepcopy(tuples_vis_list)
            differ_prob = []
            tuples_list = []
            tuples_vis_list = []
            for i in differ_prob_reduce:
                idx = differ_prob_tmp.index(i)
                differ_prob.append(differ_prob_tmp[idx])
                tuples_list.append(tuples_list_tmp[idx])
                tuples_vis_list.append(tuples_vis_list_tmp[idx])
            
            sorted_prob_idx = sorted(range(len(differ_prob)), key=lambda i: differ_prob[i], reverse=True)
            differ_prob = [differ_prob[k] for k in sorted_prob_idx]
            tuples_list = [tuples_list[k] for k in sorted_prob_idx]
            tuples_vis_list = [tuples_vis_list[k] for k in sorted_prob_idx]
            osil_pos = [osil_pos[k] for k in sorted_prob_idx]
             # Reduce by same tuples 
            tuples_list_tmp = deepcopy(tuples_list)
            tuples_vis_list_tmp = deepcopy(tuples_vis_list)
            differ_prob_tmp = deepcopy(differ_prob)
            osil_pos_tmp = deepcopy(osil_pos)
            differ_prob = []
            tuples_list = []
            tuples_vis_list = []
            osil_pos = []
            for i in range(len(tuples_vis_list_tmp)):
                ii  =  tuples_vis_list_tmp[i]
                if ii not in tuples_vis_list:
                    differ_prob.append(differ_prob_tmp[i])
                    tuples_list.append(tuples_list_tmp[i])
                    tuples_vis_list.append(tuples_vis_list_tmp[i])
                    osil_pos.append(osil_pos_tmp[i])

            ##### ----------- Explanation rule generation ----------- #####
            all_explanation = []
            rules = []
            for idx in range(len(tuples_vis_list)):
                one_explain = tuples_vis_list[idx]  
                explain_modify = ' '.join(one_explain)
                explain_str = concatenate_list_data(differ_prob[idx], explain_modify)
                all_explanation.append(explain_str)
                rules.append([explain_modify, differ_prob[idx]])

            # Embedded original LIME 
            iss = "Class " + str(target_label) + ' (1)'
            other_iss = "Non-Class " + str(target_label) + ' (0)'
            class_names = [other_iss, iss]
            iss_name = "Class " + str(target_label)  + " : " + issue_text[target_label]
            
            # Limit number of words shown
            sorted_w_and_i_cut = []
            sorted_norm_w_and_i_cut = []
            K_pos = 5
            gt_pos = sum(1 for number in local_weights if number > 0)
            len_pos = min(K_pos, gt_pos)
            for idx in range(len_pos):
                tmp1 = sorted_w_and_i[idx]
                tmp2 = sorted_norm_w_and_i[idx]
                sorted_w_and_i_cut.append([tmp1[1], tmp1[0]])
                sorted_norm_w_and_i_cut.append([tmp2[1], tmp2[0]])
            K_neg = 5
            gt_neg = sum(1 for number in local_weights if number < 0)
            len_neg = min(K_neg, gt_neg)
            for idx in range(len_neg):
                tmp3 = sorted_w_and_i[-idx-1]
                tmp4 = sorted_norm_w_and_i[-idx-1]
                sorted_w_and_i_cut.append([tmp3[1], tmp3[0]])
                sorted_norm_w_and_i_cut.append([tmp4[1], tmp4[0]])
        
            with open('../model/classifier_cc.p', 'rb') as inf:
                model_lime = pickle.load(inf)
            with open('../model/vectorizer_cc.p', 'rb') as inf:
                vectorizer_cc = pickle.load(inf)
            c = make_pipeline(vectorizer_cc, model_lime)
            explainer = LimeTextExplainer(class_names=class_names) 
            len_1 = len(sen_tk) - 1
            exp = explainer.explain_instance(coarse_tweet, c.predict_proba, num_features=len_1)
            exp2 = explainer.explain_instance(coarse_tweet, c.predict_proba, num_features=5)
            prob_max = max(ori_prob_vec[0])
            others_avg = (1-prob_max)/(ori_prob_vec.shape[1] - 1)
            prob_max_norm = prob_max/(prob_max + others_avg)
            exp2.predict_proba = np.asarray([1-prob_max_norm, prob_max_norm])

            limit_rule = 5
            rules_OLLIE, rules, ollie_anchor_allPos_, ensemble_ollie_osil_anc, tuples_list, ollie_anchor_pos_ind, ensemble_ind = gen_OLLIE3(target_idx,coarse_tweet,model,target_label,ori_prob_vec,max_length,vocab_cc, rules,w_tk,anchor_position,anchor_idx,position_list,tuples_list,tuples_vis_list,osil_pos,limit_rule)
            anchor_only = [anchor_position[i][0] for i in range(len(anchor_position))]
            onto_only = [' '.join(m) for m in tuples_vis_list_original]

            #### For LIME
            w_LIME = []
            tmp = exp.local_exp
            for w in range(len(tmp[1])):
                w_idx = tmp[1][w][0]
                w_LIME.append(coarse_tweet_tk[w_idx])
            no_ = min(20, len(tmp[1]))
            w_LIME_20 = [w_LIME[a] for a in range(no_)]
            w_LIME_20 = '; '.join(w_LIME_20)
            w_LIME_20 = [w_LIME_20]
            rules_onto, rule_ollie, rule_ensemble = write_txt(all_explanation,sorted_w_and_i,sorted_norm_w_and_i,target_idx,target_tweet_ori,target_label,actual_label,result_path,rules_OLLIE, rules,ollie_anchor_allPos_,anchor_only,onto_only, ensemble_ollie_osil_anc,w_LIME_20 )
            
            rn = random.randint(1,24)
            rn_vect.append([target_idx, rn])
            # for rn in range(1,25):
            exp2.save_to_file('../result/cc_' +str(target_idx) + '_label' + str(target_label) + '_rn' + str(rn) + '.html', rules_onto, rule_ollie, rule_ensemble, iss_name, rn) 
            no_r = min(3, len(ensemble_ollie_osil_anc), len(rules), len(ollie_anchor_allPos_))
            # coarse_tweet_tk = coarse_tweet.split()
            ############### Check accuracy - score changes ###############
            for r in range(no_r): 
                # r = 0: 1 rule removed
                # r = 1: 2 rules removed
                # r = 2: 3 rules removed
                if r == 0: #### Case 1 rule removed ##
                    sent_rem_tk = rules[r][0].split() 
                    diff_score_OSIL = rules[r][1]   # OSIL + anchor               
                    diff_score_OLLIE = ollie_anchor_allPos_[r][1] # OLLIE + anchor  
                    diff_score_ensemble = ensemble_ollie_osil_anc[r][1] # Ensemble
                    # LIME
                    lime_range = min(len(w_LIME), len(sent_rem_tk))
                    in_idx = [coarse_tweet_tk.index(w_LIME[w]) for w in range(lime_range)]
                    rest_LIME = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w not in in_idx and coarse_tweet_tk[w] in vocab_cc]
                    x_tmp = [vocab_cc[k] for k in rest_LIME]
                    rest_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                    all_prob_vec = model.predict(rest_vector, verbose=0)
                    diff_score_LIME = score_funct(all_prob_vec, ori_prob_vec)

                    # all_prob = model.predict(rest_vector, verbose=0)
                    # prob_LIME = all_prob[0][target_label]
                    # # prob_LIME = np.max(model.predict(rest_vector, verbose=0), axis = 1)
                    # diff_score_LIME = modify_score(prob_LIME, ori_prob)
                    accuracy.append([target_idx,diff_score_OSIL,diff_score_LIME,diff_score_OLLIE,diff_score_ensemble])
                else:
                    # OSIL
                    osil_idx = deepcopy(tuples_list[0])
                    for i in range(1,r+1):
                        # print(i)
                        osil_idx.extend(tuples_list[i])
                    rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in osil_idx]
                    rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
                    x_tmp = [vocab_cc[k] for k in rest_words_]
                    rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                    # all_prob = model.predict(rest_vector_, verbose=0)
                    all_prob_vec = model.predict(rest_vector_, verbose=0)
                    diff_score_OSIL = score_funct(all_prob_vec, ori_prob_vec)

                    # prob_ = all_prob[0][target_label]
                    # prob_ = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
                    # diff_score_OSIL = modify_score(prob_, ori_prob)
                    # OLLIE
                    ollie_idx = deepcopy(ollie_anchor_pos_ind[0])
                    for i in range(1,r+1):
                        ollie_idx.extend(ollie_anchor_pos_ind[i])
                    rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in ollie_idx]
                    rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
                    x_tmp = [vocab_cc[k] for k in rest_words_]
                    rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                    # all_prob = model.predict(rest_vector_, verbose=0)
                    # prob_ = all_prob[0][target_label]
                    # prob_ = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
                    # diff_score_OLLIE = modify_score(prob_, ori_prob)
                    # Ensemble
                    ensem_idx = deepcopy(ensemble_ind[0])
                    for i in range(1,r+1):
                        ensem_idx.extend(ensemble_ind[i])
                    rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in ensem_idx]
                    rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
                    x_tmp = [vocab_cc[k] for k in rest_words_]
                    rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                    all_prob_vec = model.predict(rest_vector_, verbose=0)
                    diff_score_ensemble = score_funct(all_prob_vec, ori_prob_vec)
                    # all_prob = model.predict(rest_vector_, verbose=0)
                    # prob_ = all_prob[0][target_label]
                    # # prob_ = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
                    # diff_score_ensemble = modify_score(prob_, ori_prob)
                    # LIME
                    sent_rem_tk = []
                    for i in range(r+1):
                        sent_rem_tk.extend(rules[i][0].split())
                    sent_rem_tk = list(set(sent_rem_tk))
                    lime_range = min(len(w_LIME), len(sent_rem_tk))
                    in_idx = [coarse_tweet_tk.index(w_LIME[w]) for w in range(lime_range)]
                    # in_idx = [coarse_tweet_tk.index(w_LIME[w]) for w in range(len(sent_rem_tk))]
                    rest_LIME = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w not in in_idx and coarse_tweet_tk[w] in vocab_cc]
                    x_tmp = [vocab_cc[k] for k in rest_LIME]
                    rest_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
                    all_prob_vec = model.predict(rest_vector, verbose=0)
                    diff_score_LIME = score_funct(all_prob_vec, ori_prob_vec)

                    # all_prob = model.predict(rest_vector, verbose=0)
                    # prob_LIME = all_prob[0][target_label]
                    # prob_LIME = np.max(model.predict(rest_vector, verbose=0), axis = 1)
                    # diff_score_LIME = modify_score(prob_LIME, ori_prob)
                    if r == 1:
                        accuracy2.append([target_idx,diff_score_OSIL,diff_score_LIME,diff_score_OLLIE,diff_score_ensemble])
                    else:
                        accuracy3.append([target_idx,diff_score_OSIL,diff_score_LIME,diff_score_OLLIE,diff_score_ensemble])           
    return accuracy, accuracy2, accuracy3, rn_vect

def gen_OLLIE3(target_idx,coarse_tweet,model,target_label,ori_prob_vec,max_length,vocab_cc,rules_OSIL,w_tk,anchor_position,anchor_idx,position_list,tuples_list,tuples_vis_list,osil_pos,limit_rule):
    # Add OLLIE rules
    f = open( "../data/OLLIE_cc_out/out" + str(target_idx), "r" )
    a = []
    for line in f:
        a.append(line)
    # ollie_triples = []
    args = []
    i = 1
    while i < len(a):
        tmp = a[i]
        if (tmp == '\n'):
            i += 2
        elif (tmp == 'No extractions found.\n'): 
            i += 1
            tmp1 = '0.0: (' + tmp + ')'
            if tmp not in args:
                # ollie_triples.append(tmp1)
                args.append(tmp)
        else:
            c = 0
            while c < len(tmp):
                if tmp[c] == ':':
                    break
                else:
                    c += 1
            c += 3
            tmp2 = ''
            while c < len(tmp):
                if tmp[c] == ')':
                    break
                else:
                    tmp2 = tmp2 + tmp[c]
                    c += 1
            # ollie_triples.append(tmp)
            args.append(tmp2)
            i += 1

    ollie_tmp = []
    if ('No extractions found.\n' in args) and (len(args) > 1):
        for i in args:
            if i != 'No extractions found.\n':
                ollie_tmp.append(i)
        args = deepcopy(ollie_tmp)

    coarse_tweet_rem = rem_punc(coarse_tweet)
    coarse_tweet_tk = coarse_tweet_rem.split()
    rules_OLLIE =  []
    rules_OLLIE2 =  []
    if ('No extractions found.\n' in args):
        rules_OLLIE.append('No extractions found.')
        rules_OLLIE2.append('No extractions found.')
    else:
        rules = []
        rules2 = []
        for i in range(len(args)):
            sent = args[i]
            sent_rem = sent.replace(';','')
            # sent_rem_tk = sent_rem.split()
            # rest_words = [w for w in coarse_tweet_tk if w not in sent_rem_tk and w in vocab_cc]
            # # rest_w = ' '.join(rest_w)
            # x_tmp = [vocab_cc[k] for k in rest_words]
            # rest_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            # prob = np.max(model.predict(rest_vector, verbose=0), axis = 1)
            # diff_p = modify_score(prob, ori_prob)
            # differ_prob.append(diff_p)
            rules.append(sent)
            rules2.append(sent_rem)

        # sorted_prob_idx = sorted(range(len(differ_prob)), key=lambda i: differ_prob[i], reverse=True)
        # differ_prob = [differ_prob[k] for k in sorted_prob_idx]
        # rules = [rules[k] for k in sorted_prob_idx]
        # rules2 = [rules2[k] for k in sorted_prob_idx]

        rules_OLLIE = deepcopy(rules)
        rules_OLLIE2 = deepcopy(rules2)
        # for i in range(len(rules)):
        #     rules_OLLIE.append([differ_prob[i], rules[i]])
        #     rules_OLLIE2.append([differ_prob[i], rules2[i]])

    sent = coarse_tweet.split(' . ')   
    sent_ = deepcopy(sent)
    sent = []
    for i in sent_:
        if len(i) != 0:
            sent.append(i)

    # Find position of OLLIE words in its sentence
    if rules_OLLIE2[0] != 'No extractions found.':
        sent_pos_list = []
        for i in range(len(rules_OLLIE2)):
            # print(i)
            r_i = rules_OLLIE2[i]
            zero_list = []
            for k in range(len(sent)):
                one_sent_tk = sent[k].lower().split()
                r_i_tk = r_i.lower().split()
                len_list = []
                for j in range(len(r_i_tk)):
                    occ = list(occurrences(r_i_tk[j], one_sent_tk)) 
                    len_list.append(len(occ))
                zero = len(list(occurrences(0, len_list)))
                zero_list.append(zero)
            sent_pos = np.argmin(zero_list)
            sent_pos_list.append(sent_pos)

        ollie_in_pos = []
        for i in range(len(rules_OLLIE2)):
            # print('i',i)
            r_i = rules_OLLIE2[i]
            sent_pos = sent_pos_list[i]
            # Find position of OLLIE words in its sentence
            one_sent_tk = sent[sent_pos].lower().split()
            r_i_tk = r_i.lower().split()
            ollie_list = []
            len_list = []
            for j in range(len(r_i_tk)):
                occ = list(occurrences(r_i_tk[j], one_sent_tk)) 
                ollie_list.append(occ)
                len_list.append(len(occ))
            len_list_tmp = deepcopy(len_list)
            ollie_list_tmp = deepcopy(ollie_list)
            len_list = []
            ollie_list = []
            r_i_tk_tmp = []
            for z in range(len(len_list_tmp)):
                # print('z',z)
                z0 = len_list_tmp[z]
                if z0 != 0:
                    len_list.append(len_list_tmp[z])
                    ollie_list.append(ollie_list_tmp[z])
                    r_i_tk_tmp.append(r_i_tk[z])
                # else:
                #     r_i_tk.remove(r_i_tk[z])
            r_i_tk = deepcopy(r_i_tk_tmp)
            ollie_position = []
            if 1 in len_list:
                one = list(occurrences(1, len_list)) 
                for w in range(len(r_i_tk)):
                    if w in one:
                        ollie_position.extend(ollie_list[w])
                    elif w < one[0]:
                        minus = [abs(ollie_list[one[0]][0] - ollie_list[w][wi])  for wi in range(len(ollie_list[w]))]
                        # minus = [wi for wi in minus if wi >= 0]
                        nearest = np.argmin(minus)
                        ollie_position.append(ollie_list[w][nearest]) 
                    else: # w > one[0]
                        minus = [abs(ollie_list[w][wi] - ollie_list[one[0]][0])  for wi in range(len(ollie_list[w]))]
                        # minus = [wi for wi in minus if wi >= 0]
                        nearest = np.argmin(minus)
                        ollie_position.append(ollie_list[w][nearest]) 
            else: # all of the words appear more than once in the sentence
                # position = min(len_list)
                print('Need to be addressed!')
            ollie_position = sorted(ollie_position)
            ollie_position_mod = [k + position_list[sent_pos_list[i]] for k in ollie_position]
            ollie_in_pos.append(ollie_position_mod)

        rules_OLLIE_tmp = deepcopy(rules_OLLIE)
        rules_OLLIE = []
        len_ = min(len(rules_OLLIE_tmp), len(rules_OLLIE_tmp))
        for i in range(len_):
            rules_OLLIE.append([rules[i]])
    else:
        ollie_in_pos = []
        sent_pos_list = []

    ############# Generating Rules for: OLLIE + Anchor #############
    # ollie_in_pos_added_0 = deepcopy(ollie_in_pos)
    ollie_in_pos_added = deepcopy(ollie_in_pos)
    ollie_in_pos = []
    ollie_anchor = []
    ollie_anchor_allPos = []
    ollie_anchor_pos_ind = []
    if len(anchor_idx) > 0: 
        for p_anchor in range(len(anchor_idx)):
            # print(p_anchor)
            anc = anchor_idx[p_anchor]
            if anc in sent_pos_list:
                p_tuple = list(occurrences(anc, sent_pos_list))
                p_tuple_m = min(p_tuple)
                ollie_in_pos_added[p_tuple_m].extend(anchor_position[p_anchor][1])
                ollie_in_pos.append(ollie_in_pos_added[p_tuple_m])
                part_pos = ollie_in_pos_added[p_tuple_m]
                min_ = min(part_pos)
                max_ = max(part_pos)
                a0 = [coarse_tweet_tk[a] for a in part_pos]
                a1 = [coarse_tweet_tk[a] for a in range(min_, max_+1)]
                ollie_anchor_pos_ind.append([a for a in range(min_, max_+1)])
                ollie_anchor.append(' '.join(a0))
                ollie_anchor_allPos.append(' '.join(a1))
            else:
                p_anc = list(occurrences(anc, anchor_idx))
                ollie_anchor_pos_ind.append(anchor_position[p_anc[0]][1])
                ollie_in_pos.append(anchor_position[p_anc[0]][1])
                ollie_anchor.append(anchor_position[p_anc[0]][0])
                ollie_anchor_allPos.append(anchor_position[p_anc[0]][0])

        differ_prob_allPos_ = [] 
        for i in range(len(ollie_anchor_pos_ind)):
            r_i_ = ollie_anchor_pos_ind[i]
            rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in r_i_ and coarse_tweet_tk[w] in vocab_cc]
            # sent_rem_tk_ = r_i_.split()
            rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
            x_tmp = [vocab_cc[k] for k in rest_words_]
            rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            all_prob_vec = model.predict(rest_vector_, verbose=0)
            diff_p_ = score_funct(all_prob_vec, ori_prob_vec)
            # all_prob = model.predict(rest_vector_, verbose=0)
            # prob_ = all_prob[0][target_label]

            # prob_ = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
            # diff_p_ = modify_score(prob_, ori_prob)
            differ_prob_allPos_.append(diff_p_)

        sorted_prob_idx = sorted(range(len(differ_prob_allPos_)), key=lambda i: differ_prob_allPos_[i], reverse=True)
        differ_prob_allPos_ = [differ_prob_allPos_[k] for k in sorted_prob_idx]
        ollie_anchor_allPos = [ollie_anchor_allPos[k] for k in sorted_prob_idx]
        ollie_pos = [anchor_idx[k] for k in sorted_prob_idx]
        ollie_anchor_pos_ind = [ollie_anchor_pos_ind[k] for k in sorted_prob_idx]

        # Remove duplicate prob difference which refers to same set of words in the tuple
        # Reduce by same probability
        differ_prob_reduce = list(set(differ_prob_allPos_))
        ollie_anchor_allPos_tmp = deepcopy(ollie_anchor_allPos)
        differ_prob = []
        ollie_anchor_allPos = []
        for i in differ_prob_reduce:
            idx = differ_prob_allPos_.index(i)
            differ_prob.append(differ_prob_allPos_[idx])
            ollie_anchor_allPos.append(ollie_anchor_allPos_tmp[idx])
        
        sorted_prob_idx = sorted(range(len(differ_prob)), key=lambda i: differ_prob[i], reverse=True)
        differ_prob = [differ_prob[k] for k in sorted_prob_idx]
        ollie_anchor_allPos = [ollie_anchor_allPos[k] for k in sorted_prob_idx]
        ollie_pos = [ollie_pos[k] for k in sorted_prob_idx]
        ollie_anchor_pos_ind = [ollie_anchor_pos_ind[k] for k in sorted_prob_idx]
        ollie_anchor_allPos_ = []
        for i in range(len(ollie_anchor_allPos)):
            ollie_anchor_allPos_.append([ollie_anchor_allPos[i], differ_prob[i] ]) # OLLIE + Anchor
    else:
        ollie_anchor_allPos = []
        ollie_anchor_pos_ind = []
        differ_prob_allPos_ = []
        for i in range(len(ollie_in_pos_added)):
            r_i_ = ollie_in_pos_added[i]
            min_ = min(r_i_)
            max_ = max(r_i_)
            a1 = [a for a in range(min_, max_+1)]
            ollie_w = [coarse_tweet_tk[a] for a in range(min_, max_+1)]
            ollie_w = ' '.join(ollie_w)
            rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in a1 and coarse_tweet_tk[w] in vocab_cc]
            rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
            x_tmp = [vocab_cc[k] for k in rest_words_]
            rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            # all_prob = model.predict(rest_vector_, verbose=0)
            # prob_ = all_prob[0][target_label]
            all_prob_vec = model.predict(rest_vector_, verbose=0)
            diff_p_ = score_funct(all_prob_vec, ori_prob_vec)

            # prob_ = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
            # diff_p_ = modify_score(prob_, ori_prob)
            differ_prob_allPos_.append(diff_p_)
            ollie_anchor_allPos.append(ollie_w)
            ollie_anchor_pos_ind.append(a1)

            # differ_prob_allPos_.append(diff_p_)
        sorted_prob_idx = sorted(range(len(differ_prob_allPos_)), key=lambda i: differ_prob_allPos_[i], reverse=True)
        differ_prob_allPos_ = [differ_prob_allPos_[k] for k in sorted_prob_idx]
        ollie_anchor_allPos = [ollie_anchor_allPos[k] for k in sorted_prob_idx]
        ollie_anchor_pos_ind = [ollie_anchor_pos_ind[k] for k in sorted_prob_idx]
        ollie_pos = anchor_idx # = []
        # Remove duplicate prob difference which refers to same set of words in the tuple
        # Reduce by same probability
        differ_prob_reduce = list(set(differ_prob_allPos_))
        ollie_anchor_allPos_tmp = deepcopy(ollie_anchor_allPos)
        ollie_anchor_pos_ind_tmp = deepcopy(ollie_anchor_pos_ind)
        differ_prob = []
        ollie_anchor_allPos = []
        ollie_anchor_pos_ind = []
        for i in differ_prob_reduce:
            idx = differ_prob_allPos_.index(i)
            differ_prob.append(differ_prob_allPos_[idx])
            ollie_anchor_allPos.append(ollie_anchor_allPos_tmp[idx])
            ollie_anchor_pos_ind.append(ollie_anchor_pos_ind_tmp[idx])
        
        sorted_prob_idx = sorted(range(len(differ_prob)), key=lambda i: differ_prob[i], reverse=True)
        differ_prob = [differ_prob[k] for k in sorted_prob_idx]
        ollie_anchor_allPos = [ollie_anchor_allPos[k] for k in sorted_prob_idx]
        ollie_anchor_pos_ind = [ollie_anchor_pos_ind[k] for k in sorted_prob_idx]
        ollie_anchor_allPos_ = []
        for i in range(len(ollie_anchor_allPos)):
            ollie_anchor_allPos_.append([ollie_anchor_allPos[i], differ_prob[i] ]) # OLLIE + Anchor

    ############# Generating Rules for Ensemble model: OSIL + OLLIE + Anchor #############
    # list_anchor = list(set(anchor_idx))
    list_anchor_tmp = deepcopy(ollie_pos)
    list_anchor_tmp.extend(osil_pos)
    list_anchor_tuple = list(set(list_anchor_tmp))
    ensemble_w = []
    ensemble_diff_prob = []
    ensemble_ind = []
    for p in range(len(list_anchor_tuple)):
        candidate = list_anchor_tuple[p]
        candidate_t = list(occurrences(candidate, osil_pos)) # in ontology
        candidate_anc = list(occurrences(candidate, ollie_pos)) # in ollie + anchor
        if len(candidate_t) > 0 and len(candidate_anc) == 0: # ontology only
            for t in candidate_t: 
                ensemble_w.extend([rules_OSIL[t][0]])
                ensemble_diff_prob.extend([rules_OSIL[t][1]])
                ensemble_ind.append(tuples_list[t])
        elif len(candidate_t) == 0 and len(candidate_anc) > 0: # no ontology, ollie + anchor only
            print('cannot happen') # since osil already includes anchor 
        else: #  ensemble all 3
            anc_onto = deepcopy(tuples_list[candidate_t[0]])
            anc_ollie = deepcopy(ollie_anchor_pos_ind[candidate_anc[0]])
            anc_ollie.extend(anc_onto) # all index
            min_ = min(anc_ollie)
            max_ = max(anc_ollie)
            ensemble = [a for a in range(min_, max_+1)]
            ensemble2 = [coarse_tweet_tk[a] for a in range(min_, max_+1)]
            rest_idx = [w for w in range(len(coarse_tweet_tk)) if w not in ensemble and coarse_tweet_tk[w] in vocab_cc]
            rest_words_ = [coarse_tweet_tk[w] for w in range(len(coarse_tweet_tk)) if w in rest_idx and coarse_tweet_tk[w] in vocab_cc]
            x_tmp = [vocab_cc[k] for k in rest_words_]
            rest_vector_ = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
            all_prob_vec = model.predict(rest_vector_, verbose=0)
            diff_p = score_funct(all_prob_vec, ori_prob_vec)

            # all_prob = model.predict(rest_vector_, verbose=0)
            # prob = all_prob[0][target_label]
            # prob = np.max(model.predict(rest_vector_, verbose=0), axis = 1)
            # diff_p = modify_score(prob, ori_prob)
            ensemble_diff_prob.append(diff_p)
            ensemble_w.append(' '.join(ensemble2))
            ensemble_ind.append(ensemble)

    sorted_prob_idx = sorted(range(len(ensemble_diff_prob)), key=lambda i: ensemble_diff_prob[i], reverse=True)
    ensemble_diff_prob = [ensemble_diff_prob[k] for k in sorted_prob_idx]
    ensemble_w = [ensemble_w[k] for k in sorted_prob_idx]

    ensemble_ollie_osil_anc = []
    for i in range(len(ensemble_w)):
        ensemble_ollie_osil_anc.append([ensemble_w[i], ensemble_diff_prob[i] ])
    

    # Remove 0.0 rules
    all_probs_OSIL = [rules_OSIL[i][1] for i in range(len(rules_OSIL))]
    all_probs_OLLIE = [ollie_anchor_allPos_[i][1] for i in range(len(ollie_anchor_allPos_))]
    all_probs_Ensem = [ensemble_ollie_osil_anc[i][1] for i in range(len(ensemble_ollie_osil_anc))]
    occurrences_0 = lambda lst: (i for i,e in enumerate(lst) if e <= 0)
    OSIL_0 = list(occurrences_0(all_probs_OSIL)) 
    OLLIE_0 = list(occurrences_0(all_probs_OLLIE)) 
    Ensem_0 = list(occurrences_0(all_probs_Ensem)) 
    rules_OSIL = [rules_OSIL[i] for i in range(len(rules_OSIL)) if i not in OSIL_0]
    tuples_list = [tuples_list[i] for i in range(len(rules_OSIL)) if i not in OSIL_0]
    ollie_anchor_allPos_ = [ollie_anchor_allPos_[i] for i in range(len(ollie_anchor_allPos_)) if i not in OLLIE_0]
    ollie_anchor_pos_ind = [ollie_anchor_pos_ind[i] for i in range(len(ollie_anchor_allPos_)) if i not in OLLIE_0] 
    ensemble_ollie_osil_anc = [ensemble_ollie_osil_anc[i] for i in range(len(ensemble_ollie_osil_anc)) if i not in Ensem_0]
    ensemble_ind = [ensemble_ind[i] for i in range(len(ensemble_ollie_osil_anc)) if i not in Ensem_0] 

    # Limit the number of rules 
    if len(rules_OSIL) > limit_rule:
        rules_OSIL = [rules_OSIL[i] for i in range(limit_rule) ]
        tuples_list = [tuples_list[i] for i in range(limit_rule) ]
    if len(ollie_anchor_allPos_) > limit_rule:
        ollie_anchor_allPos_ = [ollie_anchor_allPos_[i] for i in range(limit_rule) ]
        ollie_anchor_pos_ind = [ollie_anchor_pos_ind[i] for i in range(limit_rule) ] 
    if len(ensemble_ollie_osil_anc) > limit_rule:
        ensemble_ollie_osil_anc = [ensemble_ollie_osil_anc[i] for i in range(limit_rule) ]
        ensemble_ind = [ensemble_ind[i] for i in range(limit_rule) ] 
    return rules_OLLIE, rules_OSIL, ollie_anchor_allPos_, ensemble_ollie_osil_anc, tuples_list, ollie_anchor_pos_ind, ensemble_ind


def stack_info_pl(sentence):
    if sentence == 'nan':
        result = [0]*300
    else:
        result = []
        word_tokens = tokenizer.tokenize(sentence) # remove punctuation
        for word in word_tokens:
            if word in word_vectors.vocab:
                result.append(w2v_model[word])
            else:
                tmp = np.asarray([0]*300)
                result.append(tmp)
        if len(result) != 0:
            result = np.vstack(result)
            result = np.sum(result, axis=0)
        else:
            result = np.asarray([0]*300)
    return result

# def modify_score(prob, ori_prob):
#     # For the case of drug abuse only
#     p_1 = (prob - ori_prob).item()
#     diff_p = abs(- p_1)
#     return diff_p

def explain_classifier_tf_count(target_idx,target_tweet,coarse_tweet,actual_label,stopword,cause_list,join_list,abstract_concepts,model,ontology,params,result_path,concept_assign_dict,vocab_matrix,vocab_cc,max_length,issue_text,cc_property, record, accuracy, accuracy2, accuracy3,rn_vect):
    '''
    Inputs:
    target_tweet: tweet being explained
    classifier: callable to get label/prob,
        usually is the scikit-learn's Classifier.predict() or predict_proba()
    '''

    ''' Vocal_list, classifier, target_vector, sampling : stem
    Local_list: no-stem'''
    # with open(vectorizer_path,'rb') as inf:
    #     tfidf_vectorizer = pickle.load(inf)
    # vocab_list = tfidf_vectorizer.get_feature_names()
    # vocab_set = set(vocab_list)

    # if not check_target(target_tweet, vocab_set):
    #     print("The target tweet is too short/contains too many unknown words, cannot be explained")
    #     len_ = 0
    # else:

    token = nltk.word_tokenize(coarse_tweet)
    pos_coarse_tweet = nltk.pos_tag(token)
    pos_target_tweet = pos_coarse_tweet
    # pos_target_tweet = []
    # for i in range(len(pos_coarse_tweet)):
    #     tmp = pos_coarse_tweet[i][0] 
    #     if (tmp not in stopword) & (tmp != '.'):# & (tmp not in cause_list):
    #         pos_target_tweet.append(pos_coarse_tweet[i])
    # Sample z from x (target_tweet)
    # Compute z_prime (x'), z_vector, z_prob (f(z)), and z_weight (Pi_x(z)) for all samples
    z_vectors_list = [] 
    z_probs_list = [] 
    z_weights_list = [] 

    z_strs_list, z_primes_list,in_onto_concepts_stem,in_onto_concepts,abstract,isolated_concept,tuples_list,target_label,target_vector, local_vocab_list,local_vocab_dict, target_tweet_remPunc, target_tweet_stem_remPunc,rem_conj_tweet,rem_conj_tweet_remPunc,rem_posistion,anchor_position,position_list = ontology_based_sample_z(target_tweet,pos_target_tweet,ontology,abstract_concepts,model,stopword,cause_list,join_list,concept_assign_dict,vocab_matrix,vocab_cc,max_length,
                                    local_fid=params['local_fid'],
                                    no_circle=params['no_circle'],
                                    no_repeat=params['no_repeat'],
                                    sample_normal=params['sample_normal'],
                                    sample_onto=params['sample_onto'],
                                    min_words_per_tweet=params['min_words_per_tweet'])
    print('done sampling')
    # true_truple = [] 
    # for i in tuples_list:
    #     if (i not in true_truple) & (i != []):
    #         true_truple.append(i) 
    # print(true_truple)
    # len_ = len(true_truple)
    # print(len_)
    # return len_
    num_samples = len(z_strs_list)                                
    for i in range(num_samples): # create num_samples of samples
        z_str = z_strs_list[i]
        z_str = ' '.join(z_str)
        z_str = z_str.replace('.',' ') 
        sen_tk =  nltk.word_tokenize(z_str)
        x_tmp = [vocab_cc[k] for k in sen_tk]
        z_vector = pad_sequences([x_tmp] , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
        # target_label = np.argmax(target_predict)
        # z_vector = tfidf_vectorizer.transform([z_str])
        z_vectors_list.append(z_vector)
        # probs for target_class for tweets in current sample
        z_probs = np.max(model.predict(z_vector, verbose=0), axis = 1)
        # z_probs = classify_prob(z_vector, classifier, target_class=target_label)
        z_probs_list.append(z_probs)
        
        # weights for each tweet in current sample
        z_weights = ontology_based_weights(target_vector, z_vector, ontology, vocab_cc,
                                kernel_width=params['kernel_width'],
                                metric=params['metric'], cosine_scale=params['cosine_scale'])
        z_weights_list.append(z_weights)

    z_primes_list = np.asarray(z_primes_list) 
    z_probs_list = np.asarray(z_probs_list) 
    z_weights_list = np.asarray(z_weights_list)
    len_real_vocab = z_primes_list.shape[1]

    '''===================Tensorflow part====================='''
    the_graph = tf.Graph()
    with the_graph.as_default():
        # label
        T_f_z = tf.placeholder(dtype=tf.float32, shape=[None,1])
        # features
        T_z_prime = tf.placeholder(dtype=tf.float32, shape=[None, len_real_vocab])
        # sample weights
        T_sample_weight = tf.placeholder(dtype=tf.float32, shape=[None,1])
        # coef to estimate
        T_w = tf.Variable(initial_value=tf.random_uniform([len_real_vocab]), dtype=tf.float32)
        # intercept to estimate
        T_b = tf.Variable(initial_value = tf.random_uniform([1]), dtype=tf.float32)
        # compute g(z') for each tweets
        T_g_z_all = T_z_prime * T_w #+ T_b # PL: Add T_b
        T_g_z_prime = tf.reduce_sum(T_g_z_all, axis=1) # shape=[None, explain_size]
        T_g_z_prime = tf.expand_dims(T_g_z_prime, axis=1)

        loss = tf.square(T_sample_weight * (T_f_z - T_g_z_prime))
        loss_sum = tf.reduce_sum(loss)

        if reg == 'L0':
            l0_regular = tf.count_nonzero(T_w)
            l0_regular = tf.cast(l0_regular, tf.float32)
            alpha = params['l0_reg']
            total_loss = loss_sum + alpha * l0_regular
        elif reg == 'L1':
            l1_regular = tf.abs(T_w)
            alpha = params['l1_reg']
            total_loss = loss_sum + alpha * tf.reduce_sum(l1_regular)
        else:
            total_loss = loss_sum

        if method_train == 'Gradient':
            # Train optimizer using Gradient
            opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
            gradients = opt.compute_gradients(total_loss)
            train_step = opt.apply_gradients(gradients)
        elif method_train == 'Adam': 
            # Train optimizer using Adam 
            train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(total_loss)

    with tf.Session(graph=the_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(int(params['num_epoch'])):
            step_f_z   = z_probs_list
            step_z_primes  = z_primes_list #np.expand_dims(z_primes_list, axis=0)
            step_sample_weights = z_weights_list
            
            sess.run(train_step, feed_dict={T_f_z:step_f_z,
                                            T_z_prime:step_z_primes,
                                            T_sample_weight:step_sample_weights})
            _,_,_,_,_,_,_,_ = sess.run([total_loss,T_f_z,T_z_prime,T_sample_weight,T_w,T_b,T_g_z_all,T_g_z_prime], feed_dict={T_f_z:step_f_z,
                                            T_z_prime:step_z_primes,
                                            T_sample_weight:step_sample_weights})
            # print(total_loss_)
        learned_w = T_w.eval(session=sess)
        w_and_idx = [(i, learned_w[i]) for i in range(len(learned_w))]
        learned_w_dict = dict(w_and_idx)
    print('before reasoning')
    accuracy, accuracy2, accuracy3, rn_vect = ontology_reasoning(target_idx, target_tweet, target_tweet_remPunc, target_tweet_stem_remPunc,target_label, actual_label,coarse_tweet,rem_conj_tweet,rem_conj_tweet_remPunc,learned_w_dict, local_vocab_dict, result_path,model,ontology,in_onto_concepts_stem,in_onto_concepts,abstract,isolated_concept,tuples_list,rem_posistion,concept_assign_dict,pos_target_tweet,issue_text,cc_property,anchor_position,position_list,record,accuracy,rn_vect)
    print('Great job!')
    return accuracy, accuracy2, accuracy3, rn_vect






##-------------- ************* --------------##
##-------------- Main function --------------##
##-------------- ************* --------------##
tokenizer = RegexpTokenizer(r'\w+') 
occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)  
# stop_words = set(stopwords.words('english'))
word_vectors = w2v_model.wv
tmp = np.load('../data/onto_word_CC_anchor_071119_stem.npz')
ontology_list = tmp['ontology_list2']
ontology_idx = tmp['ontology_idx']
onto_vect =  []
for i in ontology_list:
    onto_vect.append(stack_info_pl(i))
ontology = [ontology_list, ontology_idx, onto_vect]
tmp = np.load('../data/onto_word_CC_anchor_071119_lower.npz')
ontology_list = tmp['ontology_list']
ontology_idx = tmp['ontology_idx']
onto_vect =  []
for i in ontology_list:
    onto_vect.append(stack_info_pl(i))
ontology_noStem = [ontology_list, ontology_idx, onto_vect]

if __name__ == '__main__':
    # Parameters
    local_fid = 10 # radius of circles drawn around a word
    no_circle = 3 # number of circles drawn in the tweet 
    num_epoch = 100 # number of epochs to train the linear model
    explain_size = 50
    method_train = 'Adam' # or 'gradient'
    l0_reg = 0.01
    l1_reg = 0.01
    max_length = 100
    reg = 'No'
 
    all_info = pd.read_csv('../data/custom_stopwords_3.csv')
    stopword = all_info['Stopword']
    stopword = [w for w in stopword]
    punct_list = [':','.','!','?',';']
    join_list  = ['and', 'but', 'or','to']
    cause_list = []
    # cause_list = [ 'because', 'cuz' , 'since',  'whereas', 'while', 'therefore', 'thus', 
    #             'thereby', 'meanwhile', 'however', 'hence', 'otherwise', 'consequently', 'when'] 
    # cause_list = [ 'because', 'cuz' , 'since',  'whereas', 'while', 'therefore', 'thus', 'as','once','so',
    #             'thereby', 'meanwhile', 'however', 'hence', 'otherwise', 'consequently', 'when'] 

    all_info = pd.read_csv('../data/concepts_property_cc_071119.csv')
    domain = all_info['Domain']
    rel = all_info['Relationship']
    range_ = all_info['Range']
    cc_property = []
    for i in range(len(rel)):
        cc_property.append([domain[i], rel[i], range_[i]])

    concept_assign_dict = {}
    # for idx in range(6):
    #     if idx in [5]:
    #         concept_assign_dict[idx] = ['adj']
    #     else:
    #         concept_assign_dict[idx] = ['noun']

    # Load pretrained model & ontology lstm_cc_100_padpost_keeppre
    with open('../model/lstm_cc_100_padpost_keeppre.json','r') as f:
        json = f.read()
    model = model_from_json(json)
    model.load_weights("../model/lstm_cc_100_padpost_keeppre.h5")
    # # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # Read concepts
    all_info = pd.read_csv('../data/abstract_concepts_CC_071119.csv')
    concepts = all_info['Concepts']
    abstracts = all_info['Abstract']
    concepts = [concepts[ i] for i in range(len(concepts))]
    abstracts = [abstracts[ i] for i in range(len(abstracts))]
    abstract_concepts = [concepts, abstracts]

    # Load vocab dictionary, voc matrix and inverse dictionary
    data = np.load('../data/vocab_dict_cc_noPunct_2.npz')
    vocab_matrix = data['vocab_matrix']
    with open('../data/vocab_cc_noPunct_2.pkl', 'rb') as inf:
        vocab_cc = pickle.load(inf)

    # Load definition of each class
    data = np.load('../data/x_all_noPunct.npz')
    issue_text= data['issuse_text']

    # Read property
    test = 'Mortgage_100_2' # 13965
    all_info = pd.read_csv('../data/CC_' + str(test) + '.csv') 
    coarse_tweets = all_info['coarse']
    clean_tweet_puncts = all_info['clean_with_punct']
    labels = all_info['issue_no']
    coarse_tweets = [coarse_tweets[ i] for i in range(len(coarse_tweets))]
    clean_tweet_puncts = [clean_tweet_puncts[ i] for i in range(len(clean_tweet_puncts))]
    no_test = len(labels) 
    print('no_test:', no_test)
    print('Read test tweets !')

    start_time = time.time()
    len_all = []
    idx_all = []
    data = np.load('../data/sorted_cc.npz')
    idx_all = data['idx_all']
    # len_all = data['len_all']
    no_test =  [1]# [44]
    cnt = 0
    record = []
    # data = np.load('record_all_cc_3.npz')
    # ac = data['accuracy']
    # ac2 = data['accuracy2']
    # ac3 = data['accuracy3']

    # accuracy = [ac[i] for i in range(len(ac))]
    # accuracy2 = [ac2[i] for i in range(len(ac2))]
    # accuracy3 = [ac3[i] for i in range(len(ac3))]
    accuracy = []
    accuracy2 = []
    accuracy3 = []
    rn_vect = []

    for i in no_test: #  idx_all:#    range(cnt, len(idx_all)):#       reversed(idx):#    range(no_test):  #   
        # i  = idx_all[ii]
        cnt += 1
        print('cc ', cnt)  
        print(i)      
        coarse_tweet = coarse_tweets[i]
        coarse_tweet = coarse_tweet.replace(' .  .  .  . ',' . ')
        coarse_tweet = coarse_tweet.replace(' .  .  . ',' . ')
        coarse_tweet = coarse_tweet.replace(' .  . ',' . ')
        coarse_tweet_mod_remPunc = rem_punc(coarse_tweet)
        coarse_tweet_mod = coarse_tweet_mod_remPunc.split()
        extra_list = [c for c in coarse_tweet_mod if c not in vocab_cc]
        coarse_tweet_tk = coarse_tweet.split()
        coarse_tweet = [c for c in coarse_tweet_tk if c not in extra_list]
        coarse_tweet = ' '.join(coarse_tweet)
        target_tweet = coarse_tweet
        # target_tweet = clean_tweet_puncts[i]
        actual_label = labels[i]
        print('Target_tweet: ', target_tweet)
        len_tweet = len(target_tweet.split())
        if len_tweet <=local_fid+1:
            print('Too short, no need to explain')
        else:
            if len_tweet < 10:
                no_repeat = 2
            else:
                no_repeat = 1
            # explain_size = len_tweet * no_repeat # num of tweets sampled from original tweets
            if len_tweet > local_fid: 
                num_samples = len_tweet * (len_tweet - local_fid) * no_repeat # number of z sampled from explain_size tweets, +1: always consist of original target tweet
            else:
                num_samples = len_tweet * no_repeat
            
            params = {'explain_size':explain_size,
                    'num_samples':num_samples,
                    'method_train':method_train,
                    'reg':reg,
                    'l0_reg':l0_reg,
                    'l1_reg':l1_reg,
                    'no_repeat': no_repeat,
                    'local_fid':local_fid, # radius of circles drawn around a word
                    'no_circle': no_circle, # number of circles drawn in the tweet
                    'sampling_target':{0: 0.5, 1: 0.5},
                    'num_epoch':num_epoch,
                    'learning_rate':0.001,
                    'base_sample_prob':0.9, # the sample prob to begin with when sampling x
                    'min_sample_prob':0.005, # minimun sample prob for sampling x
                    'sample_prob_decay':0.9, # the rate that sample probability decays when tweets are less similar (x)
                    'weight_modifier':1.5, # modify weight for ontology in vector
                    'sample_normal':0.6, # the rate to sample words from tweets (z)
                    'sample_onto':0.2, # the rate modifier for ontology (z)
                    'min_words_per_tweet':1, # minimum number of words in a sampled tweet
                    'kernel_width':25, # kernal width parameter
                    'metric':'cosine', # 'cosine' or 'euclidean'
                    'cosine_scale':100 # scale to enlarge cosine dsitance values for kernel function
                    }

            # result_path = 'Result/CC_' + test + '_reg{}_L0_{}_L1_{}_train{}_locRadius{}_numCircle{}_normalRate{}_ontoRate{}.txt'.format(
            #     params['reg'],params['l0_reg'],params['l1_reg'],params['method_train'],params['local_fid'],params['no_circle'],params['sample_normal'],params['sample_onto'])
            result_path = '../result/CC_Mortgage_check.txt'
            
            accuracy, accuracy2, accuracy3, rn_vect = explain_classifier_tf_count(i,target_tweet,coarse_tweet,actual_label,stopword,cause_list,join_list,abstract_concepts,model,ontology,params,result_path,concept_assign_dict,vocab_matrix,vocab_cc,max_length,issue_text,cc_property, record, accuracy, accuracy2, accuracy3,rn_vect)
            if cnt%10 == 0:
                np.savez('rn_vect' + str(cnt) + '.npz', rn_vect=rn_vect)
                np.savez('record_all_cc_' + str(cnt) + '.npz', accuracy=accuracy,accuracy2=accuracy2,accuracy3=accuracy3)
            print('------------ ******** ------------')
    # # print("Finish in %s seconds ---" % (time.time() - start_time))
    # # print(len(idx))
    #     if len_tuples_list >=1:
    #         len_all.append(len_tuples_list)
    #         idx_all.append(i)
    #         np.savez('tuples_cc.npz',idx_all=idx_all,len_all=len_all)
    # sort_idx =sorted((e,i) for i,e in enumerate(len_all))
    # idx  = []
    # for i1 in range(len(sort_idx)):
    #     idx.append(idx_all[sort_idx[i1][1]])
    # np.savez('sorted_cc.npz',idx=idx, idx_all=idx_all,len_all=len_all,sort_idx=sort_idx )

       # # Compare similarity - MAX
    # rules_OSIL_refined = []
    # for i in range(len(rules_OSIL)):
    #     sim_score = []
    #     for j in range(len(rules)):
    #         osil = np.expand_dims(rules_OSIL_vect[i], axis=0)
    #         ollie = np.expand_dims(rules_remPunc_vect[j], axis=0)
    #         s = cosine_similarity(osil, ollie)[0]
    #         sim_score.extend(s)
    #     ind = np.argmax(sim_score)
    #     v_max = np.max(sim_score)
    #     if (v_max > 0.5):
    #         r_mod = rules_OLLIE[ind]
    #     else:
    #         r_mod = rules_OSIL[i]
    #     rules_OSIL_refined.append(r_mod)