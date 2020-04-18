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
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import paired_cosine_distances
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
from copy import copy
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

tokenizer = RegexpTokenizer(r'\w+') 


def rem_punc(tweet):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_tweet = regex.sub(' ', tweet)
    return clean_tweet

# Build vocabulary 
def build_vocab(token_lists):
    word_counts = Counter(itertools.chain(token_lists))
    word_frequency_list_cc = word_counts.most_common()  # sort the frequency list
    inv_vocab_cc = [x[0] for x in word_frequency_list_cc] # [0] position is a placeholder
    vocab_cc = {x : i for i,x in enumerate(inv_vocab_cc)} # reserve 0 for padding
    return vocab_cc, inv_vocab_cc, word_frequency_list_cc

def build_vocab_matrix(inv_vocab_cc, word_vectors, model):
    vocab_matrix = []
    for i in range(len(inv_vocab_cc)):
        word = inv_vocab_cc[i]
        if word in word_vectors.vocab:
            vocab_matrix.append(model[word])
        else:
            tmp = np.asarray([0]*300)
            vocab_matrix.append(tmp)
    return vocab_matrix






######## Build vocabulary and inverse for CC dataset ########
all_info = pd.read_csv('../../data/CC_Mortgage_100.csv')
coarse_cc_list = all_info['coarse']
token_list = []
for i in range(len(coarse_cc_list)):  # 
    print(i)
    cc =  coarse_cc_list[i]
    cc = rem_punc(cc)
    cc_token = nltk.word_tokenize(cc)
    token_list.extend(cc_token)
vocab_cc, inv_vocab_cc, word_frequency_list_cc = build_vocab(token_list)
print('len of dict: ', len(inv_vocab_cc))
# Save vocab and inverse
f = open("../../data/vocab_cc_noPunct.pkl","wb")
pickle.dump(vocab_cc,f)
f.close()
np.savez('../../data/vocab_dict_cc_noPunct.npz',inv_vocab_cc=inv_vocab_cc,word_frequency_list_cc=word_frequency_list_cc)

######## Build vocabulary matrix for CC dataset ########
data = np.load('../../data/vocab_dict_cc_noPunct.npz')
inv_vocab_cc = data['inv_vocab_cc']
word_frequency_list_cc = data['word_frequency_list_cc']
with open('../../data/vocab_cc_noPunct.pkl', 'rb') as inf:
    vocab_cc = pickle.load(inf)
model = gensim.models.KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)  
all_info = pd.read_csv('../../data/custom_stopwords_3.csv')
stopword = all_info['Stopword']
stop_words = [w for w in stopword]
word_vectors = model.wv
print('Done load w2v model')
vocab_matrix = build_vocab_matrix(inv_vocab_cc, word_vectors, model)
np.savez('../../data/vocab_dict_cc_noPunct_2.npz',vocab_matrix=vocab_matrix, inv_vocab_cc=inv_vocab_cc,word_frequency_list_cc=word_frequency_list_cc)
f = open("../../data/vocab_cc_noPunct_2.pkl","wb")
pickle.dump(vocab_cc,f)
f.close()
