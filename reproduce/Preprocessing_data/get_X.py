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
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def rem_punc(tweet):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_tweet = regex.sub(' ', tweet)
    return clean_tweet

########-------------------- MAIN SCRIPT --------------------########
# Load vocab dictionary, voc matrix and inverse dictionary
data = np.load('../../data/vocab_dict_cc_noPunct_2.npz')
inv_vocab_cc = data['inv_vocab_cc']
word_frequency_list_cc = data['word_frequency_list_cc']
vocab_matrix = data['vocab_matrix']
with open('../../data/vocab_cc_noPunct_2.pkl', 'rb') as inf:
    vocab_cc = pickle.load(inf)
# Load data
all_info = pd.read_csv('../../data/CC_Mortgage_100_2.csv')
issue = all_info['issue_no']
clean_cc_punt_list = all_info['clean_with_punct']
coarse_cc_list = all_info['coarse']
# Vocabualarize
x_all = []
n_classes = 16
issue1 = pd.factorize(issue)[0]
issue2 = pd.factorize(issue)
a1 = issue2[0]
a2 = issue2[1]
issuse_no = [i for i in a1]
issuse_text = [i for i in a2]

issue_label = to_categorical(issue1, num_classes=n_classes)
y_all = copy(issue_label)
len_ = []
for i in range(10): # range(len(clean_cc_punt_list)): # 
    print(i)
    cc = clean_cc_punt_list[i]
    cc = rem_punc(cc)
    sen_tk =  nltk.word_tokenize(cc)
    x_tmp = [vocab_cc[k] for k in sen_tk]
    x_all.append(x_tmp)
    len_.append(len(x_tmp))
np.savez('x_all_noPunct.npz',x_all=x_all, len_=len_,issuse_no=issuse_no,issuse_text=issuse_text)

