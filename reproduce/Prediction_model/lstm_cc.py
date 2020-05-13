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
from sklearn import metrics
from sklearn import pipeline
from sklearn.externals import joblib
from keras.wrappers.scikit_learn import KerasClassifier

tokenizer = RegexpTokenizer(r'\w+') 


# LSTM model by Keras
def model_lstm_relu_embedding(X_train, X_test, Y_train, Y_test, vocab_matrix, emb_dim, time_steps, hid_dim,max_length, keep,pad):
    model = Sequential() # try with the simple sequential model
    model.add(Embedding(input_dim=vocab_matrix.shape[0], output_dim=emb_dim,
            mask_zero=True, trainable=False, weights=[vocab_matrix],
            embeddings_regularizer=None, input_length=time_steps))
    model.add(LSTM(units=time_steps)) # the LSTM layer
    model.add(Dense(16, activation='softmax')) # logistic regression layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=512, epochs=2)
    _,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    # print("Logloss score: %.2f" % (score))
    print("Accuracy: %.2f" % (acc))
    
    # serialize model to JSON
    model_json = model.to_json()
    with open('lstm_cc_' + str(max_length) + '_pad' + str(pad) + '_keep' + str(keep) + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('lstm_cc_' + str(max_length) + '_pad' + str(pad) + '_keep' + str(keep) + '.h5')
    print("Saved model to disk")


    pickle.dump( model, open( "../../model/classifier_cc2.p", "wb" ) )
    # pickle.dump( vectorizer, open( "vectorizer_cc.p", "wb" ) )

    return model

occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
def split_to_train_test_(data, label, label_no, train_frac=0.8, test_frac=0.1, val_frac=0.1):
    labels = list(set(label_no))
    idx_train = []
    idx_test = []
    idx_val = []
    for lbl in labels:
        lbl_full = list(occurrences(lbl, label_no)) 
        idx = np.arange(0 , len(lbl_full))
        np.random.shuffle(idx)
        lbl_full = [lbl_full[idx[i]] for i in range(len(idx))] 
        num = int(np.floor(train_frac*len(lbl_full)))
        num2 = int(np.floor(test_frac*len(lbl_full)))
        idx_tr = lbl_full[:num]
        idx_te = lbl_full[num:num+num2]
        idx_va = lbl_full[num+num2:]

        idx_train.extend(idx_tr)
        idx_test.extend(idx_te)
        idx_val.extend(idx_va)

    # return train_idx, test_idx, val_idx
    x_train = [data[ i] for i in idx_train]
    y_train = [label[ i] for i in idx_train]
    x_test = [data[ i] for i in idx_test]
    y_test = [label[ i] for i in idx_test]
    x_val = [data[ i] for i in idx_val]
    y_val = [label[ i] for i in idx_val]
    # return x_train, y_train, x_test, y_test
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(x_val), np.asarray(y_val)

def split_data(data, label, percent):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    num = int(np.floor(percent*len(data)))
    idx_train = idx[:num]
    x_train = [data[ i] for i in idx_train]
    y_train = [label[ i] for i in idx_train]
    idx_test = idx[num:]
    x_test = [data[ i] for i in idx_test]
    y_test = [label[ i] for i in idx_test]
    # return x_train, y_train, x_test, y_test
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)

########-------------------- MAIN SCRIPT --------------------########
# Load vocab dictionary, voc matrix and inverse dictionary
data = np.load('../../data/vocab_dict_cc_noPunct_2.npz')
vocab_matrix = data['vocab_matrix']
n_classes = 16

data = np.load('../../data/x_all_noPunct.npz', allow_pickle=True)
len_= data['len_']
x_all= data['x_all']
max_length = 100
issuse_no= data['issuse_no'] 
issuse_text= data['issuse_text'] 

issue_label = to_categorical(issuse_no, num_classes=n_classes)
y_all = copy(issue_label)
# print('max_length', max_length)
x_pad = pad_sequences(x_all , maxlen=max_length, padding='post', truncating = 'post') # truncate the post, and keep pre
keep = 'pre'
pad = 'post'
# LSTM model
hid_dim = 500
emb_dim    = 300 # input dimension
time_steps = max_length
num_units  = 128 #hidden LSTM units
learning_rate = 0.001 #learning rate for adam
batch_size    = 512
n_classes = 16

train_percent = 0.7
print('start split')
start_time = time.time()
# x_train, y_train, x_test, y_test = split_data(x_pad, y_all, train_percent)
x_train, y_train, x_test, y_test, x_val, y_val = split_to_train_test_(x_pad, y_all, issuse_no, 0.8, 0.1, 0.1)

# Train LSTM model
print('start train')
start_time = time.time()
model_lstm = model_lstm_relu_embedding(x_train, x_test, y_train, y_test, vocab_matrix, emb_dim, time_steps, hid_dim, max_length, keep,pad)
print("done train in %s seconds ---" % (time.time() - start_time))
print('ok')


# # Test LSTM model
# print('start test')
# with open('Data/lstm_cc_100.json','r') as f:
#     json = f.read()
# model = model_from_json(json)
# model.load_weights("Data/lstm_cc_100.h5")
# # # evaluate loaded model on test data
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = model.evaluate(x_train, y_train, verbose=0)
# # pred = model.predict(x_train, verbose=1)
# a = x_train[0] 
# a_t = np.expand_dims(a, axis=0)
# pred = model.predict(a_t, verbose=1)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



