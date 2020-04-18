import pandas as pd
import numpy as np
import itertools
import time
import re
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
start_time = time.time()
from data_utils_4 import compile_re2, raw_tweet_prep
html_compiled, space_replace_compiled, repeating_compiled, single_char_compiled = compile_re2()

tokenizer = RegexpTokenizer(r'\w+') 

from textblob import TextBlob
from textblob import Word

all_info = pd.read_csv('../../data/custom_stopwords_3.csv')
stopword = all_info['Stopword']
stopword = [w for w in stopword]
# stopword = []

start_time = time.time()
# all_info = pd.read_csv('Consumer_Complaints_Mortgage.csv')
all_info = pd.read_csv('../../data/CC_Mortgage_100.csv')
print("Finish load data in %s seconds ---" % (time.time() - start_time))
issue_no = all_info['issue_no']
issue_text = all_info['issue_text']
coarse = all_info['coarse']
issue_no = [i for i in issue_no]
issue_text = [i for i in issue_text]
cc_narrative = [i for i in coarse]

clean_cc_list = []
coarse_cc_list = []
clean_cc_punt_list = []
# idx = [2269,1071,6669,2726]
for i in range(len(issue_text)): #range(10): #
    print(i)
    start_time = time.time()
    sentences = cc_narrative[i]
    clean_cc, clean_cc_punt, coarse_cc = raw_tweet_prep(sentences, stopword, html_compiled, space_replace_compiled, repeating_compiled, single_char_compiled)   

    coarse_cc_list.append(coarse_cc)
    clean_cc_list.append(clean_cc)
    clean_cc_punt_list.append(clean_cc_punt)

my_csv = pd.DataFrame([issue_no,issue_text,clean_cc_punt_list,coarse_cc_list])
my_csv = pd.DataFrame.transpose(my_csv)
my_csv.to_csv('../../data/CC_Mortgage_100_3.csv', index=False)
print('Great job!')