import pandas as pd
import numpy as np
#from keras.utils import to_categorical
import itertools
import pandas as pd
import numpy as np
import preprocessor as p
import re
#from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer
st = PorterStemmer()
from textblob import Word
from copy import copy
from nltk.tokenize import word_tokenize
import string

occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
#==================================================
def expand_neg(tweet):
    all_info = pd.read_csv('../../data/neg_list.csv')
    list_neg = all_info['List']
    change_to = all_info['Change']
    new_tweet = copy(tweet)
    for i in range(len(list_neg)):
        if list_neg[i] in tweet:
            new_tweet = new_tweet.replace(list_neg[i],change_to[i]) 
    # new_tweet = tweet.replace("don’t",' not') 
    return new_tweet

def change_neg(tweet):
    list_neg = ["isnt","isn't","arent","aren't","wasnt","wasn't","werent","weren't","hasnt","hasn't","havent","haven't","hadnt",
                "hadn't","doesnt","doesn't","dont","don't","didnt","didn't","wont","won't","wouldnt","wouldn't","shant",
                "shan't","shouldnt","shouldn't","cant","can't","cannot","couldnt","couldn't","mustnt","mustn't",
                "isn’t","aren’t","wasn’t","weren’t","hasn’t","haven’t","hadn’t","doesn’t","don’t","didn’t","won’t","wouldn’t",
                "shan’t","shouldn’t","can’t","couldn’t","mustn’t","ain't","ain’t"]
    new_tweet = copy(tweet)
    for i in range(len(list_neg)):
        if list_neg[i] in tweet:
            new_tweet = new_tweet.replace(list_neg[i],' not') 
    # new_tweet = tweet.replace("don’t",' not') 
    return new_tweet

def remove_X(infor):
    sentence = str(infor)
    result = re.sub('XX/XX/XXXX', '', sentence)
    result = re.sub('XXXX/XXXX/XXXX', '', result)
    result = re.sub('XX/XX/', '', result)
    result = re.sub('XXXX', '', result)
    result = re.sub('XX', '', result)
    return result

# pre-process (without stemming) a raw tweet and return a list of words
def raw_tweet_prep(raw_tweet, stopwords, html_re, space_replace_re, repeating_re, single_char_re):
    tweet_tokenized = html_re.sub(' ', raw_tweet)
    tweet_tokenized = remove_X(tweet_tokenized)
    tweet_tokenized = p.tokenize(tweet_tokenized.lower().replace('\n',' ')) # Change to URL & lower case
    tweet_tokenized = re.sub('.00', '.', tweet_tokenized)
    # tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
    tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
    tweet_tokenized = tweet_tokenized.strip('\'[],\_"')
    # coarse_tweet = copy(tweet_tokenized)
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ : List of punctuation
    # weet_tokenized = tweet_tokenized.replace(':','.') # Change ? to .
    # tweet_tokenized = tweet_tokenized.replace("’t",' not') 
    tweet_tokenized = tweet_tokenized.replace("’s",' is') 
    tweet_tokenized = tweet_tokenized.replace("’re",' are') 
    tweet_tokenized = tweet_tokenized.replace("’d",' would') 
    tweet_tokenized = tweet_tokenized.replace("’ll",' will') 
    tweet_tokenized = tweet_tokenized.replace("’all",' will') 
    tweet_tokenized = tweet_tokenized.replace("’ve",' have')
    tweet_tokenized = tweet_tokenized.replace("’m",' am') 
    tweet_tokenized = " ".join(tweet_tokenized.split()) 
    tweet_tokenized = tweet_tokenized.replace('\\n',' ')
    tweet_tokenized = tweet_tokenized.replace("'s",' is')  
    tweet_tokenized = tweet_tokenized.replace("'re",' are') 
    tweet_tokenized = tweet_tokenized.replace("'d",' would') 
    tweet_tokenized = tweet_tokenized.replace("'ll",' will') 
    tweet_tokenized = tweet_tokenized.replace("'all",' will')
    tweet_tokenized = tweet_tokenized.replace("'ve",' have') 
    tweet_tokenized = tweet_tokenized.replace("'m",' am') 
    # Dealing with compound words can/not d/'ye gim/me gon/na got/ta lem/me mor/'n wan/na
    tweet_tokenized = tweet_tokenized.replace("gimme",' give me')
    tweet_tokenized = tweet_tokenized.replace("gonna",' going')
    tweet_tokenized = tweet_tokenized.replace("gotta",' got to')
    tweet_tokenized = tweet_tokenized.replace("lemme",' let me')
    tweet_tokenized = tweet_tokenized.replace("wanna",' want to')
    tweet_tokenized = expand_neg(tweet_tokenized)
    r = re.compile(r'([.,/#!$%^&*;:{}=_`~()-])[.,/#!$%^&*;:{}=_`~()-]+') # Remove consecutive duplicate punctions
    tweet_tokenized = r.sub(r'\1', tweet_tokenized)

    # tweet_tokenized = change_neg(tweet_tokenized)
    # chars_to_remove = ['[',']','‘','*',':','+',';','"','\\','<','=','>','^','’','\'','`','{','}','~', '(',')','@','-','%','&','#','$','_','/','|', ',']
    # tweet_tokenized = ''.join([i for i in tweet_tokenized if i not in chars_to_remove]) # Remove specific symbols
    tweet_tokenized = re.sub(r'[^a-zA-Z0-9.!?:;\s]+', '', tweet_tokenized)
    tweet_tokenized = tweet_tokenized.replace('.',' .') # Change ? to .
    tweet_tokenized = tweet_tokenized.replace('!',' .') # Change ! to .
    tweet_tokenized = tweet_tokenized.replace('?',' .') # Change ? to .
    tweet_tokenized = tweet_tokenized.replace(':',' .') # Change ? to .
    tweet_tokenized = tweet_tokenized.replace(';',' .') # Change ? to .
    coarse_tweet = "".join(tweet_tokenized)

    # Remove punctuation & stop words
    tweet_tokenized = tweet_tokenized.strip().split()
    words = [w for w in tweet_tokenized if w not in stopwords]
    clean_tweet_punct = ' '.join(words)
    # r = re.compile(r'([.,/#!$%^&*;:{}=_`~()-])[.,/#!$%^&*;:{}=_`~()-]+') # Remove consecutive duplicate punctions
    # clean_tweet_punct = r.sub(r'\1', clean_tweet_punct)

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_tweet = regex.sub(' ', clean_tweet_punct)
    return clean_tweet, clean_tweet_punct, coarse_tweet

# # pre-process (without stemming) a raw tweet and return a list of words
# def raw_tweet_prep_test(raw_tweet, stopwords, html_re, space_replace_re, repeating_re, single_char_re):
#     tweet_tokenized = html_re.sub(' ', raw_tweet)
#     tweet_tokenized = p.tokenize(tweet_tokenized.lower().replace('\n',' ')) # Change to URL & lower case
#     tweet_tokenized = space_replace_re.sub(' ', tweet_tokenized)
#     tweet_tokenized = repeating_re.sub(r"\1", tweet_tokenized)
#     tweet_tokenized = tweet_tokenized.strip('\'[],\_"')
    
#     if len(tweet_tokenized) > 0:
#         if tweet_tokenized[len(tweet_tokenized)-1] == '…':
#             print('true')
#             coarse_tweet = copy(tweet_tokenized)
#             coarse_tweet = [coarse_tweet]
#             tweet = ['nan']
#             len_clean = 0
#         else:
#             chars_to_remove = ['@','-',':','%','#','_','/','|']
#             tweet_tokenized = ' '.join([i for i in tweet_tokenized if i not in chars_to_remove]) # Remove specific symbols
#             tweet_tokenized = tweet_tokenized.replace('!','.') # Change ! to .
#             tweet_tokenized = tweet_tokenized.replace('?','.') # Change ? to .
#             # tweet_tokenized = tweet_tokenized.replace("’t",' not') 
#             tweet_tokenized = tweet_tokenized.replace("’re",' are') 
#             tweet_tokenized = tweet_tokenized.replace("’ll",' will') 
#             tweet_tokenized = tweet_tokenized.replace("’ve",' have')
#             tweet_tokenized = tweet_tokenized.replace("’m",' am') 
#             tweet_tokenized = " ".join(tweet_tokenized.split()) 
#             tweet_tokenized = tweet_tokenized.replace('\\n',' ') 

#             tweet_tokenized = tweet_tokenized.replace("'m",' am') 
#             tweet_tokenized = tweet_tokenized.replace("'re",' are') 
#             tweet_tokenized = tweet_tokenized.replace("'ll",' will') 
#             tweet_tokenized = tweet_tokenized.replace("'ve",' have') 
#             tweet_tokenized = change_neg(tweet_tokenized)

#             coarse_tweet = copy(tweet_tokenized)
#             # Remove punctuation & stop words
#             regex = re.compile('[%s]' % re.escape(string.punctuation))
#             tweet_tokenized = regex.sub('', tweet_tokenized)
#             tweet_tokenized = tweet_tokenized.strip().split()
#             words = [w for w in tweet_tokenized if w not in stopwords]
#             if len(words) > 1:
#                 idx_e = len(list(occurrences('EMOJI', words)))
#                 idx_u = len(list(occurrences('URL', words)))
#                 len_clean = len(words) - idx_e - idx_u
#                 tweet = ' '.join(words)
#                 tweet = [tweet]
#                 coarse_tweet = [coarse_tweet]
#             else:
#                 coarse_tweet = copy(tweet_tokenized)
#                 coarse_tweet = [coarse_tweet]
#                 tweet = ['nan']
#                 len_clean = 0
#     else:
#         coarse_tweet = copy(tweet_tokenized)
#         coarse_tweet = [coarse_tweet]
#         tweet = ['nan']
#         len_clean = 0

#     return tweet, coarse_tweet, len_clean
# # stem = tweet_tokenized.apply(lambda x: " ".join([st.stem(word) for word in x.split()])) # Stemming
# # stem = [st.stem(word) for word in tweet_tokenized.split()]
# # tweet_tokenized = ' '.join(stem)  

# prettify a raw tweet
def prettify_raw_tweet(raw_tweet):
    html_re = re.compile(r"&#?\w+;")
    new_tweet = html_re.sub(' ', raw_tweet)
    new_tweet = ' '.join(new_tweet.split())
    return new_tweet

# read a list of stop words
def read_stopwords(filename):
    stopwords = []
    with open(filename, 'r', encoding='utf-8') as inf:
        for line in inf:
            stopwords.append(line.strip())
    stopwords = set(stopwords)
    return stopwords

def compile_re2():
    html_entities_re = r"&#?\w+;"
    quote_s_re = ""
    non_char_re = "[^ 0-9a-zA-Z]"
    html_compiled = re.compile(html_entities_re)
    space_replace_compiled = re.compile('|'.join([quote_s_re]))
    single_char_compiled = re.compile(r"(?:\b\w\b)+")
    repeating_compiled = re.compile(r"([a-zA-Z])\1\1+")
    return html_compiled, space_replace_compiled, repeating_compiled, single_char_compiled
