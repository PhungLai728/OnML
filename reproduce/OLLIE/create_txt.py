
"""
Author: Phung Lai, CCS, NJIT
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time

def save_txt(infor):
    valid_rows = []
    for i in range(len(infor)):
        print(i)
        sentence = str(infor[i])
        f= open("OLLIE_cc_in/in" + str(i),"w+")
        f.write(sentence)
        f.close() 
    return 1

all_info = pd.read_csv('CC_Mortgage_example.csv')
tweet = all_info['coarse']
print('There are ', len(tweet), ' consumer complaints!')
save_txt(tweet)
print('Done')