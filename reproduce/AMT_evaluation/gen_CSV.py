"""
Author: Phung Lai, CCS, NJIT
"""

# Generate CSV file for uploading to AMT
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import os

files = os.listdir('./screen_shots/')
all_url  = []
for f in files:
    url = 'https://mturk-cc.s3.amazonaws.com/' + f
    all_url.append(url)
print(len(all_url))

data_w = [all_url]
my_csv = pd.DataFrame(data_w)
my_csv = pd.DataFrame.transpose(my_csv)
my_csv.to_csv('mturk-cc.csv', index=False)
print('saving')




# https://s3.amazonaws.com/mturk-s3-partial/tweet_1041.png