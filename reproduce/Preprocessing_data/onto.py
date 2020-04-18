import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import PorterStemmer
st = PorterStemmer()


tokenizer = RegexpTokenizer(r'\w+') 
def check_len(info):
    len_list = []
    for sentence in info:
        sentence =str(sentence)
        word_tokens = tokenizer.tokenize(sentence) # remove punctuation
        len_list.append(len(word_tokens))
    max_len = max(len_list)
    return len_list, max_len



########## Run First ##########
# Read data
all_info = pd.read_csv('../../data/ConSo_onto.csv')
entity = all_info['Entity']
concept = all_info['Superclass(es)']
# main_concepts = ['Negative','Object'] 
main_concepts = ['Complaint','Event','Event_Outcome','Product','Property','Thing_in_role','Negative','Object'] 
# onto = [ for i in range(len(entity))]

onto = []
for i in range(len(entity)):
    tmp = entity[i]
    tmp = tmp.strip('\'')
    tmp = tmp.strip('"') 
    onto.append(tmp)

ontology_list = []
ontology_idx = []
not_found_list = []
not_found_idx = []
for i in range(len(onto)): 
    raw = onto[i]
    print(raw)
    raw = raw.strip('\'')
    words = re.split(r'\s+', re.sub(r'[\_()!?]', ' ', raw).strip())
    if len(words) == 1:
        tmp_c = concept[i]
        if tmp_c != 'owl:Thing':
            while True:
                if tmp_c in main_concepts:
                    idx = main_concepts.index(tmp_c)
                    ontology_idx.append(idx)
                    ontology_list.extend(words) 
                    # print('ontology_idx:', ontology_idx)
                    # print('ontology_list:', ontology_list)
                    # print(main_concepts[idx])
                    break
                else:
                    if tmp_c in onto:
                        idx = onto.index(tmp_c)
                        tmp_c = concept[idx]

                    # ------- Specical case: Cannot find main concept cases (because of transforming from Protege to CSV)
                    else: 
                        # Case of belonging to several concept/subconcepts
                        # word_tokens = tokenizer.tokenize(tmp_c)
                        word_tokens = tmp_c.split()
                        # if len(word_tokens) > 1:
                        for tmp_c in word_tokens:
                            tmp_c = tmp_c.strip('\'')
                            tmp_c = tmp_c.strip('"')   
                            while True:
                                if tmp_c in main_concepts:
                                    idx = main_concepts.index(tmp_c)
                                    ontology_idx.append(idx)
                                    ontology_list.extend(words)
                                    # print('ontology_idx:', ontology_idx)
                                    # print('ontology_list:', ontology_list)
                                    # print(main_concepts[idx])
                                    break
                                else:
                                    if tmp_c in onto:
                                        idx = onto.index(tmp_c)
                                        tmp_c = concept[idx]
                        # else:
                        #     tmp_c = word_tokens.strip('\'')
                        #     tmp_c = tmp_c.strip('"')   
                        #     while True:
                        #         if tmp_c in main_concepts:
                        #             idx = main_concepts.index(tmp_c)
                        #             ontology_idx.append(idx)
                        #             ontology_list.append(words)
                        #             # print('ontology_idx:', ontology_idx)
                        #             # print('ontology_list:', ontology_list)
                        #             # print(main_concepts[idx])
                        #             break
                        #         else:
                        #             if tmp_c in onto:
                        #                 idx = onto.index(tmp_c)
                        #                 tmp_c = concept[idx]

                            # not_found_list.append(raw)
                            # not_found_idx.append(i)
                            # break
                     # -------  End specical cases ------- #

        else:
            idx = main_concepts.index(raw) 
            ontology_idx.append(idx)
            ontology_list.extend([raw]) 
print('Read ontology !')

# Remove dedundancy
list1 = ontology_list
idx1 = ontology_idx
ontology_list = []
ontology_idx = []

occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)

count = 0
idx = 0
while True:
    if count > len(list1)-1:
        break
    else: 
        # print(count)
        tmp = list1[count]
        all_inx = list(occurrences(tmp, list1))
        label = [idx1[ i] for i in all_inx]
        label = list(set(label))
        for i in range(len(label)):
            ontology_list.append(tmp)
            ontology_idx.append(label[i])
        count = count + len(all_inx)
        idx = idx + 1

np.savez('../../data/onto_CC.npz',ontology_list=ontology_list, ontology_idx=ontology_idx)
data_w = [ontology_list,ontology_idx]
my_csv = pd.DataFrame(data_w)
my_csv = pd.DataFrame.transpose(my_csv)
my_csv.to_csv('../../data/onto_CC.csv', index=False)
print('saving')



# ########## Run Second ##########
# ------------ Chaneg to lower case ------------ #
all_info = pd.read_csv('../../data/onto_CC.csv')
entity = all_info['0']
concept = all_info['1']
ontology_list = entity.apply(lambda x: " ".join(x.lower() for x in x.split()))
ontology_list = [ontology_list[i] for i in range(len(ontology_list))]
data = np.load('../../data/onto_CC.npz')
ontology_idx=data['ontology_idx']
ontology_idx = [ontology_idx[i] for i in range(len(ontology_idx))]

ontology_list2 = [st.stem(w) for w in ontology_list]


# np.savez('../../data/onto_word_CC_anchor_071119_lower.npz',ontology_list=ontology_list, ontology_idx=ontology_idx)
np.savez('../../data/onto_CC_lower.npz',ontology_list=ontology_list, ontology_idx=ontology_idx)
data_w = [ontology_list,ontology_idx]
my_csv = pd.DataFrame(data_w)
my_csv = pd.DataFrame.transpose(my_csv)
my_csv.to_csv('../../data/onto_CC_lower.csv', index=False)
print('saving')


# np.savez('../../data/onto_word_CC_anchor_071119_lower.npz',ontology_list2=ontology_list2, ontology_idx=ontology_idx)
np.savez('../../data/onto_CC_stem.npz',ontology_list2=ontology_list2, ontology_idx=ontology_idx)
data_w = [ontology_list2,ontology_idx]
my_csv = pd.DataFrame(data_w)
my_csv = pd.DataFrame.transpose(my_csv)
my_csv.to_csv('../../data/onto_CC_stem.csv', index=False)
print('saving')
# ------------ Change to lower case ------------ #