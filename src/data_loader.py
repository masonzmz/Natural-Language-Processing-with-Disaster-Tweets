'''
Author       : Mingzhe Zhang (s4566656)
Date         : 2022-08-28 19:56:07
LastEditors  : Mingzhe Zhang (s4566656)
LastEditTime : 2022-09-29 17:24:42
FilePath     : ./src/data_loader.py
'''

import pandas as pd

import string
import re

from colorama import Fore, Back, Style


CONTRACTION_DICT = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def data_preprocess():
    data_train = pd.read_csv('anaconda3/envs/mason/Kaggle_Disaster/data/train.csv')
    data_test = pd.read_csv('anaconda3/envs/mason/Kaggle_Disaster/data/test.csv')

    null_text = data_train['text'].isnull().sum()
    null_label = data_train['target'].isnull().sum()

    print(Fore.GREEN + f"{'Null value in training data':^20} | {'Null value in training label':^20} | {'Training data shape:':^10}")
    print(Fore.GREEN + "-"*100) 
    print(Fore.GREEN + f"{null_text:^27} | {null_label:^28} | {data_train.shape}")
    print(Fore.GREEN + "-"*100)
    print("\n")
    
    tweets_idx = {}
    for i in data_train.index:
        t = data_train.iloc[i]['text']
        indexes = data_train[(data_train['text'] == t)].index.values
        if len(indexes) > 1:
            tweets_idx[indexes[0]] = indexes
    
    tweets_target_duplicate={}
    target_0 = 0
    target_1 = 0

    for k in tweets_idx.keys():
        for i in range(0, len(tweets_idx[k])):
            target = data_train.iloc[tweets_idx[k][i]]['target']
            if target==0:
                target_0 = target_0 + 1
            else:
                target_1 = target_1 + 1
        tweets_target_duplicate[k] = [target_0,target_1]
        target_0 = 0
        target_1 = 0

    tweets_df = pd.DataFrame({'tweet id': tweets_target_duplicate.keys(), 'target 0': tweets_target_duplicate.values(), 'target 1': tweets_target_duplicate.values()})
    tweets_df['tweet id'] = tweets_df['tweet id'].apply(lambda index: data_train.iloc[index]['id'])
    tweets_df['target 0'] = tweets_df['target 0'].apply(lambda lst: lst[0])
    tweets_df['target 1'] = tweets_df['target 1'].apply(lambda lst: lst[1])

    inconsistent_labels = tweets_df[(tweets_df['target 0']>0) & (tweets_df['target 1']>0)]

    contractions, contractions_re = _get_contractions(CONTRACTION_DICT)

    data_train['text'] = data_train['text'].apply(lambda x : re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b').sub(r'',x))
    data_test['text'] = data_test['text'].apply(lambda x : re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b').sub(r'',x))

    data_train['text'] = data_train['text'].apply(lambda x : x.translate(str.maketrans('','',string.punctuation)))
    data_test['text'] = data_test['text'].apply(lambda x : x.translate(str.maketrans('','',string.punctuation)))

    # Clean Contractions
    data_train['text'] = data_train['text'].apply(lambda x:replace_contractions(x, contractions, contractions_re))
    data_test['text'] = data_test['text'].apply(lambda x:replace_contractions(x, contractions, contractions_re))

    # Proprecessing
    data_train['text'] = data_train['text'].apply(lambda x:preprocess_sentence(x))
    data_test['text'] = data_test['text'].apply(lambda x:preprocess_sentence(x))

    data_train, duplicate_tweets = remove_duplicates(data_train, tweets_idx)
    print(Fore.RED + f"Duplicate tweets: {duplicate_tweets}")
    print(Fore.GREEN + "-"*100)

    data_train, inconsistent_tweets = remove_inconsistent_labels(data_train, inconsistent_labels)
    print(Fore.RED + f"Duplicate tweets: {inconsistent_tweets}")
    print(Fore.GREEN + "-"*100)
    
    x_data, y_data = data_train['text'], data_train.target.values

    print(Fore.BLUE + "Original sentence: " + data_train['text'][22])
    print(Fore.BLUE + "-"*100)
    print(Fore.BLUE + "Processed sentence: " + x_data[22])
    print(Fore.BLUE + "-"*100)
    print("\n")

    return x_data, y_data, data_test


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-zA-Z '\.,-;:!?#]",' ', sentence)
    sentence = re.sub('\n', '.', sentence)
    return " ".join(sentence.split())

 
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


def replace_contractions(text, contractions, contractions_re):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


def clean_label(y_data):
    negative_labels = [251, 253, 271, 358, 407, 547, 610, 630, 974, 1365, 1221, 2141, 2250, 2279, 2653, 2686, 2692, 2695, 2704, 2812, 2836, 3060, 3133,
                       3243, 3248, 3441, 3913, 3924, 3985, 4023, 4182, 4221, 4232, 4292, 4305, 4318, 4312, 4320, 4323, 4357, 4364, 4378, 4381, 4392, 4403,
                       4414, 4415, 4420, 4453, 4597, 4770, 4917, 5163, 5194, 5470, 5467, 5477, 5497, 5507, 5509, 5574, 5758, 5813, 5823, 5833, 6108, 6091,
                       6119, 6123, 6597, 6618, 6616, 6620, 6632, 6837, 6831, 6840, 7085, 7091, 7365, 7396]
    positive_labels = [248, 584, 591, 606, 1186, 1331, 1232, 2832, 3667, 3687, 4336, 4337, 5330, 5333, 5340, 5342, 5641, 5807, 6240]

    for neg in negative_labels:
        y_data[neg] = 0
    for pos in positive_labels:
        y_data[pos] = 1

    return y_data


def remove_duplicates(data_train, tweets_idx):
    original_vols = data_train.shape[0]

    indexe_dropped = []

    for k in tweets_idx.keys():
        for i in range(0, len(tweets_idx[k])):
            if i > 0:
                indexe_dropped.append(tweets_idx[k][i])

    data_train.drop(index=indexe_dropped, inplace=True)
    current_vols = data_train.shape[0]
    return data_train, original_vols - current_vols


def remove_inconsistent_labels(data_train, inconsistent_labels):
    original_vols = data_train.shape[0]
    indexes_to_drop = data_train[data_train['id'].isin(inconsistent_labels['tweet id'])].index
    data_train.drop(index=indexes_to_drop,inplace=True)
    current_vols = data_train.shape[0]
    return data_train, original_vols - current_vols


if __name__ == "__main__":
    print("Hello!")
