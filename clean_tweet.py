# import the necessary libraries
import ast
from keras.layers import Embedding
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import subprocess
import sys
import pickle

from keras.models import model_from_json
from zeugma.embeddings import EmbeddingTransformer
from nltk.tokenize import TweetTokenizer
from symspellpy.symspellpy import SymSpell
from bs4 import BeautifulSoup
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import preprocessor as p
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
import io
import re
import string
import nltk
import pkg_resources
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')


# list of contractions to remove later
contractions = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


# Setting up SymSpell to segment words
sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Setting up Lematizer, Tokenizer and Stopwords
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))


# Removes specially chosen stop words
def cleaner(row, contractions, blacklist, whitelist):
    '''
    Blacklist: Words that are not in stop words but we will be removing 
    Whitelist: Words that are in the stop words but we will be keeping
    '''
    # remove URLs, Hashtags, Mentions, Reserved Words (RT, FAV), Emojis
    row = p.clean(row)
    # lower the text
    row = row.lower()
    # Apostrophe Handling
    for word in row.split():
        if word in contractions:
            row = row.replace(word, contractions[word.lower()])
    # remove punctuations
    row = "".join([char for char in row if char not in string.punctuation])
    row = re.sub('[0-9]+', '', row, flags=re.MULTILINE)
    # remove numbers
    row = re.sub(r'\d+', '', row, flags=re.MULTILINE)
    # Segment the words
    try:
        row = (sym_spell.word_segmentation(row)).corrected_string
    except:
        pass
    #Lemmatization and Tokenization
    row = [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((row))]
    # Stopwords
    all_stopwords = stopwords.words('english')
    all_stopwords.extend(blacklist)
    all_stopwords = set(all_stopwords) - set(whitelist)
    row = [word for word in row if word not in all_stopwords]
    return row

# vector has to run for training data A, B & C -- three times


def vector(clean_tweet_ls, training_data):
    # Vectorizing the training data
    # Put it inside another list
    clean_tweet = []
    clean_tweet.append(clean_tweet_ls)

    # Parameter indicating the number of words we'll put in the dictionary
    MAX_NUM_WORDS = 13827
    MAX_SEQUENCE_LENGTH = 40  # Maximum number of words in a sequence
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(training_data)

    # Tokenize the clean tweet
    trainX_seq = tokenizer.texts_to_sequences(clean_tweet)
    # testX_seq = tokenizer.texts_to_sequences(testX_df)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    # Padding the sequence
    trainX_seq_trunc = pad_sequences(trainX_seq, maxlen=MAX_SEQUENCE_LENGTH)
    # testX_seq_trunc = pad_sequences(testX_seq, maxlen=MAX_SEQUENCE_LENGTH)

    return trainX_seq_trunc


# takes in vectorised tweet
# embedding has to run for training data A, B & C -- three times
glove = EmbeddingTransformer('glove-twitter-100')


def embedding(vector_tweet):
    listtoStr = ' '.join([str(elem) for elem in vector_tweet])
    print(listtoStr)
    # vector = vector_tweet.apply(', '.join)
    embedded_tweet = glove.transform(listtoStr)
    return embedded_tweet

# STATS MODELS
# def prediction(embedded_tweet_A, embedded_tweet_B, embedded_tweet_C):
#     # linear regression model for A
#     model_A = pickle.load(open('model_A.pkl', 'rb'))
#     res_A = model_A.predict(embedded_tweet_A)
#     # make a prediction
#     # ynew = model.predict_classes(Xnew)
#     # NOT OFFENSIVE
#     if(res_A == 0.0):
#         return "NOT OFF"
#     else:
#         model_B = pickle.load(open('model_B.pkl', 'rb'))
#         res_B = model_B.predict(embedded_tweet_B)
#         # UNT
#         if(res_B == 1.0):
#             return "OFF, UNT"
#         else:
#             # TIN
#             model_C = pickle.load(open('model_C.pkl', 'rb'))
#             res_C = model_C.predict(embedded_tweet_C)
#             # GRP
#             if(res_C == 0.0):
#                 return "OFF, TIN, GRP"
#             # IND
#             if(res_C == 1.0):
#                 return "OFF, TIN, IND"
#             # OTH
#             else:
#                 return "OFT, TIN, OTH"


# NN MODELS
def prediction(embedded_tweet_A, embedded_tweet_B, embedded_tweet_C):
    # load json and create model
    json_file = open('modelA_NN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelA_NN.h5")
    res_A = loaded_model.predict(embedded_tweet_A)
    # NOT OFFENSIVE
    if(res_A == 0.0):
        return "NOT OFF"
    else:
        # load json and create model
        json_file = open('modelB_NN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("modelB_NN.h5")
        res_B = loaded_model.predict(embedded_tweet_B)
        # model_B = pickle.load(open('model_B.pkl', 'rb'))
        # res_B = model_B.predict(embedded_tweet_B)

        # UNT
        if(res_B == 1.0):
            return "OFF, UNT"
        else:
            # TIN

            # load json and create model
            json_file = open('modelC_NN.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("modelC_NN.h5")
            res_C = loaded_model.predict(embedded_tweet_C)
            # model_C = pickle.load(open('model_C.pkl', 'rb'))
            # res_C = model_C.predict(embedded_tweet_C)

            # GRP
            if(res_C == 0.0):
                return "OFF, TIN, GRP"
            # IND
            if(res_C == 1.0):
                return "OFF, TIN, IND"
            # OTH
            else:
                return "OFT, TIN, OTH"


# List of stopwords that want to remove in addition to the ones in stopwords corpus
badboy_list = ['url', 'user', 'ha']
# List of stopwords that we would like to keep
goodboy_list = ['i', 'he', 'she', 'it', 'him',
                'her', 'we', 'you', 'they', 'us', 'them']

clean_tweet = cleaner(
    "@USER She should ask a few native Americans what their take on this is.", contractions, badboy_list, goodboy_list)
print(clean_tweet)


train_df_A = pd.read_csv('clean_training_a.csv')
trainX_df_A = train_df_A['clean_tweet']
trainX_df_A = trainX_df_A.apply(lambda x: ast.literal_eval(x))
# print(trainX_df_A)

train_df_B = pd.read_csv('clean_training_b.csv')
trainX_df_B = train_df_B['clean_tweet']
trainX_df_B = trainX_df_B.apply(lambda x: ast.literal_eval(x))
# print(trainX_df_B)

train_df_C = pd.read_csv('clean_training_c.csv')
trainX_df_C = train_df_C['clean_tweet']
trainX_df_C = trainX_df_C.apply(lambda x: ast.literal_eval(x))
# print(trainX_df_C)

embedded_tweet_A = vector(clean_tweet, trainX_df_A)
print(embedded_tweet_A)

res_A = embedding(clean_tweet)
print(res_A)

embedded_tweet_B = vector(clean_tweet, trainX_df_B)
res_B = embedding(clean_tweet)
print(res_B)

embedded_tweet_C = vector(clean_tweet, trainX_df_C)
res_C = embedding(clean_tweet)
print(res_C)

# val = prediction(embedded_tweet_A, embedded_tweet_B, embedded_tweet_C)
# print(val)
