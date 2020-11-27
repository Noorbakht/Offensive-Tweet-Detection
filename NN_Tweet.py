# Import Libaries
# import the necessary files and libraries
# import the necessary libraries
from zeugma.embeddings import EmbeddingTransformer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
import io
import re
import string
import pickle
import nltk
import pkg_resources
import ast
import ssl
from keras.models import model_from_json

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
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

# Setting up Lematizer, Tokenizer
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
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
    #Lemmatization and Tokenization
    row = [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((row))]
    # Stopwords
    all_stopwords = stopwords.words('english')
    all_stopwords.extend(blacklist)
    all_stopwords = set(all_stopwords) - set(whitelist)
    row = [word for word in row if word not in all_stopwords]
    return row


def tokenization(tokenizer, cleaned_tweet):
    cleaned_tweet_lst = []
    cleaned_tweet_lst.append(cleaned_tweet)
    clean_seq = tokenizer.texts_to_sequences(cleaned_tweet_lst)
    clean_seq = pad_sequences(clean_seq, maxlen=40)
    return clean_seq


# loading the tokenizer
with open('tokenizer_a.pickle', 'rb') as handle:
    tokenizer_a = pickle.load(handle)

with open('tokenizer_b.pickle', 'rb') as handle:
    tokenizer_b = pickle.load(handle)

with open('tokenizer_c.pickle', 'rb') as handle:
    tokenizer_c = pickle.load(handle)


def prediction(embedded_tweet_A, embedded_tweet_B, embedded_tweet_C):
    # load json and create model
    json_file = open('modelA.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_A = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model_A.load_weights("modelA.h5")
    res_A = np.argmax(loaded_model_A.predict(embedded_tweet_A), axis=-1)
    # NOT OFFENSIVE
    if(res_A == 0.0):
        return "NOT OFFENSIVE"
    else:
        # load json and create model
        json_file = open('modelB_NN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_B = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model_B.load_weights("modelB_NN.h5")
        res_B = np.argmax(loaded_model_B.predict(embedded_tweet_B), axis=-1)

        # UNT
        if(res_B == 1.0):
            return "OFFENSIVE, UNTARGETED"
        else:
            # TIN
            # load json and create model
            json_file = open('modelC_NN.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model_C = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model_C.load_weights("modelC_NN.h5")
            res_C = np.argmax(loaded_model_C.predict(
                embedded_tweet_C), axis=-1)
            # GRP
            if(res_C == 0.0):
                return "OFFENSIVE, TARGETED, GROUP"
            # IND
            if(res_C == 1.0):
                return "OFFENSIVE, TARGETED, INDIVIDUAL"
            # OTH
            else:
                return "OFFENSIVE, TARGETED, OTHERS"
