import itertools
import pandas as pd
import json
import requests
import re
from farasa.stemmer import FarasaStemmer
import pyarabic.araby as araby
from pyarabic.araby import tokenize, is_arabicrange, strip_tashkeel

# Removing mentions
def remove_mentions(text):
    return re.sub("@[A-Za-z0-9_]+","", text)

# Removing links
def remove_links(text):
    return re.sub(r"http\S+", "", text)
    return(text)

# Normalizing Arabic letters
def normalizeArabic(text):
    text = text.replace("أ", "ا" )
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا" )
    text = text.replace("ى", "ي")
    text = text.replace("ة","ه")
    text = text.replace('اً', 'ا')
    return(text)

# Removing hashtags
def remove_hashtags(text):
    #return " ".join(filter(lambda text:text[0]!='#', text.split()))
    st = '# _'
    for i, letter in enumerate(text):
        if letter in st:
            text = text.replace(letter," ")
    return text

# Remove El-thshkeel and emoji, numbers and punctuations
def remove_el_tashkeel_and_other_non_letters(text):
    text = tokenize(text, conditions=is_arabicrange, morphs=strip_tashkeel)
    
    return ' '.join(text)

# Remove repeated letters
def remove_repeated_letters(text):
    # words like ههههههه  will be ه
    # words like جدااااا will be جدا
    return ''.join(c for c, _ in itertools.groupby(text))

# Lemmatization
stemmer_interactive = FarasaStemmer(interactive=True)
def lemmatization(text):
    stemmed_interactive = stemmer_interactive.stem(text)
    return stemmed_interactive

# proccess
def preprocess(text):
    text = remove_mentions(text)
    text = remove_links(text)
    text = normalizeArabic(text)
    text = remove_hashtags(text)
    text = remove_el_tashkeel_and_other_non_letters(text)
    text = remove_repeated_letters(text)
    text = lemmatization(text)
    return text
# read the data from phase 1
dataset = pd.read_csv('dataset_with_tweets.csv', dtype=str)

dataset['pure_tweet'] = dataset['tweet'].apply(lambda x: preprocess(x))
# save data for phase 2
dataset.to_csv('preprocessed_data.csv', index=False)
stemmer_interactive.terminate()


