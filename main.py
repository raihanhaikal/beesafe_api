from fastapi import FastAPI
import re
import string
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = FastAPI()

df_dataset = pd.read_csv('dataset/train_indo.csv')


def remove_special_char(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")


# remove number
def remove_number(text):
    return re.sub(r"\d+", "", text)


# remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()


# remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)


# remove single char
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


# NLTK word rokenize
def word_tokenize_wrapper(text):
    return word_tokenize(text)


df_dataset['deskripsi'] = df_dataset['deskripsi'].str.lower()
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_special_char)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_number)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_punctuation)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_whitespace_LT)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_whitespace_multiple)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(remove_single_char)
df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(word_tokenize_wrapper)

# stopwords_removal
list_stopwords = stopwords.words('indonesian')
txt_stopword = pd.read_csv("preprocessing/stopwords-id.txt", names=["stopwords"], header=None)
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
list_stopwords = set(list_stopwords)


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(stopwords_removal)

# Normalizing
normalizad_word = pd.read_csv("preprocessing/normalisasi.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]


def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(normalized_term)

# Stemming
# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# Stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in df_dataset['deskripsi']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)


# Apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]


df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(get_stemmed_term)


# Detokenize
def untokenize(words):
    text = ' '.join(words)
    text = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    text = text.replace(" ( ", " (").replace(" ) ", ") ")
    text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", text)
    text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)
    return text.strip()


df_dataset['deskripsi'] = df_dataset['deskripsi'].apply(untokenize)

# Load 3 Model Beesafe pake transfer learning

class Kalimat(BaseModel):
    kalimat_kasar : str

# DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER
# pip install venv
# init venv pake python3
# pip install -r mod.txt

@app.post("/predict")
def predict_kalimat(kata_kasar: Kalimat):
    # call model1 dengan method predict
    # call model2 dengan method predict
    # call model3 dengan method predict
    # condition mana yang paling besar
    # kembalikan nilai komentar / melihat / memegang yang bergantung dari condition diatas
    return kata_kasar
