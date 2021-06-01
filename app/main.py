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


@app.get("/")
async def root():
    return {"message": "Hello World"}

def preprocess_text(text):
    # lower text
    text = text.lower()
    # remove tab, new line, ans back slice
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    # remove number
    text = re.sub(r"\d+", "", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove whitespace leading & trailing
    text = text.strip()
    # remove multiple whitespace into single whitespace
    text = re.sub('\s+', ' ', text)
    #remove single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    #NLTK word tokenize
    text = word_tokenize(text)

    list_stopwords = stopwords.words('indonesian')
    txt_stopword = pd.read_csv("preprocessing/stopwords-id.txt", names=["stopwords"], header=None)
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)

    text = [word for word in text if word not in list_stopwords]

    # Normalizing
    normalizad_word = pd.read_csv("preprocessing/normalisasi.csv")

    normalizad_word_dict = {}

    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]

    text = [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in text]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    for term in text:
        if term not in term_dict:
            term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
    
    # Apply stemmed term to dataframe
    text = [term_dict[term] for term in text]

    text = ' '.join(text)
    text = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    text = text.replace(" ( ", " (").replace(" ) ", ") ")
    text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", text)
    text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)
    return text.strip()

# Load 3 Model Beesafe pake transfer learning
komenModel = tf.keras.models.load_model('models/komenModel')
tatapModel =  tf.keras.models.load_model('models/tatapModel')
pegangModel =  tf.keras.models.load_model('models/pegangModel')

class Kalimat(BaseModel):
    kalimat : str

# DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER DENGER
# pip install venv
# init venv pake python3
# pip install -r mod.txt

def response_format(condition, mes, data):
    return {'success': condition, 'message': mes, 'data': data}

@app.post("/predict")
def predict_kalimat(input: Kalimat):
    codex = ""
    kategori = ""
    text = input.kalimat
    text = preprocess_text(text)
    # call model1 dengan method predict
    komenPredict = komenModel.predict(np.expand_dims(text, 0))
    codex = komenPredict
    print(komenPredict)
    # call model2 dengan method predict
    tatapPredict = tatapModel.predict(np.expand_dims(text, 0))
    print(tatapPredict)
    # call model3 dengan method predict
    pegangPredict = pegangModel.predict(np.expand_dims(text, 0))
    print(pegangPredict)
    # condition mana yang paling besar
    if(komenPredict < 0.2 and tatapPredict < 0.2 and pegangPredict < 0.2):
        kategori = "Aman"
    elif(komenPredict > tatapPredict and komenPredict > pegangPredict):
        kategori = "Komentar"
    elif(tatapPredict > komenPredict and tatapPredict > pegangPredict):
        kategori = "Menatap / melihat"
    elif(pegangPredict > komenPredict and pegangPredict > tatapPredict):
        kategori = "Memegang / menyentuh"
    # kembalikan nilai komentar / melihat / memegang yang bergantung dari condition diatas
    return response_format(True, text, kategori)