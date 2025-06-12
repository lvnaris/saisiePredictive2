import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('punkt_tab')

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df.drop(columns=['Unnamed: 3'])
    return df

def preprocess(messages):
    tokenized_sentences = []
    for sms in messages:
        sms = re.sub(r"<[A-Z]{3}_\d+>","", sms) # supprimer tag d'anonymat
        #sms = re.sub(r"\d+", "", sms) # supprimer chiffres
        sms = sms.lower() # convertir en minuscules
        sms = re.sub(r"(?<![a-zA-Z])-|-(?![a-zA-Z])", "", sms) # supprimer tirets sauf si entre deux lettres (avant-hier)
        sms = re.sub(r"[^\w\s\-]", ' ', sms) # remplacer ponctuation avec une espace à l'exception de tirets
        sms = re.sub(r"\s+", " ", sms).strip() # supprimer espaces excessifs
        words = word_tokenize(sms) # usage du tokeniseur nltk
        tokenized_sentences.append(words)
    return tokenized_sentences

