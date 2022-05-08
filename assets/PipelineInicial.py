import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

import re, unicodedata
import contractions
import inflect
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import word_tokenize


class LimpiezaTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def limpiar_label(self,value):
        if value=='__label__0':
            return 0
        else:
            return 1
    
    def limpiar_study(self,value):
        return value.replace('study interventions are ', '')
    
    def remove_non_ascii(self,words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self,words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words =[]
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self,words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        stopwords = nltk.corpus.stopwords.words('english')
        for word in words:
            if word not in stopwords:
                new_words.append(word)
        return new_words

    def preprocessing(self,words):
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words
        
    def transform(self, X, y=None):
        X_ = X.copy()
        X_add = X_["study_and_condition"].str.split('.', 1, expand=True)
        X_add.columns = ['study', 'condition']
        X_ = pd.concat([X_, X_add], axis=1)
        X_.drop('study_and_condition', axis=1, inplace=True)
        X_['label'] = X_['label'].map(self.limpiar_label)
        X_['study'] = X_['study'].map(self.limpiar_study)
        X_['condition'] = X_['condition'].apply(contractions.fix) #Aplica la corrección de las contracciones
        X_['words'] = X_['condition'].apply(word_tokenize).apply(self.preprocessing) #Aplica la eliminación del ruido
        return X_

class NormalizacionTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def stem_words(self,words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self,words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def stem_and_lemmatize(self,words):
        stems = self.stem_words(words)
        lemmas = self.lemmatize_verbs(words)
        return stems + lemmas

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['words'] = X_['words'].apply(self.stem_and_lemmatize) #Aplica lematización y Eliminación de Prefijos y Sufijos.
        X_['words'] = X_['words'].apply(lambda x: ' '.join(map(str, x)))
        return X_