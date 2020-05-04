# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:20 2020

@author: cvicentm
"""
""" 
In this module the first preprocessing will be done. 
Corpus, and differents methods, most of all
implemented with the python library NLTK.

"""

import string
import nltk
import pickle

nltk.download('stopwords')
nltk.download('wordnet')


import get_data
from nltk import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer
#Lemmatizer only works in english, so i have found this library to work in other languages:
#parsetree give more information about the word like what kind of word it is, singular and plurar,etc
from pattern.es import parsetree
from pattern.es import Text
#lemma gives only the lemmatization of words
from pattern.es import lemma
#libreria de parsetree en: https://www.clips.uantwerpen.be/pages/pattern-en#tree
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import timeit
import itertools



def normalize(text):
    
    ## Module constants
    snowball = SnowballStemmer('spanish')
    stopwords   = set(nltk.corpus.stopwords.words('spanish'))
    punctuation = string.punctuation
    
    for token in text:
        token = token.lower() 
        #this is applying english lemmatization, so we have to prove with patter
        #the thing is that as far as i know patter only works in python 2
        token = lemma(token)
        token = snowball.stem(token)
        if token not in stopwords and token not in punctuation and len(token)>3:
            #we do it with yield to create a generator, too much faster, and when less memory problems than a list
            yield token
            
def create_corpus(n_documents):
    #tic and toc are used to know how many time the process of extaction has taken
    tic=timeit.default_timer()
    
    dic_subtitles=get_data.get_data(n_documents)
    
    #the rows where the value is empty are removed.
    dic_subtitles = {key:value for (key,value) in dic_subtitles.items() if value != ""}
    #this line can cut the dictionary of document to make faster
    dic_subtitles= dict(itertools.islice(dic_subtitles.items(), 0, n_documents))
    
    #list with all the element tokenized
    list_subt_token = [word_tokenize(value) for (key,value) in dic_subtitles.items()]
    
    generator_normalize = [list(normalize(document)) for document in list_subt_token]
    
    vectorizer_first = CountVectorizer()
    
    vectorizer_first.fit_transform([' '.join(doc) for doc in generator_normalize])
    
    words=list(vectorizer_first.vocabulary_.keys())
    vectorizer = CountVectorizer(vocabulary=words)
    
    Bow_matrix = vectorizer.fit_transform([' '.join(doc) for doc in generator_normalize])
    #now, a dictionary of the preprocesing subtitles who form the corpus will be saved in a file thanks to pickle library
    with open('pickle\dict_preprocesing_subtitles.txt', 'wb') as filename:
        pickle.dump(dic_subtitles, filename)
    with open('pickle\generator_normalize.txt', 'wb') as filename:
        pickle.dump(generator_normalize, filename)
    with open('pickle\Bow_matrix.txt', 'wb') as filename:
        pickle.dump(Bow_matrix, filename)
    with open('pickle\Vectorizer.txt', 'wb') as filename:
        pickle.dump(vectorizer, filename)
    """with open('pickle\Words.txt', 'wb') as filename:
        pickle.dump(words, filename)
       """ 
    #tic and toc are used to know how many time the process of extaction has taken
    toc=timeit.default_timer()
    print("Creation of the corpus has taken: "+str(toc-tic)+" seconds")
    return generator_normalize, Bow_matrix, vectorizer, vectorizer_first




