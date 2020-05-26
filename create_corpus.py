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
#from pattern.es import Text
#lemma gives only the lemmatization of words
from pattern.es import lemma
#libreria de parsetree en: https://www.clips.uantwerpen.be/pages/pattern-en#tree
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import timeit
import itertools





#VER COMO PONER ESTO DE FORMA MÁS ELEGANTE
stopwords   = set(nltk.corpus.stopwords.words('spanish'))
stopwords.add("unir")
snowball = SnowballStemmer('spanish')
punctuation = string.punctuation



def compounds_names(text):
    """this function put '-' between compounds names.
        For example: Estados Unidos --> Estados-Unidos"""
        
    from nltk import Text
    import logging, logging.handlers
    from datetime import datetime
    
    #LOGS-------------------------------------------------------------------------------------
    name_log_file = datetime.now().strftime('logs\load_subtitles_%d_%m_%Y.log')
    
    logging.basicConfig(filename=name_log_file, level=logging.WARNING, 
                        format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(message)s")
    
    #-----------------------------------------------------------------------------------------
        
    text_token=word_tokenize(text)
    try:
        for i in range(len(text_token)):
            if text_token[i][0].isupper() and text_token[i+1][0].isupper() and (text_token[i].lower() not in stopwords and text_token[i+1].lower() not in stopwords):
                word=str(text_token[i])+str(" ")+str(text_token[i+1])
                replace_word=str(text_token[i])+"-"+str(text_token[i+1])
                text=text.replace(word,replace_word)
                
                """text_token.insert(i,word)
                text_token.pop(i+1)
                text_token.pop(i+1)"""
    except IndexError:
        logging.warning("size of news has been changed becouse of compounds words \n")

    return text
    
def normalize(text):
 
    def lemmAndStem(word):
            word = word.lower() 
            if word not in stopwords and word not in punctuation and word != "0000000000000000000000000000000000000000" and len(word)>3:
                #we do it with yield to create a generator, too much faster, and when less memory problems than a list
                #this is applying english lemmatization, so we have to prove with patter
                #the thing is that as far as i know patter only works in python 2
                word = lemma(word)
                word = snowball.stem(word)
                if word not in stopwords and word not in punctuation and len(word)>2:
                    return word
    
    for token in text:
        ## Module constants     
        #if all the characters are uppercase then they are not stimmed nor are lemmatized, opposite they will be
        if token.lower() not in stopwords:
            if token.isupper():
                word = token
                yield word
            else:
                if lemmAndStem(token):
                    word=lemmAndStem(token)       
                    yield word
                    
def normalize_word(token):
    
    ## Module constants
    def lemmAndStem(word):
            word = word.lower() 
            if word not in stopwords and word not in punctuation and word != "0000000000000000000000000000000000000000" and len(word)>3:
                #we do it with yield to create a generator, too much faster, and when less memory problems than a list
                #this is applying english lemmatization, so we have to prove with patter
                #the thing is that as far as i know patter only works in python 2
                word = lemma(word)
                word = snowball.stem(word)
                if word not in stopwords and word not in punctuation and len(word)>2:
                    return word
    
    ## Module constants     
    #if all the characters are uppercase then they are not stimmed nor are lemmatized, opposite they will be       
    if token.lower() not in stopwords:
        if token.isupper():
            word = token
            return word
        else:
            if lemmAndStem(token):
                word=lemmAndStem(token)       
                return word
            
def create_dic_subtitles(n_documents):
    from tqdm import tqdm
    
    dic_subtitles=get_data.get_data(n_documents)
    
    #the rows where the value is empty are removed.
    print("removing empty dictionary values...")
    dic_subtitles = {key:value for (key,value) in tqdm(dic_subtitles.items()) if value != ""}
    print("analizing compounds names...")
    dic_subtitles = {key:compounds_names(value) for (key,value) in tqdm(dic_subtitles.items())}
    
    return dic_subtitles

#REMOVE REPITED SUBTITLES:--------------------------------------------------------------------------
import variables as v
    
years = v.YEARS
channels = v.CHANNELS
hour_new = v.NEWS

def normalize_title_subtitles(subtitle):
    
    def get_day(text,year):
        """this function returns the date from a subtitle title given the year"""
        n_day=text.find(year)
        day=""
        if n_day!=-1:
            day=text[n_day:(n_day+10)]
        
        return day

    channel = [channel for channel in channels if subtitle.find(channel)!=-1]
    
    day=[get_day(subtitle,year) for year in years if get_day(subtitle,year) != ""]
    
    hour = [hour for hour in hour_new if subtitle.find(hour)!=-1]
    
    new_subtitle = str(channel[0])+"_"+str(day[0])+"_"+str(hour[0])
    
    return new_subtitle

#------------------------------------------------------------------------------------------

def norm_title_sub(dic_subtitles):
    import logging, logging.handlers
    from datetime import datetime
    from tqdm import tqdm
    #CAMBIAR LA RUTA DEL LOG
    name_log_file = datetime.now().strftime('load_subtitles_%d_%m_%Y.log')
    logging.basicConfig(filename=name_log_file, level=logging.WARNING, 
                        format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(message)s")
    
    norm_dict_subt = {}
        
    for (o_key,value) in tqdm(dic_subtitles.items()):
        
        
        n_key = normalize_title_subtitles(o_key)
        
        if n_key not in norm_dict_subt:
        
            norm_dict_subt[n_key] = dic_subtitles[o_key]
        
        else:
            logging.warning("la clave "+str(n_key)+" esta repetida")
            
    return norm_dict_subt

#--------------------------------------------------------------------------------------------------


def create_corpus(n_documents):
    from tqdm import tqdm
    #tic and toc are used to know how many time the process of extaction has taken
    tic=timeit.default_timer()
    
    dic_subtitles=get_data.get_data(n_documents)
    
    #the rows where the value is empty are removed.
    print("removing empty dictionary values...")
    dic_subtitles = {key:value for (key,value) in tqdm(dic_subtitles.items()) if value != ""}
    print("removing repited subtitles...")
    dic_subtitles = norm_title_sub(dic_subtitles)
    print("analizing compounds names...")
    dic_subtitles = {key:compounds_names(value) for (key,value) in tqdm(dic_subtitles.items())}
    
    
    #this line can cut the dictionary of document to make faster
    #dic_subtitles= dict(itertools.islice(dic_subtitles.items(), 0, n_documents))
    
    #list with all the element tokenized
    print("\n tokenizing words ...")
    list_subt_token = [word_tokenize(value) for (key,value) in tqdm(dic_subtitles.items())]
    print("\n Creation of generator normalize")
    generator_normalize = [list(normalize(document)) for document in tqdm(list_subt_token)]
    generator_normalize = list(generator_normalize)
    """
    #this code is only necesary when we are using sklearn
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
    """
    #tic and toc are used to know how many time the process of extaction has taken
    toc=timeit.default_timer()
    print("Creation of the corpus has taken: "+str(toc-tic)+" seconds")
    """
    return generator_normalize, Bow_matrix, vectorizer, vectorizer_first, dic_subtitles
    """
    return generator_normalize, dic_subtitles


