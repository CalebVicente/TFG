# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:35:53 2020

@author: cvicentm
"""



n_documents = 500

import nltk
from nltk import word_tokenize
import create_corpus as c
import get_data as gt
from tqdm import tqdm

#tic and toc are used to know how many time the process of extaction has taken


dic_subtitles=gt.get_data(n_documents)

#the rows where the value is empty are removed.
print("removing empty dictionary values...")
dic_subtitles = {key:value for (key,value) in tqdm(dic_subtitles.items()) if value != ""}
print("analizing compounds names...")
dic_subtitles = {key:c.compounds_names(value) for (key,value) in tqdm(dic_subtitles.items())}


#this line can cut the dictionary of document to make faster
#dic_subtitles= dict(itertools.islice(dic_subtitles.items(), 0, n_documents))

#list with all the element tokenized
print("\n tokenizing words ...")
list_subt_token = [word_tokenize(value) for (key,value) in tqdm(dic_subtitles.items())]


from gensim.models import Word2Vec

word2vec = Word2Vec(list_subt_token, min_count=2)

vocabulary = word2vec.wv.vocab

index2word = word2vec.wv.index2word

