# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:03:26 2020

@author: cvicentm
"""

import gensim
import nltk
from nltk import word_tokenize
from tqdm import tqdm
from norm_title_sub import norm_title_sub
import numpy as np

# Download dataset
#------------------------------------------------------------------------------------------
dic_subtitles=norm_title_sub()

print("\n tokenizing words ...")
data = [word_tokenize(value) for (key,value) in tqdm(dic_subtitles.items())]

subtitles=list(dic_subtitles.keys())
#------------------------------------------------------------------------------------------

# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [subtitles[i]])

train_data = list(create_tagged_document(data))

print("starting with doc2vec....")
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

list_vec_doc2vec = [model.docvecs[subtitle] for subtitle in subtitles]

arr_vec_doc2vec = np.stack( list_vec_doc2vec, axis=0 )

#K_MEANS SIMILARITY: -------------------------------------------------------------------------
import k_means_doc2vec as km_d2v

max_clusters = 400

score = km_d2v.validator_cluster(arr_vec_doc2vec, max_clusters ,min_cluster=1)

knee = km_d2v.knee_locator_k_means(score)

km_d2v.graphic_k_means_validator(knee,score)

#k_means_optimized = KMeans(n_clusters=knee).fit(arr_vec_doc2vec)
