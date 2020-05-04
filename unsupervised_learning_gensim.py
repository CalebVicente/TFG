# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:02:29 2020

@author: cvicentm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 07:46:39 2020

@author: cvicentm
"""
from sklearn.decomposition import LatentDirichletAllocation
#import pyLDAvis.sklearn
import pickle
from create_corpus import create_corpus
import timeit
import datetime

import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
from PIL import ImageColor
from create_corpus import normalize_word
from docx import Document
from docx.shared import RGBColor
from nltk import word_tokenize
from random import randint
import os
from tqdm import tqdm

N_TOPICS = 40
#este parámetro no se puede añadir a mano
n_documents = 4763
n_printedDocuments = 50
file_lda_model = 'pickle\lda_model_'+str(N_TOPICS)+'_'+str(n_documents)+'.sav'


def printColorWordDocument(number,colors,generator_normalize,dic_subtitles,lda_model,corpus):
    #make an explanation of every of the parameters of this function
    corp_cur = corpus[number] 
    topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
    word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics if topic]
                          
    
    def wordToDocument (word, color_hex):
        color_rgb = ImageColor.getrgb(color_hex)
        run=paragraph.add_run(word+" ")
        font = run.font
        font.color.rgb = RGBColor(color_rgb[0], color_rgb[1], color_rgb[2])
    
    document_classified=[]
    word_dominanttopic_dict=dict(word_dominanttopic)
    dict_one_subtitle_token=word_tokenize(dic_subtitles[list(dic_subtitles.keys())[number]])
    for word in dict_one_subtitle_token:
        topic_word=len(colors)-1
        if normalize_word(word) in generator_normalize[number]:
            try:
                topic_word=word_dominanttopic_dict[normalize_word(word)]
            except KeyError as error:
                # Escribir aquí un log en vez de un print
                print("key error")
                
        else:
            topic_word=len(colors)-1
        document_classified.append((word,topic_word))
    
    
    #document_classified=[(word,word_dominanttopic_dict[word]) for word in generator_normalize[0]]
    document = Document()
    paragraph = document.add_paragraph()
    [wordToDocument(word,colors[topic]) for word,topic in document_classified]
    
    word_subtitles_colors="word"+list(dic_subtitles.keys())[number][9:-1]+".docx"
    document.save(word_subtitles_colors)

#Tengo que escribir para que sirve cada cosa que hace el gensim
print("the corpus is being created")
[generator_normalize, dic_subtitles]=create_corpus(n_documents)

id2word = corpora.Dictionary(generator_normalize)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in generator_normalize]

print("the model is being trained")
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=N_TOPICS, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)   

print("se va a proceder a guardar el modelo en un fichero")
# if the number of subtitles doest change, we can use the same model than the last time
pickle.dump(lda_model, open(file_lda_model, 'wb'))
#creation of the matrix of colors to print documents
colors = []
    
for i in range(N_TOPICS):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
colors.append('#000000')
#creation of the directory which content all documents printed
if not os.path.exists('word'):
    os.makedirs('word')  

print("colour´s documented are being printed")
for i in tqdm(range(n_printedDocuments)):
    printColorWordDocument(i,colors,generator_normalize,dic_subtitles,lda_model,corpus)


""""
print("estamos con el tema de imprimir las gráficas")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, Bow_matrix, vectorizer, mds='tsne')
pyLDAvis.display(panel)
"""
"""
import matplotlib.pyplot as plt
import numpy as np

dic_preprocesing_subtitles = pickle.load(open("pickle\dict_preprocesing_subtitles.txt", "rb"))
news_list = list(dic_preprocesing_subtitles)

for i in range(50):
    new_name=news_list[i]
    topic_distribution = lda.transform(Bow_matrix[i])
    numpy_distribution=np.asarray(topic_distribution)
    numpy_distribution=np.resize(numpy_distribution,(n_topics,))
    fig, ax = plt.subplots()   # Declares a figure handle
    ax.plot(np.arange(0,n_topics,1),numpy_distribution,'-*',label=new_name)
    ax.legend()
"""
