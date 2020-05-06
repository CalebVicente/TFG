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
import matplotlib.pyplot as plt
import numpy as np



def printColorWordDocument(number,colors,generator_normalize,dic_subtitles,lda_model,corpus,n_documents,n_printedDocuments):
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
    
    
    print(colors)
    [wordToDocument(word,colors[topic]) for word,topic in document_classified]  
    word_subtitles_colors="word"+list(dic_subtitles.keys())[number][9:-1]+str(n_documents)+str(n_printedDocuments)+".docx"
    document.save(word_subtitles_colors)
    

def training_model(n_documents,N_TOPICS,id2word, corpus):
    #ESCRIBIR QUE HACE ESTA FUNCIÓN Y PARA QUE SIRVE CADA UNO DE SUS PARÁMETROS
    
    
    print("the model is being trained with: "+str(N_TOPICS)+ "topics")
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=N_TOPICS, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    
    #tengo bastantes dudas si en el parámetro texts tengo que poner generator_normalize o no
    coherencemodel = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
    coherence_values = coherencemodel.get_coherence()
    #the model is gonna be saved
    # if the number of subtitles doest change, we can use the same model than the last time
    
    file_lda_model = 'pickle\lda_model_'+str(N_TOPICS)+'_'+str(n_documents)+'.sav'
    pickle.dump(lda_model, open(file_lda_model, 'wb'))
    
    return coherence_values

#PROGRAM......................................................................
def LDAmodel( n_topics, n_documents, n_printedDocuments, step=1, start=1):   
    coherencemodelArray=[]
    #Tengo que escribir para que sirve cada cosa que hace el gensim
    print(" the corpus with: "+str(n_topics)+" is being created")
    [generator_normalize, dic_subtitles]=create_corpus(n_documents)
    
    #this is creating a dictionary with all de different words of the document
    id2word = corpora.Dictionary(generator_normalize)
    
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in generator_normalize]
    

    for n_topics in range(start, n_topics, step):
        #function based on : https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#13viewthetopicsinldamodel
        coherencemodelArray.append(training_model(n_documents,n_topics,id2word,corpus))
    
    
    x = range(start, n_topics+1, step)
    #n_topics+1 because has to have the same weight than coherencemodelArray
    plt.plot(x, coherencemodelArray)
    plt.xlabel("N_Topics")
    plt.ylabel("Coherence")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    best_n_topic=coherencemodelArray.index(min(coherencemodelArray))+start
    print("el mejor modelo es: "+'pickle\lda_model_'+str(best_n_topic)+'_'+str(n_documents)+'.sav')
    f=open('pickle\lda_model_'+str(best_n_topic)+'_'+str(n_documents)+'.sav', 'rb')
    lda = pickle.load(f)
    document_per_topic=lda.get_document_topics(corpus)
    
    corp_cur = corpus[1]
    topic_percs, wordid_topics, wordid_phivalues = lda[corp_cur]
    print(wordid_topics)
    
    array_topic_per_document = np.zeros((n_documents, best_n_topic))
    
    
    for i in range(n_documents):
        for j in range(len(document_per_topic[i])):
            array_topic_per_document[i][document_per_topic[i][j][0]]= document_per_topic[i][j][1]
    
    #NUMBER OF DOCUMENTs to print results on word
    
    
    return array_topic_per_document, best_n_topic, dic_subtitles,lda,generator_normalize,corpus,id2word
"""
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
"""

""""
pip install pyldavis==2.1.0
print("estamos con el tema de imprimir las gráficas")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, Bow_matrix, vectorizer, mds='tsne')
pyLDAvis.display(panel)
"""


    
#Sería interesante hacer clustering con la matriz array_topic_per_document, consiguiendo saber que documentos se parecen y cuales no

