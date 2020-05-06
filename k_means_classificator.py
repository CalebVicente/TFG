# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:26:56 2020

@author: cvicentm
"""
#EN ESTE PROGRAMA ES IMPORTANTISIO CAMBIAR EL NOMBE DE VARIABLES Y REORDENARLO, PORQUE MUCHO ESTÁ COPIADO Y PEGADO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from random import randint
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from unsupervised_learning_gensim import printColorWordDocument
from unsupervised_learning_gensim import LDAmodel
import gensim

N_TOPICS = 5
#este parámetro no se puede añadir a mano
n_documents =200
n_printedDocuments =5
[array_topic_per_document, best_n_topic, dic_subtitles,lda,generator_normalize,corpus,id2word]=LDAmodel(n_topics=N_TOPICS,n_documents=n_documents, n_printedDocuments=n_printedDocuments)

n_printedDocuments = 4
for i in range(n_documents):
    fig, ax = plt.subplots()   # Declares a figure handle
    ax.plot(np.arange(0,best_n_topic,1),array_topic_per_document[i],'-*',label=list(dic_subtitles.keys())[i])
    ax.legend()

#Este for, es el que quiero que en el futuro te ponga una palabra clave que te describa el documento, 
#me da igual hacerlo, cogiendo la palabra más usada del tópico con machine learning, por ejemplo un perceptrón: no sé
title=[]
for i in range(best_n_topic):
    title.append('Topic_'+str(i))

dataframe = pd.DataFrame(array_topic_per_document.T, dtype="str", index=title)


print("validating number of clusters...")
Number_clusters = range(1, n_documents)
#existen muchisimas variables que se puden cambiar, y que probablemente haya que parametrizar, y probablemente validar
#darle un buen repaso a este tema
kmeans = [KMeans(n_clusters=i) for i in Number_clusters]
kmeans
score = [kmeans[i].fit(array_topic_per_document).score(array_topic_per_document) for i in range(len(kmeans))]
score
#What k-means essentially does is find cluster centers that minimize the sum of distances between data samples
plt.plot(Number_clusters,score)
plt.xlabel('n_clusters')
plt.ylabel('Score')
plt.title('K-MEANS')
plt.show()

x = range(1, len(score)+1)
from kneed import KneeLocator
#son super importantes las variables curve y direction o el KneeLocator no funcionará correctaente
kn = KneeLocator(x, score, curve='concave', direction='increasing')
print(kn.knee)


import matplotlib.pyplot as plt
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(x, score, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

k_means_optimized = KMeans(n_clusters=kn.knee).fit(array_topic_per_document)

#IMPRIMIR DOCUMENTOS DE WORD--------------------------------------------------------------
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=best_n_topic, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

colors = []

    
for i in range(best_n_topic):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
    colors.append('#000000')
    #creation of the directory which content all documents printed
    if not os.path.exists('word'):
        os.makedirs('word')  
        
print("colour´s documented are being printed")
for i in tqdm(range(n_printedDocuments)):
    printColorWordDocument(i,colors,generator_normalize,dic_subtitles,lda_model,corpus,n_documents,n_printedDocuments)

