# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:26:56 2020

@author: cvicentm
"""
#EN ESTE PROGRAMA ES IMPORTANTISIO CAMBIAR EL NOMBE DE VARIABLES Y REORDENARLO, PORQUE MUCHO ESTÁ COPIADO Y PEGADO
from sklearn.decomposition import LatentDirichletAllocation
#import pyLDAvis.sklearn

from create_corpus import create_corpus

import gensim
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


import pandas as pd
from sklearn.cluster import KMeans


from unsupervised_learning_gensim import LDAmodel
from unsupervised_learning_gensim import printColorWordDocument

N_TOPICS =100
#este parámetro no se puede añadir a mano
n_documents =4000
n_printedDocuments =5
max_clusters=200

def validator_cluster(array_topic_per_document, min_cluster=1, max_cluster=n_documents):
	"""This function is going to take the percentaje of the topic of every document, and will validate what number of
	clusters group best similar documents refers to topics
		-min_cluster: minimum number of cluster to group the documents
		-max_cluster: maximum number of cluster to group the documents (this value cant be higher than than number of documents)"""
	print("validating number of clusters...")
	Number_clusters = range(min_cluster, max_cluster)
	#existen muchisimas variables que se puden cambiar, y que probablemente haya que parametrizar, y probablemente validar
	#darle un buen repaso a este tema
	kmeans = [KMeans(n_clusters=i) for i in Number_clusters]
	kmeans
	score = [kmeans[i].fit(array_topic_per_document).score(array_topic_per_document) for i in range(len(kmeans))]
	
	return score

def topic_per_document_pandas(array_topic_per_document, best_n_topic):
	#Este for, es el que quiero que en el futuro te ponga una palabra clave que te describa el documento, 
	#me da igual hacerlo, cogiendo la palabra más usada del tópico con machine learning, por ejemplo un perceptrón: no sé
	title=[]
	for i in range(best_n_topic):
	    title.append('Topic_'+str(i))

	dataframe = pd.DataFrame(array_topic_per_document.T, dtype="str", index=title)

	return dataframe

def knee_locator_k_means(score):
	"""This funtion localize where is the optimal number of clusters"""
	from kneed import KneeLocator

	x = range(1, len(score)+1)
	#son super importantes las variables curve y direction o el KneeLocator no funcionará correctaente
	kn = KneeLocator(x, score, curve='concave', direction='increasing')
	
	return kn.knee

def graphic_k_means_validator(knee,score):
	"""This function print a graph score of all the validators scores"""
	import matplotlib.pyplot as plt

	x = range(1, len(score)+1)
	plt.xlabel('number of clusters k')
	plt.ylabel('Sum of squared distances')
	plt.plot(x, score, 'bx-')
	plt.vlines(knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

def showGraphsLDATrainedInTerminal(dic_subtitles, array_topic_per_document, best_n_topic, n_documents=n_documents):
	"""This function start with the first document an finalize with the las document indicated"""
	for i in range(n_documents):
	    fig, ax = plt.subplots()   
	    ax.plot(np.arange(0,best_n_topic,1),array_topic_per_document[i],'-*',label=list(dic_subtitles.keys())[i])
	    ax.legend()

def showOneGraphLDATrainedInTerminal(name_document, array_topic_per_document, best_n_topic):
    """This function print into a doc document only one subtitles selected by a user"""
    find_slash = name_document.find("\\")
    name_document = name_document[:find_slash]+"\\"+name_document[find_slash+1:]
    number_document = list(dic_subtitles.keys()).index(name_document)

    fig, ax = plt.subplots()   
    ax.plot(np.arange(0,best_n_topic,1),array_topic_per_document[number_document],'-*',label=name_document)
    ax.legend()

def similar_subtitles(dic_subtitles,k_means):
    """This function group all the subtitles titles for cluster group"""
    k_means_label = k_means.labels_
    index_clusters=[]
    list_subtitles=list(dic_subtitles.keys())
    k_means_label = list(k_means_optimized.labels_)
    for i in range(knee):
        index = []
        index = [list_subtitles[document_number] for document_number, n_cluster in enumerate(k_means_label) if n_cluster == i]
        index_clusters.append(index)
    
    return index_clusters

def printClusters2Document(index_clusters):
    """This function put in a document all the subtitles divided on clusters"""
    if not os.path.exists('results'):
        os.makedirs('results')
    file = "results\clusters.txt"
    with open(file, 'w') as f:
        acc = 0
        for cluster in index_clusters:
            f.write("\n")
            f.write("----------------------------------------")
            f.write("N CLUSTER: "+str(acc))
            f.write("----------------------------------------\n")
            acc = acc + 1
            for subtitles in cluster:
                f.write("%s\n" % subtitles)

#PROGRAM-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[array_topic_per_document, best_n_topic, dic_subtitles,lda,generator_normalize,corpus,id2word,coherenceModelArray]=LDAmodel(n_topics=N_TOPICS,n_documents=n_documents, n_printedDocuments=n_printedDocuments)

score = validator_cluster(array_topic_per_document, min_cluster=1, max_cluster=max_clusters)

knee = knee_locator_k_means(score)

graphic_k_means_validator(knee,score)

k_means_optimized = KMeans(n_clusters=knee).fit(array_topic_per_document)

showGraphsLDATrainedInTerminal(dic_subtitles, array_topic_per_document, best_n_topic, n_documents=5)

index_clusters = similar_subtitles(dic_subtitles,k_means_optimized)

printClusters2Document(index_clusters)


#IMPRIMIR DOCUMENTOS DE WORD--------------------------------------------------------------

print("ha llegado a entrenar el modelo")
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

print("ha llegado hasta la parte de los colores")   
for i in range(best_n_topic):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
colors.append('#000000')
#creation of the directory which content all documents printed
if not os.path.exists('word'):
    os.makedirs('word')  
      
print("colour´s documented are being printed")
for i in tqdm(range(n_printedDocuments)):
    printColorWordDocument(i,colors,generator_normalize,dic_subtitles,lda_model,corpus)

print("el mejor tópicoooooooooooo:"+str(best_n_topic))

"""
import pyLDAvis
from pyLDAvis import gensim

print("estamos con el tema de imprimir las gráficas")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, Bow_matrix, vectorizer, mds='tsne')
pyLDAvis.display(panel)

vis_data = gensim.prepare(lda_model, corpus, id2word, sort_topics=False)

print(vis_data.topic_order)
"""