# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:24:12 2020

@author: cvicentm
"""
import gensim
from random import randint
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
#start my modules importation ----------------------------
from modules.pre import get_data as g
from modules.classificator import k_means_classificator as kmc
from modules.lda.unsupervised_learning_gensim import LDAmodel
from modules.lda.unsupervised_learning_gensim import printColorWordDocument
#end my modules importation ------------------------------
"""IN THIS CODE WE WILL EXECUTE THE CODE RELATED TO LDA"""

start_topics = 38
N_TOPICS =40

#este parámetro no se puede añadir a mano
n_printedDocuments =2
max_clusters=100

[files, max_documents] = g.get_NameFiles()

#if we want to change the number of documents to analized we can do it here
n_documents=max_documents-4

#PROGRAM-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[array_topic_per_document, best_n_topic, dic_subtitles,lda,generator_normalize,corpus,id2word,coherenceModelArray]=LDAmodel(n_topics=N_TOPICS,n_documents=n_documents, n_printedDocuments=n_printedDocuments, start=start_topics)

score = kmc.validator_cluster(array_topic_per_document, max_cluster=max_clusters, min_cluster=1)

knee = kmc.knee_locator_k_means(score)

kmc.graphic_k_means_validator(knee,score)

k_means_optimized = KMeans(n_clusters=knee).fit(array_topic_per_document)

kmc.showGraphsLDATrainedInTerminal(dic_subtitles, array_topic_per_document, best_n_topic, 5)

index_clusters = kmc.similar_subtitles(dic_subtitles,k_means_optimized,knee,k_means_optimized)

kmc.printClusters2Document(index_clusters,n_documents,dic_subtitles)

#print report about main parameters of the analysis
kmc.printResults2Document(max_documents, n_documents, dic_subtitles, N_TOPICS, best_n_topic, max_clusters, knee, id2word)

topic_dataframe = kmc.topic_per_document_pandas(array_topic_per_document, best_n_topic, dic_subtitles, index_clusters)

kmc.printClusterDf(topic_dataframe, n_documents,index_clusters)

#ORDENAR UN POCO ESTE CÓDIGO
#printing into an excel all the topics of the days

print("printing into excel documents for days")
df = pd.DataFrame({'A' : [np.nan]})
days = kmc.list_days(dic_subtitles)
if not os.path.exists('results\\days\\'+str(n_documents)):
        os.makedirs('results\\days\\'+str(n_documents))
with pd.ExcelWriter('results\\days\\'+str(n_documents)+'\\day_clusters'+str(n_documents)+'.xlsx') as writer:
        df.to_excel(writer, sheet_name="main") 
for day in tqdm(days[0:200]):
    kmc.printDayDf(day, topic_dataframe, n_documents, index_clusters, dic_subtitles,k_means_optimized, knee)


#PRUEBAS---------------------------------------------------------------------------------------------------


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
if not os.path.exists('word\\'+str(n_documents)):
        os.makedirs('word\\'+str(n_documents))
      
print("colour´s documented are being printed")
for i in tqdm(range(n_printedDocuments)):
    printColorWordDocument(i,colors,generator_normalize,dic_subtitles,lda_model,corpus,n_documents)

print("el mejor tópicoooooooooooo:"+str(best_n_topic))