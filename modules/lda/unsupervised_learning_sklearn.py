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



N_TOPICS = 40
n_documents = 200
file_lda_model = 'pickle\lda_model_'+str(N_TOPICS)+'_'+str(n_documents)+'.sav'

try:
    
    f=open(file_lda_model, 'rb')
    lda = pickle.load(f)
    print("El modelo ya había sido entrenado previamente...")
    generator_normalize = pickle.load(open("pickle\generator_normalize.txt", "rb"))
    Bow_matrix = pickle.load(open("pickle\Bow_matrix.txt", "rb"))
    vectorizer = pickle.load(open("pickle\Vectorizer.txt", "rb"))
    print("Se ha conseguido importar las variables: generator_normalize, Bow_matrix y vectorizer")
    
except IOError:
    
    try:
        print("HA ENTRADO EN EL SEGUNDO TRY")
        f=open(file_lda_model, 'rb')
        lda = pickle.load(f)
        print("Se debe volver a crear el corpus")
        [generator_normalize, Bow_matrix, vectorizer, vectorizer_first]=create_corpus(n_documents)
        
    except IOError:
        
        print("AL FINAL HAY QUE HACERLO TODO")
        print("El modelo se debe entrenar ...")
        print("Creando el corpus")
        [generator_normalize, Bow_matrix, vectorizer, vectorizer_first]=create_corpus(n_documents)
        print("Proceso de creación del corpus finalizado")
        tic_all_processing=timeit.default_timer()
        lda = LatentDirichletAllocation(n_components=N_TOPICS,max_iter=500,learning_method='batch',batch_size=50)
        print("Se comienza a entrenar el corpus")
        lda.fit(Bow_matrix)
        toc_all_processing=timeit.default_timer()
        time_lda_fit=str(datetime.timedelta(seconds=int(float(toc_all_processing-tic_all_processing))))
        print("The process of training lda model with "+str(N_TOPICS)+" topics has taken "+time_lda_fit+" seconds")    
        print("Finalización del entrenamiento del corpus completada")
        topics_per_document=lda.components_
        
        print("se va a proceder a guardar el modelo en un fichero")
        # if the number of subtitles doest change, we can use the same model than the last time
        pickle.dump(lda, open(file_lda_model, 'wb'))
     

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

for i in range(400):
    new_name=news_list[i]
    topic_distribution = lda.transform(Bow_matrix[i])
    numpy_distribution=np.asarray(topic_distribution)
    numpy_distribution=np.resize(numpy_distribution,(n_topics,))
    fig, ax = plt.subplots()   # Declares a figure handle
    ax.plot(np.arange(0,n_topics,1),numpy_distribution,'-*',label=new_name)
    ax.legend()
"""
