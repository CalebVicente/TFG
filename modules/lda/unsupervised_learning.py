# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 07:46:39 2020

@author: cvicentm
"""
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
import pickle
from create_corpus import create_corpus

file_lda_model = 'pickle\lda_model.sav'

   
n_topics = 25

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
        [generator_normalize, Bow_matrix, vectorizer]=create_corpus()
        
    except IOError:
        print("AL FINAL HAY QUE HACERLO TODO")
        print("El modelo se debe entrenar ...")
        print("Creando el corpus")
        [generator_normalize, Bow_matrix, vectorizer]=create_corpus()
        print("Proceso de creación del corpus finalizado")
        
        lda = LatentDirichletAllocation(n_components=n_topics,max_iter=500,learning_method='batch',batch_size=50)
        print("Se comienza a entrenar el corpus")
        lda.fit(Bow_matrix)
        print("Finalización del entrenamiento del corpus completada")
        topics_per_document=lda.components_
        
        print("se va a proceder a guardar el modelo en un fichero")
        # if the number of subtitles doest change, we can use the same model than the last time
        pickle.dump(lda, open(file_lda_model, 'wb'))
     


print("estamos con el tema de imprimir las gráficas")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, Bow_matrix, vectorizer, mds='tsne')
pyLDAvis.display(panel)


import matplotlib.pyplot as plt
import numpy as np

for i in range(50):
    topic_distribution = lda.transform(Bow_matrix[i])
    numpy_distribution=np.asarray(topic_distribution)
    numpy_distribution=np.resize(numpy_distribution,(n_topics,))
    print(np.size(numpy_distribution))
    fig, ax = plt.subplots()   # Declares a figure handle
    ax.plot(np.arange(0,n_topics,1),numpy_distribution,'-*',label='mean topic proportion')
    ax.legend()