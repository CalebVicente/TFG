# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:36:22 2020

@author: cvicentm
"""
from unsupervised_learning import N_TOPICS
import pyLDAvis.sklearn
print("estamos con el tema de imprimir las gráficas")
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, Bow_matrix, vectorizer, mds='tsne')
pyLDAvis.display(panel)