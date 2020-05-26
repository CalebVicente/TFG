# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:31:14 2020

@author: cvicentm
"""

from sklearn.cluster import KMeans
from tqdm import tqdm


def validator_cluster(array_vector_doc2vec, max_cluster, min_cluster=1):
	"""This function is going to take the percentaje of the topic of every document, and will validate what number of
	clusters group best similar documents refers to topics
		-min_cluster: minimum number of cluster to group the documents
		-max_cluster: maximum number of cluster to group the documents (this value cant be higher than than number of documents)"""
	print("validating number of clusters...")
	Number_clusters = range(min_cluster, max_cluster)
	#existen muchisimas variables que se puden cambiar, y que probablemente haya que parametrizar, y probablemente validar
	#darle un buen repaso a este tema
	kmeans = [KMeans(n_clusters=i) for i in tqdm(Number_clusters)]
	kmeans
	score = [kmeans[i].fit(array_vector_doc2vec).score(array_vector_doc2vec) for i in tqdm(range(len(kmeans)))]
	
	return score

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
    
