# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:31:14 2020

@author: cvicentm
"""

from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('config\\config.ini')


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
    
def get_EuclDistNorm(k_means):
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    #creation of a euclidian distances matrix, [n_clustersXn_clusters] normalized
    cluster_centers = k_means.cluster_centers_
    ed = euclidean_distances(cluster_centers,cluster_centers)
    ed_norm = ed/ed.max()
    
    return ed_norm

def get_EuclDistFromDay(df_day, ed_norm, n_clusters):
    """this function get the euclidan distance dataframe with the topic of th day"""
    
    #get differents clusters in a day
    clusters_day = dict(df_day['clusters']).values()
    np_clusters_day=np.array(list(clusters_day)).astype(int)
    np_clusters_day=np.unique(np_clusters_day)
    
    #creation of the euclidian distances matrix with the topics of the day
    ed_day=[ed_norm[i][j] for i in np_clusters_day for j in np_clusters_day]
    np_ed_day=np.array(ed_day).reshape((np.size(np_clusters_day),np.size(np_clusters_day)))
    
    ed_columns=[i for i in np_clusters_day]
    df_ed_day = pd.DataFrame(np_ed_day, dtype="str", index=ed_columns)
    df_ed_day.columns=ed_columns
    
    return df_ed_day

def printDayDf(day, dataframe, n_documents, index_clusters, subtitles,k_means,n_clusters):
    """Función que imprime en un excel los documentos pertenecientes a un dia, el cluster al que pertenecen y sus datos"""

    
    subtitles_day = [subtitle for subtitle in subtitles if day in subtitle]
    #topic for every document during a day
    df_day = dataframe[subtitles_day].T.astype(float).round(3)  
    
    #euclidian distance normalize for all the clusters centers
    ed_norm = get_EuclDistNorm(k_means)
    #euclidian distance for only the clusters of the concret day
    df_ed_day= get_EuclDistFromDay(df_day, ed_norm, n_clusters)
    
    #--------------------------------------------------------------------------------------------------
    from openpyxl import load_workbook
    path_days = config['DOC2VEC']['path_results_days']
    
    book = load_workbook(path_days+str(n_documents)+'\\day_clusters'+str(n_documents)+'.xlsx')
    writer = pd.ExcelWriter(path_days+str(n_documents)+'\\day_clusters'+str(n_documents)+'.xlsx', engine='openpyxl') 
    writer.book = book
    
    ## ExcelWriter for some reason uses writer.sheets to access the sheet.
    ## If you leave it empty it will not know that sheet Main is already there
    ## and will create a new sheet.
    
    dataframe[subtitles_day].T.astype(float).round(3).to_excel(writer, sheet_name=day)  
    df_ed_day.astype(float).round(3).to_excel(writer,sheet_name=day, startrow=len(df_day.index)+5)
    
    
    writer.save()
    
def get_day(text,year):
    """this function returns the date from a subtitle title given the year"""
    n_day=text.find(year)
    day=""
    if n_day!=-1:
        day=text[n_day:(n_day+10)]
    
    return day
   
def list_days(subtitles):   
    """this functions returns the list of diferents days used in the corpus"""
    import modules.variables as v
    
    years = v.YEARS
    
    days=[get_day(text,year) for text in subtitles for year in years if get_day(text,year) != ""]
    
    days=list(set(days))
    
    return days

def vectfordoc(array_dcvector_per_document, size_vec, subtitles, index_clusters):
    
    columns=[]
    
    for j in range(np.shape(array_dcvector_per_document)[0]):
        columns.append(subtitles[j])
    
    
    clusters = np.zeros(len(subtitles))
    acc = 0
    for subtitle in subtitles:
        for i in range(len(index_clusters)):
            if subtitle in index_clusters[i]:
                clusters[acc]=i
        acc=acc+1
    
    title=[]
    #title.append("clusters")
    for i in range(size_vec):
        title.append('Topic_'+str(i))


    dataframe = pd.DataFrame(array_dcvector_per_document.T, dtype="str", index=title)
    dataframe.columns=columns
    dataframe = dataframe.T
    dataframe.insert(0,"clusters",clusters)
    dataframe = dataframe.T
    return dataframe

def similar_subtitles(dic_subtitles,k_means, n_clusters, k_means_optimized):
    """This function group all the subtitles titles for cluster group"""
    k_means_label = k_means.labels_
    index_clusters=[]
    list_subtitles=list(dic_subtitles.keys())
    k_means_label = list(k_means_optimized.labels_)
    for i in range(n_clusters):
        index = []
        index = [list_subtitles[document_number] for document_number, cluster in enumerate(k_means_label) if cluster == i]
        index_clusters.append(index)
    
    return index_clusters

def printClusterDf(dataframe, n_documents, index_clusters):
    import os
    """Función que imprime en un excel los documentos pertenecientes a un cluster, y los tópicos a los que pertence"""
    path_clusters = config['DOC2VEC']['path_results_clusters']
    if not os.path.exists(path_clusters+str(n_documents)):
        os.makedirs(path_clusters+str(n_documents))
    with pd.ExcelWriter(path_clusters+str(n_documents)+'\\ClusterDf_'+str(n_documents)+'.xlsx') as writer:
        for i in range(len(index_clusters)):
            cluster = index_clusters[i]
            dataframe[cluster].T.astype(float).round(3).to_excel(writer, sheet_name='Cluster'+str(i))
    
