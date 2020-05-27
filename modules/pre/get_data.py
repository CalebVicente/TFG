# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import sys
sys.path.insert(0,'..')
from modules.pre.function_get_data import get_news
import os
import logging, logging.handlers
from datetime import datetime
import timeit
import progressbar
from tqdm import tqdm

name_log_file = datetime.now().strftime('logs\load_subtitles_%d_%m_%Y.log')
    
logging.basicConfig(filename=name_log_file, level=logging.WARNING, 
                    format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(message)s") 


def get_NameFiles():
    path = 'subtitles'
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xlog' in file:
                files.append(os.path.join(r,file))
    
    
                
    return files, len(files)

def get_data(n_documents):
  
    
    #PROGRAM:
    #---------------------------------------------------------------------------------------
    files = []
    [files,max_files]= get_NameFiles()
    
    
    
    dic_subtitles={}
    list_subtitles=[]
    #Creation of a file with all the corpus information, this file will give us information related to
    #how many of the news are empty
    
    #os.remove("file_corpus.txt")
    try:
        os.remove("file_corpus.txt")
    except FileNotFoundError :
        print("File file_corpus.txt has been created")
    
    
    file_corpus=open("file_corpus.txt", "w",encoding="utf-8")
    
    #tic and toc are used to know how many time the process of extaction has taken
    tic=timeit.default_timer()
    #tqdm(files) is given a progress bar in this for:
    print("The process of extraction has been started: ")
    for subtitle in tqdm(files[0:n_documents]):
        
        if "1_spa" in subtitle or "Telecinco" in subtitle or "laSexta" in subtitle or "antena3" in subtitle or "Telemadrid" in subtitle:
                
                try:
                    f=open(subtitle, "r",encoding="utf-8")
                        
                    try:
                        if f.mode == 'r':
                            
                            contents = f.read()
                            #creation of a dictionary whith the content of the news
                            dic_subtitles[subtitle+"morning_new"], dic_subtitles[subtitle+"afternoon_new"]=get_news(subtitle, contents)
                            if dic_subtitles[subtitle+"morning_new"]=="":
                                logging.warning("El subtitulo: "+subtitle+"morning_new está vacio \n")
                            if dic_subtitles[subtitle+"afternoon_new"]=="":
                                logging.warning("El subtitulo: "+subtitle+"afternoon_new está vacio \n")
                                
                            #creation of a list whith the content of the news
                            #MAYBE I SHOUDNT CALL GET_NEWS(CONTENT), BECAUSE I ALREADY HAVE THE INFORMATION IN THE DICTIONARY
                            #list_subtitles.append(get_news(subtitle, contents))
                            
                            file_corpus.write(subtitle+"morning_new"+"\n")
                            file_corpus.write("----------------------------------------------------------------\n")
                            file_corpus.write(dic_subtitles[subtitle+"morning_new"])
                            file_corpus.write("----------------------------------------------------------------\n")
                            file_corpus.write(subtitle+"afternoon_new"+"\n")
                            file_corpus.write("----------------------------------------------------------------\n")
                            file_corpus.write(dic_subtitles[subtitle+"afternoon_new"])
                            file_corpus.write("----------------------------------------------------------------\n")
                        f.close()
                    except UnicodeDecodeError:
                        logging.warning("UnicodeDecodeError --- The subtititle: "+subtitle+" cant be decodified  \n")
                            
                except OSError:
                     logging.warning("OSError --- The subtititle: "+subtitle+" cant be opened  \n")
            
                   
    toc=timeit.default_timer()
    time_process=tic-toc
    print("The process of estract all subtitles has taken "+str(time_process)+" seconds")            
                
    
    file_corpus.close()

    return dic_subtitles
