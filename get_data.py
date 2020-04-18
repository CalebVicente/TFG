# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""


import function_get_data
import os
import logging, logging.handlers
from datetime import datetime

def get_data():
    
    name_log_file = datetime.now().strftime('logs\load_subtitles_%d_%m_%Y.log')
    
    logging.basicConfig(filename=name_log_file, level=logging.WARNING, 
                        format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(message)s")
    
    
    #PROGRAM:
    #---------------------------------------------------------------------------------------
    
    path = 'subtitles'
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xlog' in file:
                files.append(os.path.join(r,file))
    
    
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
    
    
    for subtitle in files:
        
        if "La1" in subtitle or "Telecinco" in subtitle or "laSexta" in subtitle or "antena3" in subtitle:
    
                f=open(subtitle, "r",encoding="utf-8")
                if f.mode == 'r':
                    contents = f.read()
                    #creation of a dictionary whith the content of the news
                    dic_subtitles[subtitle+"morning_new"], dic_subtitles[subtitle+"afternoon_new"]=function_get_data.get_news(subtitle, contents)
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
    
    file_corpus.close()

    return dic_subtitles
