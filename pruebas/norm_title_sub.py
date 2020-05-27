# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:19:15 2020

@author: cvicentm
"""

import create_corpus as c
import variables as v
from tqdm import tqdm
import get_data as g



years = v.YEARS
channels = v.CHANNELS
hour_new = v.NEWS

def normalize_title_subtitles(subtitle):
    
    def get_day(text,year):
        """this function returns the date from a subtitle title given the year"""
        n_day=text.find(year)
        day=""
        if n_day!=-1:
            day=text[n_day:(n_day+10)]
        
        return day

    channel = [channel for channel in channels if subtitle.find(channel)!=-1]
    
    day=[get_day(subtitle,year) for year in years if get_day(subtitle,year) != ""]
    
    hour = [hour for hour in hour_new if subtitle.find(hour)!=-1]
    
    new_subtitle = str(channel[0])+"_"+str(day[0])+"_"+str(hour[0])
    
    return new_subtitle

#------------------------------------------------------------------------------------------
[files, max_documents] = g.get_NameFiles()


def norm_title_sub(max_documents=max_documents):
    import logging, logging.handlers
    from datetime import datetime
    name_log_file = datetime.now().strftime('load_subtitles_%d_%m_%Y.log')
    logging.basicConfig(filename=name_log_file, level=logging.WARNING, 
                        format="%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(message)s")
    
    
    dic_subtitles=c.create_dic_subtitles(max_documents)
    
    norm_dict_subt = {}
        
    for (o_key,value) in tqdm(dic_subtitles.items()):
        
        
        n_key = normalize_title_subtitles(o_key)
        
        if n_key not in norm_dict_subt:
        
            norm_dict_subt[n_key] = dic_subtitles[o_key]
        
        else:
            #PONER ESTO COMO LOG
            logging.warning("la clave "+str(n_key)+" esta repetida")
            
    return norm_dict_subt
