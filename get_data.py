# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import os
import re


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

for subtitle in files:
    
    if "La1" in subtitle or "Telecinco" in subtitle or "laSexta" in subtitle or "antena3" in subtitle:

            f=open(subtitle, "r",encoding="utf-8")
            if f.mode == 'r':
                contents = f.read()
                dic_subtitles[subtitle]=get_news(contents)



#FUNCTIONS:
#---------------------------------------------------------------------------------------

def get_news(subtitles_day):
    #I have to put differents hours depends on the channel where i catch subtitles
    news_start=subtitles_day.find("15:00:")
    news_end=subtitles_day.rfind("15:40:")
    new=subtitles_day[news_start:news_end]
    new = clean_subtitles(new)
    return new   

def clean_subtitles(subtitle):
    #Maybe in this function i have to delete spaces between diferent sentences, because they are all separated because of the times of subtitles           
    
    subtitle_without_tag=re.sub('<[^>]+>', '', subtitle)
    subtitle_without_time=re.sub('[[@*&?].*[$@*]?]', '', subtitle_without_tag)
    
    return subtitle_without_time
