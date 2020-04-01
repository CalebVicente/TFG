# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:34:59 2020

@author: cvicentm
"""

#FUNCTIONS:
#---------------------------------------------------------------------------------------

def get_news_between_times(subtitles_day, string_ini, string_final, num_range):
    news_start=subtitles_day.find(string_ini)
    news_end=subtitles_day.rfind(string_final)
    #I have to put differents hours depends on the channel where i catch subtitles
    initial_finder=subtitles_day.find(string_ini)
    initial_finder_end=subtitles_day.rfind(string_final)
    if initial_finder!=-1 and initial_finder_end!=-1:
        news_start=news_start
        news_end=news_end
    else:
        change_ini=0
        change_final=0
        for i in range(num_range):
            if initial_finder==-1 and change_ini==0:
                if i<10:
                    str_num="0"+str(i)
                else:
                    str_num=str(i)
                if subtitles_day.find(string_ini[0:3]+str_num+":")!=-1:
                    news_start=subtitles_day.find(string_ini[0:3]+str_num+":")
                    change_ini=1
            if initial_finder_end==-1 and change_final==0:
                if i>30:
                    str_num_end="0"+str(num_range-i)
                else:
                    str_num_end=str(num_range-i)
                if subtitles_day.rfind(string_final[0:3]+str_num_end+":")!=-1:
                    news_end=subtitles_day.rfind(string_final[0:3]+str_num_end+":")
                    change_ini=1
            
    new=subtitles_day[news_start:news_end]
    
    new = clean_subtitles(new)
    
    return new

def get_news(name, subtitles_day):
    
    new_morning=""
    new_afternoon=""
    num_range=0
    num_range2=0
    
    if "La1" in name:
        string_ini = "15:00:"
        string_final = "15:59:"
        num_range = 59 
        string_ini2 = "21:00:"
        string_final2 = "21:59:"
        num_range2 = 59 
    else: 
        if "Telecinco" in name:
            string_ini = "15:00:"
            string_final = "15:38:"
            num_range2 = 38 
            string_ini2 = "21:10:"
            string_final2 = "21:44:"
            num_range2 = 44 
        else:           
            if "antena3" in name:
                string_ini = "15:00:"
                string_final = "15:45:"
                num_range = 45
                string_ini2 = "21:00:"
                string_final2 = "21:30:"
                num_range2 = 30
            else: 
                if "Sexta" in name:
                    string_ini = "14:00:"
                    string_final = "14:55:"
                    num_range = 55
                    string_ini2 = "20:00:"
                    string_final2 = "20:55:"
                    num_range2 = 55
                

        new_morning = get_news_between_times(subtitles_day, string_ini, string_final, num_range)
        new_afternoon = get_news_between_times(subtitles_day, string_ini2, string_final2, num_range2)
    
    return new_morning , new_afternoon

def clean_subtitles(subtitle):
    #Maybe in this function i have to delete spaces between diferent sentences, because they are all separated because of the times of subtitles           
    subtitle_without_tag=re.sub('<[^>]+>', '', subtitle)
    subtitle_without_time=re.sub('[[@*&?].*[$@*]?]', '', subtitle_without_tag)
    
    return subtitle_without_time

#this function is not working
def dictToList(dict):
    list=[]
    for key, string in dict:
        list.append(string)
    return list
