# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import nltk
from nltk import word_tokenize

import gensim.corpora as corpora

generator_normalize=[['primera','palabra', 'vocabulario','frase','el', 'el', 'dia', 'vaya', 'dia','vocabulario','vocabulario'],
                    ['segunda', 'frase', 'vocabulario','vocabulario']]

id2word = corpora.Dictionary(generator_normalize)
corpus = [id2word.doc2bow(text) for text in generator_normalize]
