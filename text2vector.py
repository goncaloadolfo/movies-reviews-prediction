# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 22:47:19 2018

@author: Gonçalo Adolfo, Frederico Costa
"""

import pickle
import re
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def text2vector(criticas):
    '''
    Retorna a representação tf-idf da lista de strings recebida. 
    '''
    print("Recebidas " + str(len(criticas)) + " críticas")
    # carregar vocabulário
    dataset = pickle.load(open("datasets/dataset_max_6539.p", "rb"))
    vocab = dataset['vocab']
    print("Dimensão do vocabulário: ", len(vocab))
    
    # aplicar limpeza
    criticas_limpas = []
    stemmingObj = LancasterStemmer()
    
    for critica in criticas:
            # retirar br do html
            critica = critica.replace('<br />', ' ')
        
            # retirar caracteres que não do alfabeto
            critica = re.sub(r'[^a-zA-Z]+', ' ', critica)
        
            # stemming
            array_stem = [stemmingObj.stem(palavra) for palavra in critica.split()]
            texto = ' '.join(array_stem)
            criticas_limpas.append(texto)
            
    # aplicar tf-idf
    tf_idf = TfidfVectorizer(min_df=5, token_pattern=r"\b\w\w+\b", vocabulary=vocab)
    tf_idf.fit(criticas_limpas)
    X = tf_idf.transform(criticas_limpas)
    print("Shape do array tf-idf: ", X.shape)
    
    return X
            







