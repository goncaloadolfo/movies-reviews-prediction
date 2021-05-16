# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 01:55:22 2018

@author: Gonçalo Adolfo, Frederico Costa
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from text2vector import text2vector


def multiClassify(criticas):

    ##### carregar dataset
    dataset = pickle.load(open("datasets/dataset_max_6539.p", "rb"))
    X = dataset['data']
    y = dataset['target']

    dl = LogisticRegression(penalty='l2', tol=1e-4, C=1, solver='saga', max_iter=1000,
                            multi_class='multinomial', verbose=0)

    dl.fit(X, y)
    ye = dl.predict(criticas)

    return ye


if __name__ == "__main__":
    ## # carregar corpus
    dataset = pickle.load(open("datasets/imdbCriticas.p", "rb"))
    corpus = dataset["data"]
    tc = dataset['target']
    y = tc[1000:1500]
    
    criticas_teste = corpus[1000:1500]
    criticas_vector = text2vector(criticas_teste)
    ye = multiClassify(criticas_vector)
    print("Acertos: ", np.sum(y == ye))
