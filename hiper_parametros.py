# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:58:33 2018

@author: Gonçalo Adolfo, Frederico Costa
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from metricas_desempenho import bin_tc


def estimar_hiperparametros(path_dataset, clf, grelha):
    '''
    Imprime na consola os melhores parâmetros da grelha para um dado classificador
    e para um dado dataset. São estimados através de uma pesquisa em grelha.
    '''
    # carregar o dataset
    dataset = pickle.load(open(path_dataset, "rb"))
    X = dataset['data']
    y = dataset['target']
    
    # dividir em treino e teste
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.25)
    y1_bin = bin_tc(y1)
    y2_bin = bin_tc(y2)
    
    # criar o objeto grid search
    grid_search = GridSearchCV(clf, grelha, cv=4).fit(X1, y1)
    
    # classificar
    ye1 = grid_search.predict(X1)
    ye2 = grid_search.predict(X2)
    ye1_bin = bin_tc(ye1)
    ye2_bin = bin_tc(ye2)
    
    # visualizar scores e melhores parâmetros
    print("Score no conjunto de treino(caso multiclasse): ", np.sum(ye1 == y1) / len(ye1))
    print("Score no conjunto de teste(caso multiclasse): ", np.sum(ye2 == y2) / len(ye2))
    
    print("Score no conjunto de treino(caso binário): ", np.sum(ye1_bin == y1_bin) / len(ye1_bin))
    print("Score no conjunto de teste(caso binário): ", np.sum(ye2_bin == y2_bin) / len(ye2_bin))
    
    print("Melhores parâmetros:", grid_search.best_params_)
    
    
def hiper_parametros_dl(dataset):
    '''
    Estimação melhor C e penalty para um discriminante logístico
    para um determinado dataset.
    '''
    #### definição dl e grid search
    dl = LogisticRegression(penalty='l2', tol=1e-4, solver='saga', max_iter=1000,
                        multi_class='multinomial', verbose=0)
    
    # ir regulando
    grelha = {'C': [0.01, 0.1, 1, 10, 100]}
    
    estimar_hiperparametros(dataset, dl, grelha)
    
    
def hiper_parametros_svm(dataset):
    '''
    Estimação melhor C para um SVM linear para um determinado 
    dataset.
    '''
    #### definição svm linear e gridsearch
    linear_svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4, 
                       multi_class='ovr', max_iter=1000)
    
    # ir regulando
    grelha = {'C': [0.01, 0.1, 1, 10, 100]}
    
    estimar_hiperparametros(dataset, linear_svc, grelha)
    
    
def hiper_parametros_nc(dataset):
    '''
    Estimação melhor métrica para um NearestCentroid para um determinado
    dataset.
    '''
    nc = NearestCentroid()

    grelha = {'metric':['manhattan', 'euclidean', 'cityblock', 'cosine']}

    estimar_hiperparametros(dataset, nc, grelha)


if __name__ == "__main__":
    #### datasets
    path_base = "datasets/"
    datasets = ["dataset1.p", "dataset2.p", "dataset3.p", "dataset4.p", "dataset5.p", "dataset6.p", "dataset_max_6539.p"]
  
    for dataset in datasets:
        print("#### DATASET: ", path_base + dataset)
        print()
        
        print("Hiper parametros DL:")
        hiper_parametros_dl(path_base + dataset)
        
        print("Hiper parametros Linear SVM:")
        hiper_parametros_svm(path_base + dataset)
        
        print("Hiper parametros NC:")
        hiper_parametros_nc(path_base + dataset)
        
        print()
        print()
    
    
    
    
    
        
    
    
    
    
    
