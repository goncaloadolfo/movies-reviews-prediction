# -*- coding: utf-8 -*-
"""
Módulo com métodos que permitem retirar desempenho para o problema multiclasse
e binário.
Created on Sun Dec 23 17:41:29 2018

@author: Gonçalo Adolfo, Frederico Costa
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from metricas_desempenho import bin_tc, desempenho_multiclasse, desempenho_bin, vis_palavras_maior_interesse
from sklearn.model_selection import StratifiedKFold


# datasets
PATH_BASE = "./datasets/"
DATASETS = ["dataset1.p", "dataset2.p", "dataset3.p", "dataset4.p", "dataset5.p", "dataset6.p"]


def scores_datasets(clf):
    '''
    Método que imprime os scores para cada dataset.
    '''
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
        
    for path_dataset in DATASETS:
        print("Path do dataset: ", PATH_BASE + path_dataset)
        
        ##### carregar dataset
        dataset = pickle.load(open(PATH_BASE + path_dataset, "rb"))
        X = dataset['data']
        y = dataset['target']
        vocab = dataset['vocab']
        print("Tamanho do vocabulário: ", len(vocab))
        y_bin = bin_tc(y)
        
        # classificar
        ye = cross_val_predict(clf, X, y, cv=kfold)
        ye_bin = bin_tc(ye)
        
        # scores
        score_multi = np.sum(y == ye) / len(y)
        score_bin = np.sum(y_bin == ye_bin) / len(y_bin)
        print("Score(multi-classe): ", score_multi)
        print("Score(binário): ", score_bin)
        print()


def desempenho_completo(X, y, y_bin, vocab, clf, nc=False):
    '''
    Método para retirar desempenho para um dado dataset: probabilidade de acerto
    e matriz de confusão(caso bin e multi); precision, recall, f-score, 
    curva precision-recall(caso bin). 
    '''   
    
    #### estimar classes com validação cruzada
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    ye = cross_val_predict(clf, X, y, cv=kfold)
    ye_bin = bin_tc(ye)
    
    ####### desempenho caso multinomial
    desempenho_multiclasse(y, ye)    
    
    ###### desempenho caso binário 
    desempenho_bin(clf, X, y_bin, ye_bin, nc)
    
    # se não for o NearestCentroid
    if not nc:
        #### visualizar palavras maior interesse
        vis_palavras_maior_interesse(clf, X, y_bin, vocab)
    
    plt.show()
      

if __name__ == "__main__":
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import NearestCentroid
    

    #dataset = "datasets/dataset1.p"
    dataset = "datasets/dataset_max_6539.p"
    #dataset = "datasets/dataset_max_9004.p"


    
    dataset = pickle.load(open(dataset, "rb"))
    X = dataset['data']
    y = dataset['target']
    vocab = dataset['vocab']
    y_bin = bin_tc(y) 
    


##    dl = LogisticRegression(penalty='l2', tol=1e-4, C=1, solver='saga', max_iter=1000,
##                            multi_class='multinomial', verbose=0)
##    scores_datasets(dl)
##    desempenho_completo(X, y, y_bin, vocab, dl)
    

##    linear_svc = LinearSVC(penalty='l2', C=0.1, loss='squared_hinge', dual=False, tol=1e-4, 
##                       multi_class='ovr', max_iter=1000)
##    scores_datasets(linear_svc)
##    desempenho_completo(X, y, y_bin, vocab, linear_svc)


##    nc = NearestCentroid(metric='cosine')
##    scores_datasets(nc)
##    desempenho_completo(X, y, y_bin, vocab, nc, True)








