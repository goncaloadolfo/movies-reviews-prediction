# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:39:33 2019

@author: Gonçalo Adolfo, Frederico Costa
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def agrupar_criticas(k):
    '''
    Dado um número de grupos pretendidos, aplica o classificador
    k-médias para agrupar críticas de cinema. Imprime e desenha plots
    relativos às palavras mais significativas em cada grupo.
    '''
    ### carregar dataset
    dataset = pickle.load(open("datasets/dataset1.p", "rb"))
    X = dataset['data']
    X = X[:10000] # apenas com 1000 criticas para diminuir tempo de processamento
    vocab = np.array(dataset['vocab'])
    print("dataset carregado...")
    
    #### aplicar k-medias com k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=1e-4)
    kmeans.fit(X)
    ye = kmeans.labels_
    clusters = kmeans.cluster_centers_
    print("classificação concluída")
    
    # ver histograma das classes
    plt.figure(), plt.title("Amostras por classe"), plt.xlabel("Classe"), plt.ylabel("Nº de amostras")
    plt.hist(ye)
    plt.grid(True)
    
    # ver palavras mais importantes em cada cluster
    for i in range(len(clusters)):
        print("A processar cluster " + str(i))
        Xi = X[ye == i] # amostras do cluster i
        maxs = np.max(Xi, axis=0).toarray().ravel() # máximos de cada dim
        
        # ordenar por ordem decrescente
        idx = np.argsort(-maxs) 
        maxs_ordenados = maxs[idx]
        vocab_ordenado = vocab[idx]
        
        # visualizar palavras mais distintivas em cada classe
        plt.figure(), plt.title("Palavras de maior interesse Cluster " + str(i)), plt.xlabel("Palavra")
        plt.ylabel("tf-idf"), plt.grid(True)
        plt.xticks(rotation=90)
        plt.plot(vocab_ordenado[:30], maxs_ordenados[:30], '.')
        
        #mais palavras
        print(vocab_ordenado[:60])
        
    plt.show()
    

if __name__ == "__main__":
    
    #agrupar_criticas(12)
    
    #agrupar_criticas(20)
    
    #agrupar_criticas(5)
    
