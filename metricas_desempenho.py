# -*- coding: utf-8 -*-
"""
Módulo com métodos auxiliares para a avaliação de um classificador.
Created on Sun Dec 23 17:41:29 2018

@author: Gonçalo Adolfo, Frederico Costa
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def bin_tc(tc):
    '''
    Transforma um array de classes em classes binárias(só 0's e 1's) conforme as considerações de críticas positivas
    e negativas
    :param tc: array de inteiros
    '''
    new_tc = tc.copy()
    new_tc[tc <= 4] = 0
    new_tc[tc >= 7] = 1
    return new_tc


# função retirada do sklearn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def desempenho_multiclasse(y, ye):
    '''
    Com base na TC e na EC retira métricas de desempenho para o caso
    multinomial: probabilidade total de acerto / matriz de confusão.
    '''
    # probabilidade total de acerto
    prob_acerto = np.sum(y == ye) / len(y)
    print("#### desempenho problema multiclasse")
    print("Probabilidade total de acerto(caso multinomial): ", prob_acerto)
    
    # matriz de confusão
    matriz_confusao_mc = confusion_matrix(y, ye)
    plot_confusion_matrix(matriz_confusao_mc, np.unique(y))
    
    plt.show()
    

def desempenho_bin(classificador, X, y_bin, ye_bin, nc=False):
    '''
    Com base na TC e na EC retira métricas de desempenho para o caso
    binário: probabilidade total de acerto / matriz de confusão / precision
    recall e f-score / curva precision recall e respetiva área.
    '''
    # probabilidade total de acerto
    prob_acerto = np.sum(y_bin == ye_bin) / len(y_bin)
    print("#### desempenho problema binário")
    print("Probabilidade total de acerto(caso binário): ", prob_acerto)
    
    # matriz de confusão
    matriz_confusao_bin = confusion_matrix(y_bin, ye_bin)
    plot_confusion_matrix(matriz_confusao_bin, np.unique(y_bin))
    
    # precision, recall e f-score
    cl_vis = classification_report(y_bin, ye_bin, target_names=["C.negativas", "C.positivas"])
    print()
    print(cl_vis)
    
    # se não for o NearestCentroid
    if not nc:
        # curva precision recall
        X1, X2, y1, y2 = train_test_split(X, y_bin, test_size=0.2)
        classificador.fit(X1, y1)
        decision_function = classificador.decision_function(X2)
        precisions, recalls, limiars = precision_recall_curve(y2, decision_function)
        
        plt.figure(), plt.title("Curva Precision-Recall"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
        plt.plot(recalls, precisions)
        
        # marcar limiar default
        ye2 = classificador.predict(X2)
        tp = np.sum(y2[ye2 == 1] == 1)
        fn = np.sum(y2[ye2 == 0] == 1)
        prec = tp / (tp + fn) 
        array_auxiliar = np.abs(precisions - prec)
        indice_limiar_default = np.where(array_auxiliar == array_auxiliar.min())
        plt.plot(recalls[indice_limiar_default], precisions[indice_limiar_default], 'ok')
        print("Área por baixo da curva: ", average_precision_score(y2, decision_function))
    
    plt.show()
    
    
def vis_palavras_maior_interesse(classificador, X, y_bin, vocab):
    '''
    Visualização das palavras mais discriminativas de críticas positivas de
    negativas com base nos pesos estimados.
    '''
    # estimar os pesos
    classificador.fit(X, y_bin)
    W = (classificador.coef_).ravel()
    
    # obter pesos e vocabulário maior interesse
    idxs_ordenados = np.argsort(-np.abs(W))
    vocab = np.array(vocab)
    vocab_maior_interesse = vocab[idxs_ordenados[:20]] 
    ws_interesse = W[idxs_ordenados[:20]]
    
    # visualizar
    plt.figure(), plt.title("Palavras de maior interesse"), plt.xlabel("Palavra"), plt.ylabel("W"), plt.grid(True)
    plt.xticks(rotation=90)
    plt.plot(vocab_maior_interesse, ws_interesse, '.')
    
    plt.show()
    