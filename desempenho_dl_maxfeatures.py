import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from metricas_desempenho import bin_tc, desempenho_multiclasse, desempenho_bin, vis_palavras_maior_interesse, \
    plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from hiper_parametros import estimar_hiperparametros
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC


# NAO USEI A FUNCAO DESEMPENHO BIN POR CAUSA DO TRAIN TEST SPLIT JA EXISTENTE NA CURVA PR

def desempenho_dl_maxfeatures(classificador, dataset):
    dataset = pickle.load(open(dataset, "rb"))
    X = dataset['data']
    y = dataset['target']
    vocab = dataset['vocab']
    print("Tamanho do vocabulário: ", len(vocab))
    y_bin = bin_tc(y)

    #### estimar classes com validação cruzada - Resultados muito perto da divisao train_test(nao da para obter pesos)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # ye = cross_val_predict(classificador, X, y_bin, cv=kfold)

    # Divisao treino teste
    X1, X2, y1, y2 = train_test_split(X, y_bin, test_size=0.2)

    # classificar
    classificador.fit(X1, y1)
    ye = classificador.predict(X2)

    # scores
    score_bin = np.sum(y2 == ye) / len(y2)
    print("Score(binário): ", score_bin)
    f1score = f1_score(y2, ye, average='binary')
    print("F1 Score(binário): ", f1score)
    print()

    # matriz de confusão
    matriz_confusao_bin = confusion_matrix(y2, ye)
    plot_confusion_matrix(matriz_confusao_bin, np.unique(y_bin))

    # precision, recall e f-score
    cl_vis = classification_report(y2, ye, target_names=["C.negativas", "C.positivas"])
    print()
    print(cl_vis)

    # curva precision recall
    decision_function = classificador.decision_function(X2)
    precisions, recalls, limiars = precision_recall_curve(y2, decision_function)

    plt.figure(), plt.title("Curva Precision-Recall"), plt.xlabel("Precision"), plt.ylabel("Recall"), plt.grid(True)
    plt.plot(precisions, recalls)

    # marcar limiar default
    ye2 = classificador.predict(X2)
    tp = np.sum(y2[ye2 == 1] == 1)
    fn = np.sum(y2[ye2 == 0] == 1)
    prec = tp / (tp + fn)
    array_auxiliar = np.abs(precisions - prec)
    indice_limiar_default = np.where(array_auxiliar == array_auxiliar.min())
    plt.plot(precisions[indice_limiar_default], recalls[indice_limiar_default], 'ok')
    print("Área por baixo da curva: ", average_precision_score(y2, decision_function))
    plt.show()

    return precisions, recalls


dl = LogisticRegression(penalty='l2', tol=1e-4, C=1, solver='saga', max_iter=10000,
                            multi_class='multinomial', verbose=0)

# lsvc = LinearSVC(penalty='l2', C=0.1, loss='squared_hinge', dual=False, tol=1e-4,multi_class='ovr', max_iter=1000) #TESTE COM OUTRO CLASSIFICADOR

# Observei que com max features a 6641 (c=10) o desempenho do discrimante logistico sem lasso
# é igual e ultrapassa ligeiramente a partir deste tamanho para cima
##for mf in [73, 821, 6539, 9004, 14126, 35769, 73313]:
##     pathDataset = "./datasets/dataset_max_" + str(mf) + ".p"
##     p, r = desempenho_dl_maxfeatures(dl, pathDataset)
#     if mf == 73: pr1 = [p, r]
#     if mf == 821: pr2 = [p, r]
#     if mf == 6539: pr3 = [p, r]
#     if mf == 9004: pr4 = [p, r]
#     if mf == 14126: pr5 = [p, r]
#     if mf == 35769: pr6 = [p, r]
#     if mf == 73313: pr7 = [p, r]
#
# prec_rec = {'c.1': pr1, 'c1': pr2, 'c10': pr3, 'c100': pr4, 'c1000': pr5, 'c10000': pr6, 'c100000': pr7}
#
# pickle.dump(prec_rec, open("./pr_dimensao_dic/pr_no_lasso.p", "wb"))
