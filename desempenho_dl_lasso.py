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

dataset = pickle.load(open("./datasets/dataset6.p", "rb"))
X = dataset['data']
y = dataset['target']
vocab = dataset['vocab']
print("Tamanho do vocabulário: ", len(vocab))
y_bin = bin_tc(y)


def estimar_parametros(dataset):
    # estimar melhores parametros
    dl = LogisticRegression(penalty='l2', tol=1e-4, C=1, solver='saga', max_iter=1000,
                           multi_class='multinomial', verbose=0)
    grelha = {'C': [0.001, 0.01, 0.1, 10, 100, 1000, 10000, 100000, 1000000], 'solver': ['liblinear']}

    X1, X2, y1, y2 = train_test_split(X, y_bin, test_size=0.2)

    # criar o objeto grid search
    grid_search = GridSearchCV(dl, grelha, cv=4).fit(X1, y1)

    # classificar
    ye1 = grid_search.predict(X1)
    ye2 = grid_search.predict(X2)

    # visualizar scores e melhores parâmetros

    print("Score no conjunto de treino(caso binário): ", np.sum(ye1 == y1) / len(ye1))
    print("Score no conjunto de teste(caso binário): ", np.sum(ye2 == y2) / len(ye2))

    print("Melhores parâmetros:", grid_search.best_params_)


def desempenho_dl_lasso(c, vocab):
    dl = LogisticRegression(penalty='l1', tol=1e-4, C=1, solver='saga', max_iter=1000,
                            multi_class='multinomial', verbose=0)

    #### estimar classes com validação cruzada - Resultados muito perto da divisao (nao da para obter pesos)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # ye = cross_val_predict(dl, X, y_bin, cv=kfold)

    # Divisao treino teste
    X1, X2, y1, y2 = train_test_split(X, y_bin, test_size=0.2)

    # classificar
    dl.fit(X1, y1)
    ye = dl.predict(X2)

    # Vejo quantos pesos foram regularizados para zero
    nReg = np.sum(dl.coef_ == 0)
    print("Numero de pesos regularizados a 0: " + str(nReg))
    print("Tamanho do vocabulario apos regularizacao: " + str(len(vocab) - nReg))

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
    decision_function = dl.decision_function(X2)
    precisions, recalls, limiars = precision_recall_curve(y2, decision_function)

    plt.figure(), plt.title("Curva Precision-Recall"), plt.xlabel("Precision"), plt.ylabel("Recall"), plt.grid(True)
    plt.plot(precisions, recalls)

    # marcar limiar default
    ye2 = dl.predict(X2)
    tp = np.sum(y2[ye2 == 1] == 1)
    fn = np.sum(y2[ye2 == 0] == 1)
    prec = tp / (tp + fn)
    array_auxiliar = np.abs(precisions - prec)
    indice_limiar_default = np.where(array_auxiliar == array_auxiliar.min())
    plt.plot(precisions[indice_limiar_default], recalls[indice_limiar_default], 'ok')
    print("Área por baixo da curva: ", average_precision_score(y2, decision_function))

    '''
        Visualização das palavras mais discriminativas de críticas positivas de
        negativas com base nos pesos estimados.
        '''
    # estimar os pesos
    W = (dl.coef_).ravel()

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

    return precisions, recalls


# estimar_parametros("./datasets/dataset6.p")

##for c in [0.1, 1, 10, 100, 1000, 10000, 100000]:
##    p, r = desempenho_dl_lasso(c, vocab)
     # guardar precision e recall em um pickle
#     if c == 0.1: pr1 = [p, r]
#     if c == 1: pr2 = [p, r]
#     if c == 10: pr3 = [p, r]
#     if c == 100: pr4 = [p, r]
#     if c == 1000: pr5 = [p, r]
#     if c == 10000: pr6 = [p, r]
#     if c == 100000: pr7 = [p, r]
#
# prec_rec = {'c.1': pr1, 'c1': pr2, 'c10': pr3, 'c100': pr4, 'c1000': pr5, 'c10000': pr6, 'c100000': pr7}
#
# pickle.dump(prec_rec, open("./pr_dimensao_dic/pr_lasso.p", "wb"))

# prs = dataset = pickle.load(open("./pr_dimensao_dic/pr_lasso", "rb"))


