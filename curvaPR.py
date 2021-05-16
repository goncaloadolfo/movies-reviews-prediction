
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from metricas_desempenho import bin_tc, desempenho_multiclasse, desempenho_bin, vis_palavras_maior_interesse
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def curvaPR(path_dataset, clf, nc=False):
    precisions = 0
    recalls = 0

    ##### carregar dataset
    dataset = pickle.load(open(path_dataset, "rb"))
    X = dataset['data']
    y = dataset['target']
    vocab = dataset['vocab']
    y_bin = bin_tc(y)

    #### estimar classes com validação cruzada
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    ye = cross_val_predict(clf, X, y, cv=kfold)
    ye_bin = bin_tc(ye)

    # precision, recall e f-score
    cl_vis = classification_report(y_bin, ye_bin, target_names=["C.negativas", "C.positivas"])
    print()
    print(cl_vis)

    # se não for o NearestCentroid
    if not nc:
        # curva precision recall
        X1, X2, y1, y2 = train_test_split(X, y_bin, test_size=0.2)
        clf.fit(X1, y1)
        decision_function = clf.decision_function(X2)
        precisions, recalls, limiars = precision_recall_curve(y2, decision_function)


        # marcar limiar default
        ye2 = clf.predict(X2)
        tp = np.sum(y2[ye2 == 1] == 1)
        fn = np.sum(y2[ye2 == 0] == 1)
        prec = tp / (tp + fn)
        array_auxiliar = np.abs(precisions - prec)
        indice_limiar_default = np.where(array_auxiliar == array_auxiliar.min())
        plt.plot(precisions[indice_limiar_default], recalls[indice_limiar_default], 'ok')
        print("Área por baixo da curva: ", average_precision_score(y2, decision_function))


    return precisions, recalls




# ##### desempenho para o discriminante logístico
dl = LogisticRegression(penalty='l2', tol=1e-4, C=1, solver='saga', max_iter=1000,
                        multi_class='multinomial', verbose=0)

##### desempenho para o SVM linear
linear_svc = LinearSVC(penalty='l2', C=0.1, loss='squared_hinge', dual=False, tol=1e-4, 
                       multi_class='ovr', max_iter=1000)


p1,r1 = curvaPR("datasets/dataset1.p", dl, False)
p2,r2 = curvaPR("datasets/dataset1.p", linear_svc, False)


plt.title("Curva Precision-Recall"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(p1, r1)
plt.plot(p2, r2)
plt.show()

