# -*- coding: utf-8 -*-
'''
Construção do vocabulário com base no corpus.

@author: Gonçalo Adolfo, Frederico Costa
'''

import pickle
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


# Construir dicionarios com max features obtidas com a regularizacao Lasso no discriminante
# logistico




def construir_dataset(minDf, pattern, stemmingObj, nGramas, pathDataset, mf):
    '''
    Método que guarda num pickle uma matriz com os valores tf-idf, a respetiva
    TC de cada amostra assim como o vocabulário. Os parâmetros permitem controlar
    o grau de limpeza pretendido, o tipo de stemming e o uso de n-gramas.
    '''

    timestamp_inicial = time.time()

    # carregar corpus
    dataset = pickle.load(open("datasets/imdbCriticas.p", "rb"))
    corpus = dataset["data"]
    tc = dataset['target']
    print("Dimensão do corpus: ", len(corpus))

    new_corpus = []

    for critica in corpus:
        # retirar br do html
        critica = critica.replace('<br />', ' ')

        # retirar caracteres que não do alfabeto
        critica = re.sub(r'[^a-zA-Z]+', ' ', critica)

        # stemming
        array_stem = [stemmingObj.stem(palavra) for palavra in critica.split()]
        texto = ' '.join(array_stem)
        new_corpus.append(texto)

    # tf-idf
    if nGramas is None:
        tf_idf = TfidfVectorizer(min_df=minDf, token_pattern=pattern)

    else:
        tf_idf = TfidfVectorizer(min_df=minDf, token_pattern=pattern, ngram_range=nGramas, max_features=mf)

    tf_idf.fit(new_corpus)
    vocab = tf_idf.get_feature_names()
    print("Dimensão final do vocabulário: ", len(vocab))
    X = tf_idf.transform(new_corpus)
    print("Shape do dicionário: ", X.shape)

    # guardar o dataset em um pickle
    new_dataset = {'data': X, 'target': tc, 'vocab': vocab}
    pickle.dump(new_dataset, open(pathDataset, "wb"))

    print("Tempo de processamento(s): ", time.time() - timestamp_inicial)


# Construcao Dicionarios com max Features
for mf in [73, 821, 6539, 9004, 14126, 35769, 73313]:
    print("##### Novo dataset...")
    minDf = 5
    pattern = r"\b\w\w+\b"
    stemmingObj = LancasterStemmer()
    nGramas = (1, 2)
    pathDataset = "./datasets/dataset_max_" + str(mf) + ".p"
    construir_dataset(minDf, pattern, stemmingObj, nGramas, pathDataset, mf)
    print()
