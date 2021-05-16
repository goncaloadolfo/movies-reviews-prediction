import pickle
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import matplotlib.pyplot as plt

pr_no_lasso = pickle.load(open("./pr_dimensao_dic/pr_no_lasso.p", "rb"))
pr_lasso = pickle.load(open("./pr_dimensao_dic/pr_lasso.p", "rb"))


plt.title("Tamanho de dicionario = 73"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c.1"][0], pr_lasso["c.1"][1], label='regularizado')
plt.plot(pr_no_lasso["c.1"][0], pr_no_lasso["c.1"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 821"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c1"][0], pr_lasso["c1"][1], label='regularizado')
plt.plot(pr_no_lasso["c1"][0], pr_no_lasso["c1"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 6539"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c10"][0], pr_lasso["c10"][1], label='regularizado')
plt.plot(pr_no_lasso["c10"][0], pr_no_lasso["c10"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 9004"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c100"][0], pr_lasso["c100"][1], label='regularizado')
plt.plot(pr_no_lasso["c100"][0], pr_no_lasso["c100"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 14126"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c1000"][0], pr_lasso["c1000"][1], label='regularizado')
plt.plot(pr_no_lasso["c1000"][0], pr_no_lasso["c1000"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 35769"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c10000"][0], pr_lasso["c10000"][1], label='regularizado')
plt.plot(pr_no_lasso["c10000"][0], pr_no_lasso["c10000"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()

plt.title("Tamanho de dicionario = 73313"), plt.xlabel("Recall"), plt.ylabel("Precision"), plt.grid(True)
plt.plot(pr_lasso["c100000"][0], pr_lasso["c100000"][1], label='regularizado')
plt.plot(pr_no_lasso["c100000"][0], pr_no_lasso["c100000"][1], label='com max features')
plt.legend(loc='upper right')
plt.show()
