import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

import gensim
import pymorphy2


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

import warnings


df_train = pd.read_csv('../Data/topics_data/train.tsv', sep='\t', encoding='utf-8', lineterminator='\n', 
                       names=['Index', 'Title', 'Text', 'targets'])

df_test = pd.read_csv('../Data/topics_data/test.tsv', sep='\t', encoding='utf-8', lineterminator='\n', 
                       names=['Index', 'Title', 'Text'])

initshape = df_train.shape[0]
df_train = df_train[~df_train['targets'].isna()]
print(f'{initshape - df_train.shape[0]} Rows were deleted')

df_train['Text'].fillna('Notext', inplace=True)
df_test['Text'].fillna('Notext', inplace=True)

def transflabels(x):
    x = [int(i) for i in x.split(',')]
    return x

df_train['targets'] = df_train['targets'].apply(transflabels)

mlb = MultiLabelBinarizer()
mlb.fit(df_train['targets'])
Y = mlb.transform(df_train['targets'])

# Стоп слова для TF-IDF
stop_words_list = []
with open('OtherFiles/stopword_rus.txt') as F:
    for line in F:
        stop_words_list.append(F.readline().rstrip())
        
tfidf_title = TfidfVectorizer(ngram_range=(1, 3), max_features=30000, stop_words=stop_words_list)
X_idf_title = tfidf_title.fit_transform(df_train['Title'])
X_idf_title_test = tfidf_title.transform(df_test['Title'])

tfidf_text = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, stop_words=stop_words_list)
X_idf_text = tfidf_text.fit_transform(df_train['Text'])
X_idf_text_test = tfidf_text.transform(df_test['Text'])

X_conc_train = csr_matrix(hstack([X_idf_title, X_idf_text]))
X_conc_test = csr_matrix(hstack([X_idf_title_test, X_idf_text_test]))

X_train, X_val, Y_train, Y_val = train_test_split(X_conc_train, Y, test_size=0.25)

def fthreshold(x, th):
    return 1 if x > th else 0

fthreshold = np.vectorize(fthreshold)


svc_model = SVC(C=1)
svc_model.fit(X_train, Y_train[:, 0])
clf_svc = OneVsRestClassifier(lgbm_model, n_jobs=-1)
clf_svc.fit(X_train, Y_train)

with open('svc_model.pkl', 'wb') as F:
    pickle.dump(clf_svc, F)

preds = clf_svc.predict_proba(X_val)

f1_score(Y_val, fthreshold(preds, th=0.15), average='samples')

def correctpredicts(predicts, pred_max):
    """
    If all of predictions are lower than threshold, get most likelihood label
    """
    for line in range(len(predicts)):
        x = ','.join([str(i) for i in list(predicts[line])])
        if len(x) == 0:
            x = str(pred_max[line])
        predicts[line] = x
    return predicts

clf_svc.fit(X_conc_train, Y)

pred = clf_svc.predict_proba(X_conc_test)
pred_max = np.argmax(pred, axis=1)
pred = fthreshold(pred, 0.15)

pred = mlb.inverse_transform(pred)

pred = correctpredicts(pred, pred_max)

def submissionfile(predicts, submit_name):
    with open('../Submissions/'+submit_name+'.csv', 'w') as f:
        for ind, val in zip(df_test['Index'].values, predicts):
            f.write(str(ind))
            f.write('\t')
            f.write(val)
            f.write('\n')
            
            
submissionfile(pred, 'Submit_11_SVC_015')















































