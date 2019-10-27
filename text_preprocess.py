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

import re

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

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


X = []
sentences = list(df_train['Text'])
for sen in sentences:
    X.append(preprocess_text(sen))

y_train = df_train['targets'].values

with open('OtherFiles/sentences_train.pkl', 'wb') as F:
    pickle.dump(sentences, F)


X = []
sentences = list(df_test['Text'])
for sen in sentences:
    X.append(preprocess_text(sen))

with open('OtherFiles/sentences_test.pkl', 'wb') as F:
    pickle.dump(sentences, F)










































