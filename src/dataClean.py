from nltk.corpus import stopwords
import re
import pandas as pd
import nltk.data
import nltk
import numpy as np

def getWordVectors(X_train_features,X_test_features,word_vecs):
	X_train_features['word_vectors'] = [ [ word_vecs[word] for word in sentence if word in word_vecs] for sentence in X_train_features['text']]
	X_test_features['word_vectors'] = [ [ word_vecs[word] for word in sentence if word in word_vecs] for sentence in X_test_features['text']] 
	return X_train_features,X_test_features

def getSentenceVectors(X_train_features,X_test_features):
	X_train_features['sentence_vectors'] =[np.mean(sentence,axis = 0) for sentence in X_train_features['word_vectors']]
	X_test_features['sentence_vectors'] =[np.mean(sentence,axis = 0) for sentence in X_test_features['word_vectors']] 
	return X_train_features,X_test_features

def clean(X_train,X_test):
	X_train_features = X_train.copy()
	X_test_features = X_test.copy()
	X_train_features['text'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_train['text']]
	X_test_features['text'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_test['text']]
	return X_train_features,X_test_features  