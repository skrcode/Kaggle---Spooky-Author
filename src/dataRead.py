import pandas as pd
import sys
import glob
import errno
import csv
import numpy as np

def do():
	train = "../data/train.csv"
	test = "../data/test.csv"
	wv = "../../glove.6B/glove.6B.50d.txt"
	X_train = pd.read_csv( train, header=0,delimiter="," )
	X_test = pd.read_csv( test, header=0,delimiter="," )
	word_vecs = {}
	with open(wv) as f:
	    for line in f:
	       vals = line.split()
	       word_vecs[vals[0]] = np.array(vals[1::],dtype=float)
	return X_train,X_test,word_vecs