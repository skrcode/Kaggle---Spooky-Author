import pandas as pd
import sys
import glob
import errno

def do():
	train = "../data/train.csv"
	test = "../data/test.csv"
	X_train = pd.read_csv( train, header=0,delimiter="," )
	X_test = pd.read_csv( test, header=0,delimiter="," )

	return X_train,X_test