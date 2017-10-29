from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
def do(result,X_train_features):
	eap = log_loss(np.array(pd.get_dummies(X_train_features['author'])['EAP']), np.array(result['EAP']))
	mws = log_loss(np.array(pd.get_dummies(X_train_features['author'])['MWS']), np.array(result['MWS']))
	hpl = log_loss(np.array(pd.get_dummies(X_train_features['author'])['HPL']), np.array(result['HPL']))
	return 'EAP:',eap,'MWS:',mws,'HPL:',hpl 