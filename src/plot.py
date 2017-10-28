import numpy as np
import matplotlib.pyplot as plt

def do(X_train_features):
	la = np.linalg
	author = [author for author in X_train_features['author']]
	X = [sentence for sentence in X_train_features['sentence_vectors']]
	U, s, Vh = la.svd(X, full_matrices = False)

	#plot
	for i in range(len(author)):
	    plt.text(U[i,0], U[i,1], author[i])
	plt.show()