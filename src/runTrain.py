from sklearn.ensemble import RandomForestClassifier
#import cPickle
from sklearn.externals import joblib

def do(X_train_features):
	return

# X = X_train_features[range(0,100)]
# Y = X_train_features['is_duplicate']

# # Fit a random forest and extract predictions 
# forest = RandomForestClassifier(n_estimators = 700,min_samples_leaf=3,n_jobs=-1)

# # Fitting the forest may take a few minutes
# print "Fitting a random forest to labeled training data..."
# forest = forest.fit(X,Y)

# joblib.dump(forest, '../models/train/forest') 

# print "Completed fitting a random forest to labeled training data..."
# return forest
# #result = forest.predict(test_centroids)
# #return result

# # Read Train and Test Files
# X_train,X_test = data_read.do()
# # Get Word2Vec format sentences from data
# train_sentences,test_sentences = get_sentences.do(X_train['review'],X_test['review'])
# # Train Word2Vec model
# model = train_word2vec.do(train_sentences)
# # Get centroids Bag of Words
# train_centroids,test_centroids = centroid_bow.do(model,X_train,X_test)
# # Train a RandomForest to get results on test data
# result = train.do(X_train,train_centroids,test_centroids)
# # Output to file
# write_result.do(result,X_test)