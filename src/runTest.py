def do(X_test_features):
	results = X_test_features.copy()
	results['EAP'] = X_test_features['id']
	results['EAP'] = 0.33
	results['HPL'] = X_test_features['id']
	results['HPL'] = 0.33
	results['MWS'] = X_test_features['id']
	results['MWS'] = 0.34
	results = results.drop('text',axis=1)
	return results
	# X = X_test_features[range(0,100)]
	# forest = joblib.load('../models/train/forest')
	# result = forest.predict(X)
	# review = ''
	# with open("insert_answer", "r") as ins:
	#     array = []
	#     for line in ins:
	#         review += line
	# # Retrieve computed model
	# from gensim.models import Word2Vec
	# model = Word2Vec.load("src/Word2Vec_AnswerClass")
	# # Get centroids Bag of Words
	# with open('variables/num_clusters', 'rb') as f:
	# 	num_clusters = cPickle.load(f)
	# with open('variables/word_centroid_map', 'rb') as f:
	# 	word_centroid_map = cPickle.load(f)
	# # Test the saved RandomForest to get results on test data                                                                                                                                                                                                       
	# forest = joblib.load('models/forest')
	# print compute_genre_test(word_centroid_map,num_clusters,forest,review)