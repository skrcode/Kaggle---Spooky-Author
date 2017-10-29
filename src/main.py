#Importing all the libraries
import dataRead
import dataClean
import runTrain
import runTest
import writeResult
import WeightedWordFrequency as wwf

# Read data
#reload(dataRead)
reload()
X_train,X_test,word_vecs = dataRead.do()

#reload(dataClean)
X_train_features,X_test_features = dataClean.clean(X_train,X_test)
X_train_features,X_test_features = dataClean.removeStopWords(X_train_features,X_test_features)
X_train_features,X_test_features = dataClean.getWordVectors(X_train_features,X_test_features,word_vecs)
X_train_features,X_test_features = dataClean.getSentenceVectors(X_train_features,X_test_features)

# Train model
wwfpredictor = wwf.WeightedWordFrequencyPredictor()
wwfpredictor.fit(X_train_features.iloc[:,:-1],X_train_features.iloc[:,-1])
wwfpredictor.predict(X_test_features)

result = X_test[['id']]
result['EAP'] = [(row[0]/sum(row)) for row in wwfpredictor.probabilities]
result['HPL'] = [row[1]/sum(row) for row in wwfpredictor.probabilities]
result['MWS'] = [row[2]/sum(row) for row in wwfpredictor.probabilities]
result.to_csv('spooky.csv',index=False)
#reload(runTrain)
runTrain.do(X_train_features)

# Test model
#reload(runTest)
result = runTest.do(X_test_features)

# Write Results
#reload(writeResult)
writeResult.do(result,X_test_features)