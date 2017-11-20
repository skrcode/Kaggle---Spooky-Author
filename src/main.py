#Importing all the libraries
import dataRead
import dataClean
import runTrain
import runTest
import writeResult
import WeightedWordFrequency as wwf
import WeightedNonWordFrequency as wnwf

# Read data
#reload(dataRead)
reload()
X_train,X_test,word_vecs = dataRead.do()

#reload(dataClean)
X_train_features,X_test_features = dataClean.clean(X_train,X_test)
X_train_features,X_test_features = dataClean.removeStopWords(X_train_features,X_test_features)
X_train_features,X_test_features = dataClean.getWordVectors(X_train_features,X_test_features,word_vecs)
X_train_features,X_test_features = dataClean.getSentenceVectors(X_train_features,X_test_features)

# Train model Word model
wwfpredictor = wwf.WeightedWordFrequencyPredictor()
wwfpredictor.fit(X_train_features.iloc[:,:-1],X_train_features.iloc[:,-1])
wwfpredictor.predict(X_test_features)

#NonWord model
wnwfpredictor = wnwf.WeightedNonWordFrequencyPredictor()
wnwfpredictor.fit(X_train_features.iloc[:,:-1],X_train_features.iloc[:,-1])
wnwfpredictor.predict(X_test_features)

result = X_test[['id']]
result['EAP'] = [(row[0]/sum(row)) for row in wnwfpredictor.probabilities]
result['HPL'] = [row[1]/sum(row) for row in wnwfpredictor.probabilities]
result['MWS'] = [row[2]/sum(row) for row in wnwfpredictor.probabilities]

result2 = X_test[['id']]
result2['EAP'] = [(row[0]/sum(row)) for row in wwfpredictor.probabilities]
result2['HPL'] = [row[1]/sum(row) for row in wwfpredictor.probabilities]
result2['MWS'] = [row[2]/sum(row) for row in wwfpredictor.probabilities]

resultFinal = X_test[['id']]
resultFinal['EAP'] = [result['EAP'][i]*result2['EAP'][i] for i in range(len(X_test))]
resultFinal['HPL'] = [result['HPL'][i]*result2['HPL'][i] for i in range(len(X_test))]
resultFinal['MWS'] = [result['MWS'][i]*result2['MWS'][i] for i in range(len(X_test))]

resultToSend = X_test[['id']]
resultToSend['EAP'] = [resultFinal['EAP'][i]/(resultFinal['EAP'][i]+resultFinal['HPL'][i]+
             resultFinal['MWS'][i]) for i in range(len(X_test))]
resultToSend['HPL'] = [resultFinal['HPL'][i]/(resultFinal['EAP'][i]+resultFinal['HPL'][i]+
             resultFinal['MWS'][i]) for i in range(len(X_test))]
resultToSend['MWS'] = [resultFinal['MWS'][i]/(resultFinal['EAP'][i]+resultFinal['HPL'][i]+
             resultFinal['MWS'][i]) for i in range(len(X_test))]

resultToSendFinal = resultToSend.copy()
for i in range(len(resultToSend)):
    if resultToSendFinal['EAP'][i] == 1:
        resultToSendFinal.iloc[i,1] = 0.9
        resultToSendFinal.iloc[i,2] = 0.05
        resultToSendFinal.iloc[i,3] = 0.05
    elif resultToSendFinal['HPL'][i] == 1:
        resultToSendFinal.iloc[i,1] = 0.05
        resultToSendFinal.iloc[i,2] = 0.9
        resultToSendFinal.iloc[i,3] = 0.05
    elif resultToSendFinal['MWS'][i] == 1:
        resultToSendFinal.iloc[i,1] = 0.05
        resultToSendFinal.iloc[i,2] = 0.05
        resultToSendFinal.iloc[i,3] = 0.9  
        
for i in range(len(resultToSend)):
    if resultToSendFinal['EAP'][i] == 0:
        resultToSendFinal.iloc[i,1] = 0.0002
        resultToSendFinal.iloc[i,2] -= 0.0001
        resultToSendFinal.iloc[i,3] -= 0.0001
    elif resultToSendFinal['HPL'][i] == 0:
        resultToSendFinal.iloc[i,1] -= 0.0001
        resultToSendFinal.iloc[i,2] = 0.0002
        resultToSendFinal.iloc[i,3] -= 0.0001
    elif resultToSendFinal['MWS'][i] == 0:
        resultToSendFinal.iloc[i,1] -= 0.0001
        resultToSendFinal.iloc[i,2] -= 0.0001
        resultToSendFinal.iloc[i,3] = 0.0002 
        
resultToSendFinal.to_csv('spooky3.csv',index=False)
#reload(runTrain)
runTrain.do(X_train_features)

# Test model
#reload(runTest)
result = runTest.do(X_test_features)

# Write Results
#reload(writeResult)
writeResult.do(result,X_test_features)