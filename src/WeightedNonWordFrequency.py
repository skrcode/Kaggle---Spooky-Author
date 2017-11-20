#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 08:58:34 2017

@author: 300006804
"""
import enchant
from functools import reduce
from nltk.stem.porter import PorterStemmer
class WeightedNonWordFrequencyPredictor:
    EAP = {};HPL={};MWS={}
    probabilities = []
    def __init__(self):
        print("Hello")
        
    def populate(self,author,sentence,d):
        nonEnglishWords = list(filter((lambda x: not d.check(x)),sentence))
        for word in nonEnglishWords:
            author[word] += 1
        return author
        
    def fit(self,X_train,Y_train):
        allWords = reduce((lambda x,y: x+y),X_train['text'])
        d = enchant.Dict('en')
        nonEnglishWords = list(filter((lambda x: not d.check(x)),allWords))  
        uniqueNonEnglishWords = set(nonEnglishWords)
        self.EAP = dict.fromkeys(uniqueNonEnglishWords,0)
        self.HPL = dict.fromkeys(uniqueNonEnglishWords,0)
        self.MWS = dict.fromkeys(uniqueNonEnglishWords,0)    
        for i in range(len(Y_train)):
            if(Y_train.iloc[i]) == 'EAP':
                self.EAP = self.populate(self.EAP,X_train.iloc[i,1],d)
            elif(Y_train.iloc[i]) == 'HPL':
                self.HPL = self.populate(self.HPL,X_train.iloc[i,1],d)
            else:
                self.MWS = self.populate(self.MWS,X_train.iloc[i,1],d)
                
    def predict(self,X_test):
        d = enchant.Dict('en')
        for i in range(len(X_test)):
            probs = []
            nonEnglishWords = list(filter((lambda x: not d.check(x)),X_test.iloc[i,1]))
            for word in nonEnglishWords:
                if (word in self.EAP):
                    total = self.EAP[word] + self.HPL[word] + self.MWS[word]
                    probs.append([self.EAP[word]/total,self.HPL[word]/total,self.MWS[word]/total])
                else:
                    probs.append([1,1,1])
            if len(nonEnglishWords) > 0:
                self.probabilities.append([sum(row[0]for row in probs),sum(row[1]for row in probs),
                                           sum(row[2]for row in probs)])
            else:
                self.probabilities.append([1,1,1])