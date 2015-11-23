# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:34:37 2015

@author: motro
"""

import numpy as np
from helpers import *
from optTesters import *
import pandas as pd
import os
import time
import sklearn.cross_validation as splitter

# parameters
trainFile = os.path.realpath('../../jhu_data/trainData.csv')
c = np.exp(-5) # regularization strength
ntrain = 4050 # 45^2 * 2
method = 1


## gather data
fullData = pd.read_csv(trainFile, sep=',', 
                      header=0, index_col=None)
nobs , nft = fullData.shape
ntrain = min(ntrain,nobs)

# gather only ntrain samples (hopefully with PAV this will go up to 12k)
trainsplit = splitter.StratifiedShuffleSplit(fullData['score'], 
                                            n_iter = 1, train_size = ntrain)
escape = 0
for trainInds, testInds in trainsplit:
    escape += 1
    if escape > 1:
        break
    Xtrain = np.array(fullData.drop(['score'],axis=1))[trainInds,:]
    Ytrain = np.array(fullData['score'])[trainInds]
    Xtest = np.array(fullData.drop(['score'],axis=1))[testInds,:]
    Ytest = np.array(fullData['score'])[testInds]

# Learn scores
startTime = time.time()
[w_sol, functionCost] = optScore(Xtrain,Ytrain,Xtest,Ytest,method,c)
rankingRunTime = time.time() - startTime

## evaluations
scores = np.append( np.dot(Xtrain,w_sol) , np.dot(Xtest,w_sol) )
truth = np.append(Ytrain, Ytest)
relevanceVector = truth[np.argsort(-scores)]

prc_k = 22 # k value for Precision @ k

print "\nrun time in s = " + str(rankingRunTime)
print "function cost = " + str(functionCost)
print "nDCG = " + str(nDCG(relevanceVector))
print "nDCG2 = " + str(nDCG2(relevanceVector))
print "Precision@" + str(prc_k) + " = " + str(precK(truth,scores,prc_k)) + " (total# = " + str(np.size(truth,axis=0)) + ")"
