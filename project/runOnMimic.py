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

# parameters
c = np.exp(-5) # regularization strength
nft = 20
ntrain = 2000
ntest = 200
bins = 10.
method = 2

## gather data
readFolder = os.path.realpath('./interpolatedv1.1')
filelist = os.listdir(readFolder)
npatients = 100#len(filelist)

first = True
for fileName in filelist[:npatients]:
    patient = pd.read_csv(readFolder +'/'+ fileName, sep=',', 
                          header=None, index_col=0)
    patient.columns = list((np.str(i) for i in range(44)))+['score','treatment']
    patient.drop(np.where(np.isnan(patient['score']))[0], axis=0, inplace=True)
    patient.drop(['5','15'],axis=1, inplace=True)
    if first:
        fullData = patient.copy()
        first=False
    else:
        fullData = fullData.append(patient)

nobs , nft = fullData.shape
ntrain = min(ntrain, nobs)
Xtrain = np.array( fullData.drop(['score','treatment'],axis=1).iloc[:ntrain] )
Ytrain = np.array( fullData['score'].iloc[:ntrain] )
Xtest = np.array( fullData.drop(['score','treatment'],axis=1).iloc[ntrain:] )
Ytest = np.array( fullData['score'].iloc[ntrain:] )

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
