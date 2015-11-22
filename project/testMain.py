# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:00:12 2015

@author: twankim
"""

import numpy as np
from helpers import *
from optTesters import *

# parameters
c = np.exp(-5) # regularization strength
nft = 20
ntrain = 80
ntest = 200
bins = 10.
method = 1

## create simulated data
[Xtrain, Xtest, Ytrain, Ytest, mapping] = simulate(nft, ntrain, ntest, bins)

# Learn scores
[w_sol, functionCost] = optScore(Xtrain,Ytrain,Xtest,Ytest,method,c)

## evaluations
mappingCosDist = np.dot(mapping,w_sol)/ np.power( np.dot(mapping,mapping) *
                    np.dot(np.transpose(w_sol),w_sol), .5)

scores = np.append( np.dot(Xtrain,w_sol) , np.dot(Xtest,w_sol) )
truth = np.append(Ytrain, Ytest)
relevanceVector = truth[np.argsort(-scores)]

prc_k = 22 # k value for Precision @ k

print "\nfunction cost = " + str(functionCost)
print "mapping error = " + str(mappingCosDist) 
print "nDCG = " + str(nDCG(relevanceVector))
print "nDCG2 = " + str(nDCG2(relevanceVector))
print "Precision@" + str(prc_k) + " = " + str(precK(truth,scores,prc_k)) + " (total# = " + str(np.size(truth,axis=0)) + ")"
