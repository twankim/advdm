# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:25:57 2015

@author: motro
"""

from cvxopt import matrix, solvers
import numpy as np
from helpers import *
    

# parameters
c = np.exp(-1) # regularization strength
nft = 20
ntrain = 80
ntest = 200
bins = 10.

## create simulated data
[Xtrain, Xtest, Ytrain, Ytest, mapping] = simulate(nft, ntrain, ntest, bins)

# sort
[Xtrain, Ytrain] = sortDown(Xtrain, Ytrain)
[Xtest, Ytest] = sortDown(Xtest, Ytest)


## MR with auxiliary margin and generic QP solver
if True:    

    # QP : min .5*z'Pz + qz s.t. Gz <= h
    # for this case, z = [w (weights), r (ranks), u (auxiliary ranks)]
    # P = 2(A'A + cI_w), where A is [-X I 0]
    # G is difference matrix, h is the required margin (currently set to 1) 
    rlen = Xtrain.shape[0]
    wlen = Xtrain.shape[1]
    [adjDiff, ulen] = diffMatrix(margins(Ytrain)) 
    
    A = np.concatenate((-Xtrain, np.eye(rlen), np.zeros((rlen,ulen))),
                       axis = 1)
    Iw = np.eye(wlen,wlen+rlen+ulen)
    P = 2*np.dot(A.transpose(),A) + 2*c*np.dot(Iw.transpose(),Iw)
    q = np.array([0.]*(wlen+rlen+ulen))
    G = -np.append(np.zeros((adjDiff.shape[0], wlen)), adjDiff , axis=1)
    h = np.array([-1.]*(G.shape[0]))
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    
    result = solvers.qp(P,q,G,h)
    
    x_out = np.array(result['x']).flatten()
    w_sol = x_out[:wlen]
    r_sol = x_out[wlen:wlen+rlen]
    
    trainError = np.sum(np.power( r_sol - np.dot(Xtrain,w_sol) ,2))
    magnitude = np.dot(w_sol, w_sol)
    functionCost = trainError + c*magnitude

    
## evaluations
mappingCosDist = np.dot(mapping,w_sol)/ np.power( np.dot(mapping,mapping) *
                    np.dot(np.transpose(w_sol),w_sol), .5)
#mappingCosDist = mappingCosDist[0][0]

#scores = np.append( np.dot(Xtrain,w_sol) , np.dot(Xtest,w_sol) )
#truth = np.append(Ytrain, Ytest)
scores = np.dot(Xtest,w_sol)
truth = Ytest
relevanceVector = truth[np.argsort(-scores)]

prc_k = 3 # Precision @ k


print "function cost " + str(functionCost)
print "mapping error " + str(mappingCosDist) 
print "nDCG " + str(nDCG(relevanceVector))
print "nDCG2 " + str(nDCG(relevanceVector))
print "precision@" + str(prc_k) + " " + str(precK(truth,scores,prc_k))