# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:25:57 2015

@author: motro
"""

from cvxopt import matrix, solvers
import numpy as np
from helpers import *
    

# parameters
c = np.exp(-5) # regularization strength
nft = 20
ntrain = 80
ntest = 200
bins = 10.
method = 1
# 1 = MR with auxiliary margin and generic QP solver
# 2 = MR with auxiliary margin and soft cost - not complete yet
# 3 = MINLIP with hard cost - min ||w|| s.t. DXw > c1
# 4 = MINLIP with sort cost - min ||w|| - c1'DXw s.t. DXw > 0
# 5 = nonlinear MR with QP and newton's method  


## create simulated data
[Xtrain, Xtest, Ytrain, Ytest, mapping] = simulate(nft, ntrain, ntest, bins)

# sort
[Xtrain, Ytrain] = sortDown(Xtrain, Ytrain)
[Xtest, Ytest] = sortDown(Xtest, Ytest)


## MR with auxiliary margin and generic QP solver
if method == 1:    

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
    
## MR with auxiliary margin and soft cost - not complete yet
if method == 2:

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
    
## MINLIP with hard cost - min ||w|| s.t. DXw > c1
if method == 3:    

    # QP : min .5*z'Pz + qz s.t. Gz <= h
    # z = [w u], P = [I 0]', q = 0, G = -D[X 0; 0 I], h = -c1
    nobs = Xtrain.shape[0]
    wlen = Xtrain.shape[1]
    [adjDiff, ulen] = diffMatrix(margins(Ytrain)) 

    P = np.append(np.eye(wlen, wlen+ulen), np.zeros((ulen, wlen+ulen)), axis=0)
    q = np.array([0.]*(wlen+ulen))    
    
    G = np.append( np.append(Xtrain,np.zeros((ulen,wlen)),axis=0),
                  np.append(np.zeros((nobs,ulen)),np.eye(ulen),axis=0), axis=1)
    G = -np.dot( adjDiff, G)
    h = np.array([-c]*(G.shape[0]))
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    
    result = solvers.qp(P,q,G,h) 
    x_out = np.array(result['x']).flatten()
    w_sol = x_out[:wlen]
    
    functionCost = np.dot(w_sol, w_sol)
    
## MINLIP with sort cost - min ||w|| - c1'DXw s.t. DXw > 0
if method == 4:    

    # QP : min .5*z'Pz + qz s.t. Gz <= h
    # z = [w u], P = [I 0]', q = 0, G = -D[X 0; 0 I], h = -c1
    nobs = Xtrain.shape[0]
    wlen = Xtrain.shape[1]
    [adjDiff, ulen] = diffMatrix(margins(Ytrain)) 

    P = np.append(np.eye(wlen, wlen+ulen), np.zeros((ulen, wlen+ulen)), axis=0)
    
    G = np.append( np.append(Xtrain,np.zeros((ulen,wlen)),axis=0),
                  np.append(np.zeros((nobs,ulen)),np.eye(ulen),axis=0), axis=1)
    G = -np.dot( adjDiff, G)
    q = c*np.dot(np.transpose(G) , np.array([1.]*G.shape[0]))
    h = np.array([0.]*(G.shape[0]))
    
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    
    result = solvers.qp(P,q,G,h) 
    x_out = np.array(result['x']).flatten()
    w_sol = x_out[:wlen]
    
    functionCost = np.dot(w_sol, w_sol)

## nonlinear MR with QP and newton's method    
if method == 5:

    nobs = Xtrain.shape[0]
    nft = Xtrain.shape[1]
    adjDiff = np.eye(nobs-1,nobs) - np.eye(nobs-1,nobs,1)
    Xinv = np.linalg.pinv(Xtrain)
    covmatrix = np.dot(np.transpose(Xtrain),Xtrain)
    eigs, vecs = np.linalg.eig(covmatrix)
    descentstep = 1/(4*np.max(eigs))
    
    def dpsi(x):
        return x
    def ddphi(x):
        return np.ones(np.shape(x))

    r = Ytrain # easy initialization    
    for wholerun in range(5):
        # gradient descent with w
        w = np.dot(Xinv,r) # solve for least squares, initialization
        escape = 0
        grad = 100
        while escape < 1000 and norm(grad) > .01:
            grad = np.dot(np.transpose(Xtrain), dpsi(np.dot(Xtrain,w))-r) + c*w
            w = w - descentstep*grad
            escape += 1
        
        # sort difference matrix
        pockets = margins(Ytrain)
        counter = 0
        for pockety in pockets:        
            if len(pockety) > 1:
                pocket = np.arange(counter,counter+len(pockety))
                sortedInds = np.argsort(-np.dot(Xtrain[pocket],w))
                adjDiff[:,pocket] = adjDiff[:,pocket[sortedInds]]
            counter += len(pockety)
            
        # solve quadratic program for r        
        P = 2*matrix(np.eye(nobs))
        q = matrix(np.array([0.]*nobs))
        G = matrix(-adjDiff)
        h = matrix(np.array([-1.]*(nobs-1)))
        
        result = solvers.qp(P,q,G,h) 
        r = np.array(result['x']).flatten()
        
    w_sol = w
    functionCost = 0
        
        
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