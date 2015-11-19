# -*- coding: utf-8 -*-

import numpy as np

# some lazy simulated data (gaussian with a few covariances,
#       and weighted relationship with output)
def simulate(nft, ntrain, ntest, bins = 0):
    featureCovariance = np.diag(np.random.uniform(0.0,5.0,size=nft))
    correlatePairs = [(1,2),(1,3),(4,5)]
    for pair in correlatePairs:
        cov = featureCovariance[pair[0],pair[0]]*featureCovariance[pair[1],pair[1]]
        cov = cov*np.random.uniform(0.0,1.0)
        featureCovariance[pair[0],pair[1]]=cov
    
    X = np.random.multivariate_normal(mean=[0]*nft,cov=featureCovariance,
                                      size=ntrain+ntest)
    
    mapping = np.random.exponential(1.0, size=nft)
    Scores = np.dot(X,mapping) + np.random.normal(0.0,0.3,size=ntrain+ntest)
    if bins > 0: # discretize
        breaks = np.arange(bins)/bins*100.
        breakVals = np.percentile(Scores, breaks.tolist())
        Scores = [sum(breakVals < score + .0000001) for score in Scores]
    
    Xtrain = X[:ntrain]
    Xtest = X[ntrain:]
    Ytrain = Scores[:ntrain]
    Ytest = Scores[ntrain:]
    return [Xtrain, Xtest, Ytrain, Ytest, mapping]


def sortDown(X,Y):
    Y = np.array(Y)
    sortedInds = np.argsort(-Y) # descending
    return [X[sortedInds],Y[sortedInds]]
    

def margins(Y): # takes a sorted (descending) list and folds same elements
    # ex: [4,4,3,2,2,1] -> [[4,4],[3][2,2],[1]]
    foldedY = []
    prev = Y[0]+1
    for y in Y:
        if y == prev:
            foldedY[len(foldedY)-1] += [y]
        else:
            foldedY += [[y]]
            prev = y
    return foldedY
    
def diffMatrix(Y): # takes a folded sorted list and makes a difference matrix
    # w/ auxiliary variables if needed
#==============================================================================
#   ex: [[3,3],[2,2],[1]] -> |1  0  0  0  0 -1|
#                            |0  1  0  0  0 -1|
#                            |0  0 -1  0  0  1|
#                            |0  0  0 -1  0  1|
#                            |0  0  1  0 -1  0|
#                            |0  0  0  1 -1  0|
#==============================================================================
    rowLen = [0]
    colLen = [0]
    miniDiffMatrices = []
    for ind in range(len(Y)-1):
        aLen = len(Y[ind])
        bLen = len(Y[ind+1])
        if aLen > 1 and bLen > 1: # group > group, need auxiliary variable
            rowLen += [aLen + bLen]
            colLen += [aLen + 1]
            diff = np.zeros((aLen + bLen, aLen + bLen + 1))
            for j in range(aLen):
                diff[j,j] = 1
                diff[j,aLen] = -1
            for j in range(bLen):
                diff[aLen+j,aLen] = 1
                diff[aLen+j,aLen+j+1] = -1
        elif aLen > 1: # group > single number
            rowLen += [aLen]
            colLen += [aLen]
            diff = np.zeros((aLen, aLen+1))
            for j in range(aLen):
                diff[j,j] = 1
                diff[j,aLen] = -1
        else: # single number > group, or single number > single number
            rowLen += [bLen]
            colLen += [1]
            diff = np.zeros((bLen, bLen+1))
            for j in range(bLen):
                diff[j,0] = 1
                diff[j,j+1] = -1
        miniDiffMatrices += [diff]
            
    colLen += [len(Y[len(Y)-1])]

    D = np.zeros((sum(rowLen), sum(colLen)))
    auxVars = []
    rowLen = np.cumsum(rowLen)
    colLen = np.cumsum(colLen)
    
    for k in range(len(miniDiffMatrices)):
        diff = miniDiffMatrices[k]
        rpos = rowLen[k]
        cpos = colLen[k]
        size = diff.shape
        D[rpos:rpos+size[0], cpos:cpos+size[1]] = diff
        auxiliary = np.all(diff!=0, 0) # easy way to check for auxiliaries
        if np.any(auxiliary):
            auxVars += [ cpos + np.argwhere(auxiliary)[0][0] ]
            
    naux = len(auxVars)
    
    # moving auxiliaries to rightmost columns with a permutation matrix
    dcols = D.shape[1]
    Tr = np.eye(dcols)
    for u in range(len(auxVars)): 
        Tr[auxVars[u]+1:dcols,:] = Tr[auxVars[u]:dcols-1,:]
        Tr[auxVars[u],:] = Tr[auxVars[u],:]*0
        Tr[auxVars[u],dcols-naux+u] = 1
    D = np.dot(D,Tr)
    
    return [D, naux]
    

def nDCG(rel):
    rel = rel - min(rel)
    logterm = np.log2(np.arange(len(rel))+1)
    logterm[0]=1
    sortrel = np.sort(rel)
    dcg = np.sum(rel/logterm)
    maxdcg = np.sum(np.flipud(sortrel)/logterm)
    mindcg = np.sum(sortrel/logterm)
    return (dcg - mindcg)/(maxdcg - mindcg)
    
def nDCG2(rel):
    rel = rel - min(rel)
    logterm = np.log2(np.arange(len(rel))+2)
    sortrel = np.sort(rel)
    dcg = np.sum((np.power(2,rel)-1)/logterm)
    maxdcg = np.sum((np.power(2,np.flipud(sortrel))-1)/logterm)
    return dcg/maxdcg
    
def precK(s_true, s_pred, k):
    # s_true: true ranking score (higher the value, higher the rank)
    # s_pred: predicted ranking score
    r_true = np.argsort(-s_true) # ground truth rank based on s_true
    r_pred = np.argsort(-s_pred) # predicted rank based on s_pred

    prc_k = 0
    for i in range(k):
        for j in range(k):
            if r_true[i] == r_pred[j]:
                prc_k = prc_k+1
                break

    prc_k = prc_k/float(k)
    return prc_k