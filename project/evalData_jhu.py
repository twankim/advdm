# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:06:12 2015

@author: twankim
"""
# Modified version of evalData for jhu_Data

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import preprocessing as pp
    
# Augment trendFeatures to Risk Scores (For Predicting Septic Shock)
# xScore: score column of a patient
def trendFeature(pScore):
    tLength = np.shape(pScore)[0]
    ts = np.array(pScore['time']) # time steps
    ts = ts - ts[0] + 1 # Set starting time as 1
    ts2 = np.power(ts,2) # t^2

    # time difference
    tsDiff = np.diff(ts)
    if len(tsDiff) == 0:
        tsDiff = ts
    else:
        tsDiff = np.insert(tsDiff, 0, tsDiff[0])

    # score difference
    xScore = np.array(pScore['score'])
    scoreDiff = np.diff(xScore)
    if len(scoreDiff) == 0:
        scoreDiff = xScore
    else:
        scoreDiff = np.insert(scoreDiff, 0, scoreDiff[0])

    tFeature = np.zeros((tLength,7)) # Augmented features
    for i in range(tLength):
        tempScore = xScore[:i+1]
        tempSdiff = scoreDiff[:i+1]
        tempTs = ts[:i+1]
        tempTdiff = tsDiff[:i+1]
        tempTs2 = ts2[:i+1]
        
        # Average score since admission
        tFeature[i,0] = np.sum(np.multiply(tempTdiff,tempScore))/tempTs[-1]
        
        # Linear weighted score
        tFeature[i,1] = np.sum(np.multiply(tempTs,tempScore))/np.sum(tempTs)
        
        # Quadratic weighted score
        tFeature[i,2] = np.sum(np.multiply(tempTs2,tempScore))/np.sum(tempTs2)
        
        if i == 0:            
            # Average score change rate
            tFeature[i,3] = 0
            
            # Linearly weighted average score change rate
            tFeature[i,4] = 0
        
            # Average absolute score change rate
            tFeature[i,5] = 0
        
            # Linearly weighted absolute score change rate
            tFeature[i,6] = 0
        else:
            # Average score change rate
            tFeature[i,3] = (tempScore[-1]-tempScore[0])/(tempTs[-1]-tempTs[0])
            
            # Linearly weighted average score change rate
            tFeature[i,4] = np.sum(np.multiply(tempSdiff,tempTdiff))/(tempTs[-1]-tempTs[0])
        
            # Average absolute score change rate
            tFeature[i,5] = np.sum(np.abs(tempSdiff))/(tempTs[-1]-tempTs[0])
        
            # Linearly weighted absolute score change rate
            tFeature[i,6] = np.sum(np.multiply(np.abs(tempSdiff),tempTdiff))/(tempTs[-1]-tempTs[0])
    
    return tFeature

# Calculate score of data using given parameter and data with different mode    
def scoreCalc(wData, w_sol, mode):
    if mode == 1:
        if len(w_sol[0].params) > wData.shape[1]:
            wData = np.concatenate((wData, np.ones((wData.shape[0],1))),axis=1)
        scoreData = w_sol[0].predict(wData, transform = False)
    else: # inner product
        if len(w_sol) > wData.shape[1]:
            wData =np.concatenate((wData, np.ones((wData.shape[0],1))),axis=1)
        scoreData = np.dot(wData,w_sol)
        
    return scoreData
        
# Xfeature: Original features only
def makeDataShockFeat(fileName):
    # Read csv file and find patient ID lists
    wholeData = pd.read_csv(fileName)

    # Original features only
    Xfeature = wholeData.drop(['time'],axis=1)
    
    return Xfeature

# Read csv file and make 3 different data
# Xscore: score only
# Xderived: score + derived 7 derived features
# Each data will include 'ID', 'truth' additionally            
def makeDataShock(fileName,w_sol,mode):
    # Read csv file and find patient ID lists
    wholeData = pd.read_csv(fileName)
    idWhole = np.array(wholeData['ID'])
    idList, idx = np.unique(wholeData['ID'], return_index=True)
    idList = idWhole[np.sort(idx)]
    
    # Calculate scores for each data row using learend parameter w_sol
    wData = np.array(wholeData.drop(['time','ID','truth'],axis=1))
    wholeData['score'] = scoreCalc(wData,w_sol,mode)

    # Data type 1: score only
    Xscore = wholeData[['score','ID','truth']]
    # Data type 3: Original features only
    
    # Data type 2: score + derived 7 derived features
    # Add 7 features to Xderived
    for i in range(len(idList)):
        tempData = wholeData.loc[wholeData['ID']==idList[i]]
        pScore = tempData[['score','time']]
        if i == 0:
            tFeature = trendFeature(pScore)
        else:
            tFeature = np.concatenate((tFeature,trendFeature(pScore)))

    Xderived = pd.DataFrame(tFeature,columns=['tfeat1','tfeat2','tfeat3','tfeat4','tfeat5','tfeat6','tfeat7'])
    Xderived['score'] = wholeData['score']    
    Xderived['ID'] = wholeData['ID']
    Xderived['truth'] = wholeData['truth']
            
    return Xscore, Xderived
    
# Calculate prediction scoure (AUC) from Train/Test data
def septicPredict(dataTrain,dataTest):
    # Make Train/Test data for learning
    Xtrain = dataTrain.drop(['ID','truth'],axis=1)
    Ytrain = dataTrain['truth']
    Xtest = dataTest.drop(['ID','truth'],axis=1)
    Ytest = dataTest['truth']

    # Make patient ID list
    pid = np.array(dataTest['ID'])
    pidUniq, idx = np.unique(pid, return_index=True)
    pidUniq = pid[np.sort(idx)]
    
    # Scale data with standard scaler
    std_scaler = pp.StandardScaler()
    Xtrain = std_scaler.fit_transform(Xtrain)
    Xtest = std_scaler.transform(Xtest)
    
    # Use logistic regression to learn and predict
    lmlogit = lm.LogisticRegression('l2', C=1)
    lmlogit.fit(Xtrain,Ytrain)
    
    Yscore = np.zeros(len(pidUniq))
    scoreTemp = lmlogit.predict_proba(Xtest)[:,1]
    Ytrue = np.zeros(len(pidUniq))
    
    # Make Ytrue and Yscore patient-wise
    # At least one point in one patient data exceeds threshold -> positive
    # Sample max point of scores for each patient
    for i in range(len(pidUniq)):
        temp = pid == pidUniq[i]
        idxTemp = np.where(temp)[0]
        Yscore[i] = np.max(scoreTemp[idxTemp])
        Ytrue[i] = Ytest[idxTemp[0]] # Just select first truth element of patient
    
    return Ytrue, Yscore
    
def plotAccLift(Ytrue, Yscore):
    xRatio = np.arange(0,1.01,0.01)
    yRatio = np.zeros(np.shape(xRatio))
    nPos = np.sum(Ytrue == 1)
    nTot = len(Ytrue)
    
    idx = np.argsort(-Yscore)
    YtrueSorted = Ytrue[idx]
    
    for i in np.arange(1,len(xRatio)):
        rr = xRatio[i]
        nR = int(np.floor(rr*nTot))
        trueTemp = YtrueSorted[:nR]
        yRatio[i] = np.sum(trueTemp == 1)/(nPos*rr)
        
    return xRatio,yRatio