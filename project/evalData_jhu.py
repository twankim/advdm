# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:06:12 2015

@author: twankim
"""
# Modified version of evalData for jhu_Data

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
#import sklearn.metrics as met
from sklearn import preprocessing as pp
import random
    
# Augment trendFeatures to Risk Scores (For Predicting Septic Shock)
# xScore: score column of a patient
def trendFeature(xScore):
    tLength = np.size(xScore)
    ts = xScore.index.values.tolist() # time steps
    ts =  [x - ts[0]+1 for x in ts] # Set starting time as 1
#    ts = [x+1 for x in ts] # to avoid 0 value
    ts2 = np.power(ts,2) # t^2

    # time difference
    tsDiff = np.diff(ts)
    if len(tsDiff) == 0:
        tsDiff = ts
    else:
        tsDiff = np.insert(tsDiff, 0, tsDiff[0])

    # score difference
    scoreDiff = np.diff(xScore[0])
    if len(scoreDiff) == 0:
        scoreDiff = xScore[0]
    else:
        scoreDiff = np.insert(scoreDiff, 0, scoreDiff[0])

    tFeature = np.zeros((tLength,8)) # Augmented features
    tFeature[:,0] = np.transpose(np.array(xScore))
    for i in range(tLength):
        tempScore = xScore.iloc[:i+1]
        tempSdiff = scoreDiff[:i+1]
        tempTs = ts[:i+1]
        tempTdiff = tsDiff[:i+1]
        tempTs2 = ts2[:i+1]
        
        # Average score since admission
        tFeature[i,1] = np.sum(np.multiply(tempTdiff,tempScore[0]))/tempTs[-1]
        
        # Linear weighted score
        tFeature[i,2] = np.sum(np.multiply(tempTs,tempScore[0]))/np.sum(tempTs)
        
        # Quadratic weighted score
        tFeature[i,3] = np.sum(np.multiply(tempTs2,tempScore[0]))/np.sum(tempTs2)
        
        if i == 0:            
            # Average score change rate
            tFeature[i,4] = 0
            
            # Linearly weighted average score change rate
            tFeature[i,5] = 0
        
            # Average absolute score change rate
            tFeature[i,6] = 0
        
            # Linearly weighted absolute score change rate
            tFeature[i,7] = 0
        else:
            # Average score change rate
            tFeature[i,4] = (tempScore.iloc[-1]-tempScore.iloc[0])/(tempTs[-1]-tempTs[0])
            
            # Linearly weighted average score change rate
            tFeature[i,5] = np.sum(np.multiply(tempSdiff,tempTdiff))/(tempTs[-1]-tempTs[0])
        
            # Average absolute score change rate
            tFeature[i,6] = np.sum(np.abs(tempSdiff))/(tempTs[-1]-tempTs[0])
        
            # Linearly weighted absolute score change rate
            tFeature[i,7] = np.sum(np.multiply(np.abs(tempSdiff),tempTdiff))/(tempTs[-1]-tempTs[0])

    tFeature = pd.DataFrame(tFeature,index = ts)   
    
    return tFeature
    
def makePosNegShock(fileName):
    dataPos = pd.DataFrame() # Positive sample
    dataNeg = pd.DataFrame() # Negative sample
    pidPos = pd.DataFrame() # Unique patient id
    pidNeg = pd.DataFrame() # Unique patient id

    wholeData = pd.read_csv(fileName)
    idList = np.unique(wholeData['ID'])
    # Read data file from directory

    numP = 0 # Label patient number (numP) to be included in Train & Test data
    for i in range(len(idList)):
        # Select data with one patient ID
        pData = wholeData.loc[wholeData['ID']==idList[i]]
        
        if np.sum(pData['truth'] == 1) == 0:
            # Get negative samples (No treatment, no septic shock)
            dataNeg = dataNeg.append(pData)
            # label time data with numP
            pidTemp1 = pd.DataFrame({"numP": [numP]*(len(pData))})
            pidNeg = pidNeg.append(pidTemp1)
            numP = numP+1 # increase patient number (numP)
        elif np.sum(pData['truth']==1) > 0:
            # Get positive samples (septic shock occured)
            dataPos = dataPos.append(pData)
            pidTemp2 = pd.DataFrame({"numP": [numP]*(len(pData))})
            pidPos = pidPos.append(pidTemp2)
            numP = numP+1 # increase patient number (numP) 
    
    return dataPos, dataNeg, pidPos, pidNeg
    
def makeDataShock(dataPos, dataNeg, pidPos, pidNeg, w_sol):
    # Divide positive samples into train & test based on patient id
    pidUniqPos = np.unique(pidPos)
    nPidUniqPos = int(np.floor(len(pidUniqPos)*0.7))
    pidUniqPosTrain = np.sort(random.sample(pidUniqPos,nPidUniqPos))
    pidUniqPosTest = np.sort(filter(lambda x: x not in pidUniqPosTrain,pidUniqPos))
    
    # Divide negative samples into train & test based on patient id
    pidUniqNeg = np.unique(pidNeg)
    nPidUniqNeg = int(np.floor(len(pidUniqNeg)*0.7))
    pidUniqNegTrain = np.sort(random.sample(pidUniqNeg,nPidUniqNeg))
    pidUniqNegTest = np.sort(filter(lambda x: x not in pidUniqNegTrain,pidUniqNeg))
    
    # Calculate scores for each data row using learend parameter w_sol
    timePos = dataPos['time']
    timeNeg = dataNeg['time']
    xPos = np.array(dataPos.drop(["time","ID","truth"],axis=1))
    xNeg = np.array(dataNeg.drop(["time","ID","truth"],axis=1))
    scorePos = pd.DataFrame(np.dot(xPos,w_sol),index = timePos.tolist())
    scoreNeg = pd.DataFrame(np.dot(xNeg,w_sol),index = timeNeg.tolist())
    
    # ------------------------------ Train Data ------------------------------
    # Make Train data for scores
    Xtrain = pd.DataFrame()
    pidTrain = pd.DataFrame()
    idxTrain2 = np.array([]) # index for Training set
    
    # Extract 8 features and append to train data based on separated patient ids
    # Positive samples
    for i in np.arange(0,len(pidUniqPosTrain)):
        temp = np.array(pidPos == pidUniqPosTrain[i])
        temp = np.where(temp)[0]
        idxTrain2 = np.concatenate((idxTrain2,temp))
        Xtrain = Xtrain.append(trendFeature(scorePos[temp[0]:temp[-1]+1]))
        pidTrain = pidTrain.append(pidPos[temp[0]:temp[-1]+1])
        
    Ytrain_temp = np.ones(len(pidTrain))
    Xtrain2 = xPos[idxTrain2.astype(int),:] # Dataset for feature based prediction    
    idxTrain2 = np.array([]) # clean idx for negative sample
    
    # Negative samples
    for i in np.arange(0,len(pidUniqNegTrain)):
        temp = np.array(pidNeg == pidUniqNegTrain[i])
        temp = np.where(temp)[0]
        idxTrain2 = np.concatenate((idxTrain2,temp))
        Xtrain = Xtrain.append(trendFeature(scoreNeg[temp[0]:temp[-1]+1]))
        pidTrain = pidTrain.append(pidNeg[temp[0]:temp[-1]+1])
       
    Xtrain = np.array(Xtrain) # Transform to array type
    pidTrain = np.array(pidTrain) # Transform to array type
    Ytrain = np.zeros(len(pidTrain))
    Ytrain[0:len(Ytrain_temp)] = Ytrain_temp # 0,1 label (0: no shock/ 1: shock)
    
    Xtrain2 = np.concatenate((Xtrain2,xNeg[idxTrain2.astype(int),:]))
        
    # ------------------------------ Test Data ------------------------------
    # Make Test data for scores
    Xtest = pd.DataFrame()
    pidTest = pd.DataFrame()
    idxTest2 = np.array([]) # index for Test set
    
    # Extract 8 features and append to test data based on separated patient ids
    # Positive samples
    for i in np.arange(0,len(pidUniqPosTest)):
        temp = np.array(pidPos == pidUniqPosTest[i])
        temp = np.where(temp)[0]
        idxTest2 = np.concatenate((idxTest2,temp))
        Xtest = Xtest.append(trendFeature(scorePos[temp[0]:temp[-1]+1]))
        pidTest = pidTest.append(pidPos[temp[0]:temp[-1]+1])
        
    Ytest_temp = np.ones(len(pidUniqPosTest))
    Xtest2 = xPos[idxTest2.astype(int),:] # Dataset for feature based prediction
    idxTest2 = np.array([]) # clean idx for negative sample
    
    # Negative samples
    for i in np.arange(0,len(pidUniqNegTest)):
        temp = np.array(pidNeg == pidUniqNegTest[i])
        temp = np.where(temp)[0]
        idxTest2 = np.concatenate((idxTest2,temp))
        Xtest = Xtest.append(trendFeature(scoreNeg[temp[0]:temp[-1]+1]))
        pidTest = pidTest.append(pidNeg[temp[0]:temp[-1]+1])
       
    Xtest = np.array(Xtest) # Transform to array type
    pidTest = np.array(pidTest) # Transform to array type
    Ytest = np.concatenate((Ytest_temp,np.zeros(len(pidUniqNegTest)))) # 0,1 label (0: no shock/ 1: shock)
    
    Xtest2 = np.concatenate((Xtest2,xNeg[idxTest2.astype(int),:]))
    
    # Ytrain is for whole data
    # Ytest is for patient-wise data    
    
    return Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest, Xtrain2, Xtest2
    
def septicPredict(Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest, Xtrain2, Xtest2):    
    std_scaler = pp.StandardScaler()
    Xtrain = std_scaler.fit_transform(Xtrain)
    Xtest = std_scaler.transform(Xtest)
    
    std_scaler2 = pp.StandardScaler()
    Xtrain2 = std_scaler2.fit_transform(Xtrain2)
    Xtest2 = std_scaler2.transform(Xtest2) 

    lmlogit = lm.LogisticRegression('l2', C=1)
    lmlogit.fit(Xtrain,Ytrain)
    
    lmlogit2 = lm.LogisticRegression('l2', C=1)
    lmlogit2.fit(Xtrain2,Ytrain)

    indices = np.unique(pidTest, return_index = True)[1]
    pidUniqTest = [pidTest[idx] for idx in sorted(indices)]

    scoreTempMR = lmlogit.predict_proba(Xtest)[:,1]
    scoreTempFeat = lmlogit2.predict_proba(Xtest2)[:,1]
    
    # calculate TPR and FPR of two different methods
    thres = np.arange(0,1.01,0.01) # Threshold
    tprMR = np.zeros(len(thres))
    fprMR = np.zeros(len(thres))
    tprFeat = np.zeros(len(thres))
    fprFeat = np.zeros(len(thres))
    nPos = float(np.sum(Ytest == 1))
    nNeg = float(np.sum(Ytest == 0))
    
    for ithre in range(len(thres)):
        thre = thres[ithre]
        tpMR = 0
        fpMR = 0
        tpFeat = 0
        fpFeat = 0
        for i in range(len(pidUniqTest)):
            temp = np.array(pidTest == pidUniqTest[i])
            tempMR = scoreTempMR[np.where(temp)[0]]
            tempFeat = scoreTempFeat[np.where(temp)[0]]
            
            # MR
            # TP: at least one score is greater than threshold & had septic shock
            # FP: at least one score is greater than threshold & never had septic shock
            if (np.max(tempMR)>=thre) & (Ytest[i] == 1):
                tpMR = tpMR + 1
            elif (np.sum(tempMR>=thre)>0) & (Ytest[i] == 0):
                fpMR = fpMR + 1
            if (np.max(tempFeat)>=thre) & (Ytest[i] == 1):
                tpFeat = tpFeat + 1
            elif (np.sum(tempFeat>=thre)>0) & (Ytest[i] == 0):
                fpFeat = fpFeat + 1
            
        tprMR[ithre] = tpMR / nPos
        fprMR[ithre] = fpMR / nNeg
        tprFeat[ithre] = tpFeat / nPos
        fprFeat[ithre] = fpFeat / nNeg
        
    return tprMR, fprMR, tprFeat, fprFeat
    