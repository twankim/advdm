# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:06:12 2015

@author: twankim
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
#import sklearn.metrics as met
from sklearn import preprocessing as pp
from os import listdir
import random

dataPath = "/Users/twankim/tdub/gitwan/advdm/project/interpolatedv1.1"
headerFile = "feature_header.csv" # File for feature header

def readTuple(tName,mode): # tName: Name of tuple file
    if mode == 1: # csv doesn't include feature header
        tData = pd.read_csv(dataPath+"/"+ tName, header = None, index_col = 0)
        temp = pd.read_csv(headerFile)
        headerFeat = temp.columns
        tData.columns = headerFeat
    else: # csv includes header
        tData = pd.read_csv(dataPath+"/"+ tName)
    return tData
    
# Augment trendFeatures to Risk Scores (For Predicting Septic Shock)
# xScore: score column of a patient
def trendFeature(xScore):
    tLength = np.size(xScore)
    ts = xScore.index.values.tolist() # time steps
    ts = [x+1 for x in ts] # to avoid 0 value
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
    
def makeDataShock(w_sol):
    dataPos = pd.DataFrame() # Positive sample
    dataNeg = pd.DataFrame() # Negative sample
    pidPos = pd.DataFrame() # Unique patient id
    pidNeg = pd.DataFrame() # Unique patient id

    flist = listdir(dataPath)
    numP = 0
    for i in range(len(flist)):
        if flist[i].startswith('tuple') & flist[i].endswith('csv'):
            pData = readTuple(flist[i],1)
            pData_whole = pData
            pData.drop(np.where(np.isnan(pData['score']))[0], axis=0, inplace=True)
            pData.drop(['riker_sas','pco2'],axis=1, inplace=True)
            if (np.sum(pData_whole['treatment'] == 1) == 0) & (np.sum(pData['score'] == 3) == 0):
                dataNeg = dataNeg.append(pData)
                pidTemp1 = pd.DataFrame({"numP": [numP]*(len(pData))})
                pidNeg = pidNeg.append(pidTemp1)
                numP = numP+1
            elif np.sum(pData['score']==3) > 0:
                tempData = pData.score[pData.score==3].index.tolist()
                idxS = tempData[0]-1
                tpData = pData.ix[:idxS]
                idxS2 = len(tpData.index)
                if len(tpData.index) > 48:
                    tpData.ix[tpData.index[idxS2-48:idxS2]]
                dataPos = dataPos.append(tpData)
                pidTemp2 = pd.DataFrame({"numP": [numP]*(len(tpData))})
                pidPos = pidPos.append(pidTemp2)
                numP = numP+1           
#    labelPos = np.ones((len(dataPos.index),1))
#    labelNeg = np.zeros((len(dataNeg.index),1))
    
#    return dataPos, dataNeg, pidPos, pidNeg, labelPos, labelNeg
    
    pidUniqPos = np.unique(pidPos)
    nPidUniqPos = int(np.floor(len(pidUniqPos)*0.7))
    pidUniqPosTrain = np.sort(random.sample(pidUniqPos,nPidUniqPos))
    pidUniqPosTest = np.sort(filter(lambda x: x not in pidUniqPosTrain,pidUniqPos))

#    trainIdxPos = pidPos == pidUniqPosTrain[0]
#    for i in np.arange(1,len(pidUniqPosTrain)):
#        trainIdxPos= trainIdxPos | (pidPos == pidUniqPosTrain[i])
#    testIdxPos = ~trainIdxPos
    
    pidUniqNeg = np.unique(pidNeg)
    nPidUniqNeg = int(np.floor(len(pidUniqNeg)*0.7))
    pidUniqNegTrain = np.sort(random.sample(pidUniqNeg,nPidUniqNeg))
    pidUniqNegTest = np.sort(filter(lambda x: x not in pidUniqNegTrain,pidUniqNeg))
    
#    trainIdxNeg = pidNeg == pidUniqNegTrain[0]
#    for i in np.arange(1,len(pidUniqNegTrain)):
#        trainIdxNeg= trainIdxNeg | (pidNeg == pidUniqNegTrain[i])
#    testIdxNeg = ~trainIdxNeg
    
    xPos = np.array(dataPos.drop(["score","treatment"],axis=1))
    xNeg = np.array(dataNeg.drop(["score","treatment"],axis=1))
    scorePos = pd.DataFrame(np.dot(xPos,w_sol),index = dataPos.index.values.tolist())
    scoreNeg = pd.DataFrame(np.dot(xNeg,w_sol),index = dataNeg.index.values.tolist())
    
    Xtrain = pd.DataFrame()
    pidTrain = pd.DataFrame()
    
    for i in np.arange(0,len(pidUniqPosTrain)):
        temp = np.array(pidPos == pidUniqPosTrain[i])
        temp = np.where(temp)[0]
        Xtrain = Xtrain.append(trendFeature(scorePos[temp[0]:temp[-1]+1]))
        pidTrain = pidTrain.append(pidPos[temp[0]:temp[-1]+1])
        
    Ytrain_temp = np.ones((len(pidTrain),1))
    
    for i in np.arange(0,len(pidUniqNegTrain)):
        temp = np.array(pidNeg == pidUniqNegTrain[i])
        temp = np.where(temp)[0]
        Xtrain = Xtrain.append(trendFeature(scoreNeg[temp[0]:temp[-1]+1]))
        pidTrain = pidTrain.append(pidNeg[temp[0]:temp[-1]+1])
       
    Xtrain = np.array(Xtrain)
    pidTrain = np.array(pidTrain)
    Ytrain = np.zeros((len(pidTrain),1))
    Ytrain[0:len(Ytrain_temp)] = Ytrain_temp
    
    Xtest = pd.DataFrame()
    pidTest = pd.DataFrame()
    
    for i in np.arange(0,len(pidUniqPosTest)):
        temp = np.array(pidPos == pidUniqPosTest[i])
        temp = np.where(temp)[0]
        Xtest = Xtest.append(trendFeature(scorePos[temp[0]:temp[-1]+1]))
        pidTest = pidTest.append(pidPos[temp[0]:temp[-1]+1])
        
    Ytest_temp = np.ones((len(pidUniqPosTest),1))
    
    for i in np.arange(0,len(pidUniqNegTest)):
        temp = np.array(pidNeg == pidUniqNegTest[i])
        temp = np.where(temp)[0]
        Xtest = Xtest.append(trendFeature(scoreNeg[temp[0]:temp[-1]+1]))
        pidTest = pidTest.append(pidNeg[temp[0]:temp[-1]+1])
       
    Xtest = np.array(Xtest)
    pidTest = np.array(pidTest)
    Ytest = np.concatenate((Ytest_temp,np.zeros((len(pidUniqNegTest),1))))
    
    # Ytrain is for whole data
    # Ytest is for patient-wise data    
    
    return Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest
    
def septicPredict(Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest):
    Ytrain = np.reshape(Ytrain,np.shape(Ytrain)[0])
    Ytest = np.reshape(Ytest,np.shape(Ytest)[0])
    
    std_scaler = pp.StandardScaler()
    Xtrain = std_scaler.fit_transform(Xtrain)
    Xtest = std_scaler.transform(Xtest)

    lmlogit = lm.LogisticRegression('l2', C=1)
    lmlogit.fit(Xtrain,Ytrain)

    scorePredict = np.zeros(np.shape(Ytest))
    indices = np.unique(pidTest, return_index = True)[1]
    pidUniqTest = [pidTest[idx] for idx in sorted(indices)]

    scoreTemp = lmlogit.predict_proba(Xtest)[:,1]

    for i in range(len(pidUniqTest)):
        temp = np.array(pidTest == pidUniqTest[i])
        scorePredict[i] = np.max(scoreTemp[np.where(temp)[0]])
        
    return scorePredict
    