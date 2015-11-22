# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:06:12 2015

@author: twankim
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as lm

dataPath = "data/tuple_"

def readTuple(tName): # tName: Name of tuple
    tData = pd.read_csv(dataPath+tName)
    return tData
    
# Augment trendFeatures to Risk Scores
# xScore: score column of a patient
def trendFeature(xScore):
    tLength = np.size(xScore)
    ts = np.arange(1.0,tLength+1) # time steps
    ts2 = np.power(ts,2)
    tsDiff = np.diff(ts)
    tsDiff = np.insert(tsDiff, 0, tsDiff[0])
    scoreDiff = np.diff(xScore)
    scoreDiff = np.insert(scoreDiff, 0, scoreDiff[0])

    tFeature = np.zeros((tLength,8)) # Augmented features
    tFeature[:,0] = xScore
    for i in range(tLength):
        tempScore = xScore.iloc[:i+1]
        tempSdiff = scoreDiff[:i+1]
        tempTs = ts[:i+1]
        tempTdiff = tsDiff[:i+1]
        tempTs2 = ts2[:i+1]
        
        # Average score since admission
        tFeature[i,1] = np.sum(np.multiply(tempTdiff,tempScore))/tempTs[-1]
        
        # Linear weighted score
        tFeature[i,2] = np.sum(np.multiply(tempTs,tempScore))/np.sum(tempTs)
        
        # Quadratic weighted score
        tFeature[i,3] = np.sum(np.multiply(tempTs2,tempScore))/np.sum(tempTs2)
        
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
    
    return tFeature
    
def 