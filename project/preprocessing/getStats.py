# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:19:06 2015

@author: motro
"""

import pandas as pd
import numpy as np
import os


readFolder = os.path.realpath('./mimic_unscoredv1.1')
writeFolder = os.path.realpath('.')
filelist = os.listdir(readFolder)
npatients = len(filelist)

#numericFeatures = [11,15,5,17] # the missing ones in v2, for v1
numericFeatures = range(26)+[27,28,32,33] # v2 and v3

counter = 0
for fileName in filelist[:npatients]:
    #fileName ='tuple_1141_1411.csv'
    
    patient = pd.read_csv(readFolder +'/'+ fileName, sep=',', 
                          header=0, index_col=0)
    for numft in patient.columns.values[numericFeatures]:
        patient[numft][patient[numft]<=0] = np.nan

    if counter == 0:
        maxvals = np.nanmax(patient,axis=0)
        minvals = np.nanmin(patient,axis=0)
        meanvals = np.nanmean(patient,axis=0)
        present = 1 - np.all(np.isnan(patient),axis=0)
        first = False
    else:
        newmeanvals = np.nanmean(patient,axis=0)                     
        newmaxvals = np.nanmax(patient,axis=0)
        newminvals = np.nanmin(patient,axis=0)
        newpresent = 1 - np.all(np.isnan(patient),axis=0)
        maxvals = np.nanmax([maxvals,newmaxvals], axis=0)
        minvals = np.nanmin([minvals,newminvals], axis=0)
        for j in range(len(meanvals)):
            if not (np.isnan(meanvals[j]) or np.isnan(newmeanvals[j])):
                meanvals[j] = (meanvals[j]*counter + newmeanvals[j])/(counter+1)
            elif np.isnan(meanvals[j]):
                meanvals[j] = newmeanvals[j]
        present = present + newpresent
    counter += 1

stats = pd.DataFrame({'min':minvals,'mean':meanvals,'max':maxvals},
                    index=patient.columns).transpose()
stats.to_csv(writeFolder + "/stats.csv",index=True, na_rep="NaN")