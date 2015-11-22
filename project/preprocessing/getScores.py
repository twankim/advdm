# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

'''
takes feature data and adds two columns: sepsis scores and treatment
scores : 3 = septic shock, 2 = severe, 1 = sepsis, 0 = no sepsis
returns NaN if conditions not met
treatment = 1 if treatment started here, 0 otherwise (no NaNs)
hourly data assumed

last update: 11/22 morning
'''

def getLatestMeasurement(data, feature, currentTime, limit):
    dfeature = data[feature]
    dtime = dfeature[max(currentTime-limit,0):currentTime+1]
    if np.all(np.isnan(dtime)):
        return np.nan
    return dtime.loc[dtime.last_valid_index()]


def hasSIRS(heart_rate,temp,resp_rate,PaCO2,WBC):
    counter = 0
    if not np.isnan(heart_rate) and heart_rate > 90:
        counter += 1
    if not np.isnan(temp) and (temp > 38 or temp < 36):
        counter += 1
    if not np.isnan(temp) and resp_rate > 2:
        counter += 1
    elif not np.isnan(PaCO2) and PaCO2 < 32:
        counter += 1
    if not np.isnan(WBC) and (WBC > 12 or WBC < 4):
        counter += 1
    return counter > 1
        
    
def hasSevereSepsis(sysBP, bloodLactate, urine, chronicRenal, creatinine,
                    chronicLiver, bilirubin, platelet, INR, pneumonia, PaO2,
                    FiO2, acuteLungInfect, wordContains):
    if not wordContains:
        return False
    counter = False
    if not np.isnan(sysBP) and sysBP < 90:
        counter = True
    if not np.isnan(bloodLactate) and bloodLactate > 2:
        counter = True
    if not np.isnan(urine) and urine < .5:
        counter = True
    if (not np.isnan(chronicRenal) and not np.isnan(creatinine) and
            chronicRenal and creatinine > 2):
        counter = True
    if (not np.isnan(chronicLiver) and not np.isnan(bilirubin) and
            chronicLiver and bilirubin > 2):
        counter = True
    if not np.isnan(platelet) and platelet < 100:
        counter = True
    if not np.isnan(INR) and INR > 1.5:
        counter = True
    if (not np.isnan(pneumonia) and not np.isnan(PaO2) and not np.isnan(FiO2) and
            pneumonia and PaO2/FiO2 < 200):
        counter = True
    if not (np.isnan(acuteLungInfect) or np.isnan(pneumonia) or np.isnan(PaO2) or
            np.isnan(FiO2)) and (acuteLungInfect and not pneumonia and
            PaO2/FiO2 < 250):
        counter = True
    return counter

    
def getScores(pdata):
    score = []
    thisscore = 0    
    
    chronicRenal = pdata.loc[0,'cri']
    chronicLiver = pdata.loc[0,'cld']
    pneumonia = pdata.loc[0,'pneumonia']
    acuteLungInfect = pdata.loc[0,'ali']
    
    for time in range(pdata.shape[0]):
        
        heart_rate = getLatestMeasurement(pdata,'heartrate',time,2)
        temp = getLatestMeasurement(pdata,'temp',time,8)
        resp_rate = getLatestMeasurement(pdata,'resp_rate',time,2)
        PaCO2 = getLatestMeasurement(pdata,'paco2',time,8)
        WBC = getLatestMeasurement(pdata,'wbc',time,8)
        
        if hasSIRS(heart_rate,temp,resp_rate,PaCO2,WBC):
            
            SysBP = getLatestMeasurement(pdata,'abp',time,2)
            bloodLactate = getLatestMeasurement(pdata,'bloodlactate',time,2)
            urine = getLatestMeasurement(pdata,'urine_output',time,2)
            creatinine = getLatestMeasurement(pdata,'creatinine',time,8)
            bilirubin = getLatestMeasurement(pdata,'bilirubin',time,8)
            platelet = getLatestMeasurement(pdata,'platelet_cnt',time,8)
            PaO2 = getLatestMeasurement(pdata,'pao2',time,8)
            FiO2 = getLatestMeasurement(pdata,'fio2',time,8)
            INR = getLatestMeasurement(pdata,'inr',time,8)                                   
            
            if hasSevereSepsis(SysBP, bloodLactate, urine, chronicRenal,
                               creatinine, chronicLiver, bilirubin, platelet,
                               INR, pneumonia, PaO2, FiO2, acuteLungInfect,
                               wordContains=True):
                                   # for now ignoring clinical notes
                lastSysBP = getLatestMeasurement(pdata,'abp',time-1,2)
                if (thisscore >= 2 and not np.isnan(SysBP) and
                        SysBP < 90 and lastSysBP < 90):
                    thisscore = 3 # septic shock
                    
                else:
                    thisscore = 2 # severe sepsis            
            else:
                thisscore = 1 # SIRS
        elif not np.any(np.isnan([heart_rate,temp,resp_rate,PaCO2,WBC])):
            thisscore = 0 # no sepsis
        else:
            thisscore = -1 # not enough info
            
        score += [thisscore]
    
    return np.array(score, dtype='float')
    
    
def getTreatment(pdata, sumA, sumB):
    fluidtreat = pdata['fluidbolus']*pdata['score'] > 0
    treatment = np.zeros(fluidtreat.shape)
    #sumA += sum(pdata['fluidbolus']>0)
    sumA += sum(fluidtreat)
    for time in (ind for (ind,z) in enumerate(fluidtreat) if z):
        if not np.any(treatment[max(0,time-4):time]>0): # first in 5 hours
            sumB += 1
            if getLatestMeasurement(pdata,'abp',time,5) < 100:
                 treatment[time] = 1
    return (treatment, sumA ,sumB)
        
        
    
# apply to data
readFolder = os.path.realpath('./mimic_unscoredv1.1')
writeFolder = os.path.realpath('./mimic_scoredv1.1')
filelist = os.listdir(readFolder)
npatients = len(filelist)

numericFeatures = range(26)+[27,28,32,33] # for v0.2

patientsWithScore = 0.
patientsWithSepsis = 0.
timesWithScore = 0.
timesWithSepsis = 0.
totaltimes = 0.
numTreatments = 0.
sumBoluses = 0
sumTreats = 0 
for fileName in filelist[:npatients]:
    #fileName ='tuple_1141_1411.csv'
    
    patient = pd.read_csv(readFolder +'/'+ fileName, sep=',', 
                          header=0, index_col=0)
    
    # transform all 0 values in numeric features to nans                       
    for numft in patient.columns.values[numericFeatures]:
        patient[numft][patient[numft]<=0] = np.nan
        
    score = getScores(patient)
    
    totaltimes += patient.shape[0]
    timesWithScore += np.sum(score>=0)
    timesWithSepsis += np.sum(score>0)
    if np.any(score>=0):
        patientsWithScore += 1
    if np.any(score>0):
        patientsWithSepsis += 1
    
    score[score<0] = np.nan
    patient['score'] = score
    
    treatment, sumBoluses, sumTreats = getTreatment(patient, sumBoluses, sumTreats)   
    patient['treatment'] = treatment
    numTreatments += sum(treatment)
    patient = patient.drop('fluidbolus',1)
    
    #patient.to_csv(writeFolder + "/" + fileName, index=False, na_rep='NaN')

print ('percent patients with score: ' +
            str(patientsWithScore/npatients) + '\n' +
       'percent scored patients with sepsis: ' +
           str(patientsWithSepsis/patientsWithScore) + '\n' +
       'percent times with score: ' + str(timesWithScore/totaltimes) + '\n' +
       'percent scored times with sepsis: ' + 
           str(timesWithSepsis/timesWithScore) + '\n' + 
       'number of treatments: ' + str(numTreatments))
