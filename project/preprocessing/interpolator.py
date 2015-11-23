# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:25:51 2015

@author: Rahi
"""

import numpy as np
import pandas
#from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
from scipy import interpolate
import os

def interpolator1d (data,stat):

    numericFeatures = range(24)+[25,26,30,31] # v1.2 scored

    for i in numericFeatures:#range (data.shape[1]-2):
        a = data.iloc[:,i]
        b = a.count()
        mx = stat.iloc[0,i+1]
        mn = stat.iloc[2,i+1]
        av = stat.iloc[1,i+1]
        
        if b>2:          
            fi = a[a.notnull()].index[1]
            li = a[a.notnull()].index[-1]
            x = a[a.notnull()].index.values
            y = a[a.notnull()]
            y=np.array(y)
            # linear interpolation
            f = interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True,
                                     bounds_error=False, fill_value=np.nan)
            x_new = a.index.values
            y_new = f(x_new)
            if x[0]>0:       
                mlow= (y[2]-y[1])/(x[2]-x[1])
                y_low= mlow* (x_new[0:x[0]]- x_new[x[0]])+y_new[x[0]]
                y_low[y_low<mn]=mn
                y_low[y_low>mx]=mx
                y_new[0:x[0]]=y_low
            if x[-1]< x_new[-1]:
                mhigh= (y[-2]-y[-1])/(x[-2]-x[-1])
                y_high= mhigh* (x_new[x[-1]+1:]- x_new[x[-1]])+y_new[x[-1]]
                y_high[y_high<mn]=mn
                y_high[y_high>mx]=mx
                y_new[x[-1]+1:]=y_high
        elif b==1:
            y_new = a.copy()
            y_new[:] = a[a.notnull()].iloc[0]
        else:
            y_new = np.zeros(a.shape[0])
            y_new[0:a.shape[0]] = av     
        data.iloc[:,i]=y_new
          
    return data       


readFolder = os.path.realpath('../mimic_scoredv1.2')
writeFolder = os.path.realpath('../mimic_finalv1.2')
stat = pandas.read_csv(os.path.realpath('./stats.csv'),header=0,index_col=0)
filelist = os.listdir(readFolder)
npatients = len(filelist)

for fileName in filelist[:npatients]:
    patient = pandas.read_csv(readFolder + "/" + fileName, header=0,
                              index_col=False)
    patient = interpolator1d(patient,stat)
    
    patient.drop('pco2', axis=1, inplace = True) # currently completely missing
    
    patient.to_csv(writeFolder + "/" + fileName, index=False,na_rep='NaN') 