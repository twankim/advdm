# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:22:36 2015

@author: twankim
"""

from evalData_jhu import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

fileTrain = '/Users/twankim/tdub/gitwan/advdm/project/jhu_data/predictDataTrain.csv'
fileTest = '/Users/twankim/tdub/gitwan/advdm/project/jhu_data/predictDataTest.csv'

saveFlagFeat = True
#saveFlagFeat = False

w_sol1 = np.load('sol_1.npy') # Linear Regression
w_sol4 = np.load('sol_4.npy') # MR_Squared Equlidian
w_sol6 = np.load('sol_6.npy') # MR I-divergence

mode1 = 0 # Selecting score function
mode4 = 0
mode6 = 1

w_sols = [w_sol1, w_sol4, w_sol6]
modes = [mode1,mode4,mode6]
nameNs = [1,4,6]
saveFlags = [True,True,True]
#saveFlags = [False,False,False]

print "<Feature Only>"
if saveFlagFeat:
    print "----- Making Training sample data..."
    XfeatureTrain = makeDataShockFeat(fileTrain)
    print "----- Making Test sample data..."
    XfeatureTest = makeDataShockFeat(fileTest)
    np.save('trainFeat',XfeatureTrain)
    np.save('testFeat',XfeatureTest)
else:
    print "----- Loading Training sample data..."
    XfeatureTrain = np.load('trainFeat.npy')
    print "----- Loading Test sample data..."
    XfeatureTest = np.load('testFeat.npy')

print "----- Predicting septic shock..."
YtrueFeat, YscoreFeat = septicPredict(XfeatureTrain,XfeatureTest)

YtrueMRs = [None]*len(w_sols)
YscoreMRs = [None]*len(w_sols)
YtrueMRderiveds = [None]*len(w_sols)
YscoreMRderiveds = [None]*len(w_sols)

for i in range(len(w_sols)):
    print "<Method " + str(i+1) + ">"
    w_sol = w_sols[i]
    mode = modes[i]
    nameN = nameNs[i]
    saveFlag = saveFlags[i]
    
    if saveFlag:
        print "----- Making Training sample data..."
        # Make data from Training set
        XscoreTrain, XderivedTrain = makeDataShock(fileTrain,w_sol,mode)
        print "----- Making Test sample data..."
        # Make data from Test set
        XscoreTest, XderivedTest = makeDataShock(fileTest,w_sol,mode)
        
        np.save('trainTdub'+str(nameN),[XscoreTrain, XderivedTrain])
        np.save('testTdub'+str(nameN),[XscoreTest, XderivedTest])
    else:
        print "----- Loading Training sample data..."
        XscoreTrain, XderivedTrain = np.load('trainTdub'+str(nameN)+'.npy')
        print "----- Loading Test sample data..."
        # Make data from Test set
        XscoreTest, XderivedTest = np.load('testTdub'+str(nameN)+'.npy')
        
    
    print "----- Predicting septic shock..."
    YtrueMRs[i], YscoreMRs[i] = septicPredict(XscoreTrain,XscoreTest)
    YtrueMRderiveds[i], YscoreMRderiveds[i] = septicPredict(XderivedTrain,XderivedTest)

for i in range(len(w_sols)):
    print "AUC (MRscore "+ str(i+1)+ ")         = " + str(roc_auc_score(YtrueMRs[i],YscoreMRs[i]))    
    print "AUC (MRscore+Derived "+ str(i+1)+ ") = " + str(roc_auc_score(YtrueMRderiveds[i],YscoreMRderiveds[i]))

print "AUC (Features only)     = " + str(roc_auc_score(YtrueFeat,YscoreFeat))

np.save('resultWhole',[YtrueFeat, YscoreFeat,YtrueMRs,YscoreMRs,YtrueMRderiveds, YscoreMRderiveds])

# plot ROC curve
plt.figure()
fprFeat, tprFeat,_ = roc_curve(YtrueFeat,YscoreFeat)
plt.plot(fprFeat, tprFeat,'k-')
for i in range(len(w_sols)):
    fprMRderived, tprMRderived,_ = roc_curve(YtrueMRderiveds[i],YscoreMRderiveds[i])
    plt.plot(fprMRderived,tprMRderived)
#    fprMR, tprMR,_ = roc_curve(YtrueMR,YscoreMR,pos_label=1)
#    plt.plot(fprMR,tprMR,fprMRderived,tprMRderived)    

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['Feature','LR+derived','MR_Euclid+derived','MR_Idiv+derived'])
plt.title('ROC')