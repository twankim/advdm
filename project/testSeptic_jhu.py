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

saveFlag = False

fileTrain = 'predictDataTrain.csv'
fileTest = 'predictDataTest.csv'
w_sol = np.load('w_sol.npy')

mode = 0 # Selecting score function

if saveFlag:
    print "----- Making Training sample data..."
    # Make data from Training set
    XscoreTrain, XderivedTrain, XfeatureTrain = makeDataShock(fileTrain,w_sol,mode)
    print "----- Making Test sample data..."
    # Make data from Test set
    XscoreTest, XderivedTest, XfeatureTest = makeDataShock(fileTest,w_sol,mode)
    
    np.save('trainTdub',[XscoreTrain, XderivedTrain, XfeatureTrain])
    np.save('testTdub',[XscoreTest, XderivedTest, XfeatureTest])
else:
    print "----- Loading Training sample data..."
    XscoreTrain, XderivedTrain, XfeatureTrain = np.load('trainTdub.npy')
    print "----- Loading Test sample data..."
    # Make data from Test set
    XscoreTest, XderivedTest, XfeatureTest = np.load('testTdub.npy')
    

print "----- Predicting septic shock..."
YtrueMR, YscoreMR = septicPredict(XscoreTrain,XscoreTest)
YtrueMRderived, YscoreMRderived = septicPredict(XderivedTrain,XderivedTest)
YtrueFeat, YscoreFeat = septicPredict(XfeatureTrain,XfeatureTest)

print "AUC (MRscore) = " + str(roc_auc_score(YtrueMR,YscoreMR))    
print "AUC (MRscore+Derived) = " + str(roc_auc_score(YtrueMRderived,YscoreMRderived))
print "AUC (Features only) = " + str(roc_auc_score(YtrueFeat,YscoreFeat))

fprMR, tprMR,_ = roc_curve(YtrueMR,YscoreMR,pos_label=1)
fprMRderived, tprMRderived,_ = roc_curve(YtrueMRderived,YscoreMRderived)
fprFeat, tprFeat,_ = roc_curve(YtrueFeat,YscoreFeat)

plt.figure()
plt.plot(fprMR,tprMR,'b-',fprMRderived,tprMRderived,'r-', fprFeat, tprFeat,'k-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['MR','MR+derived','Feature'])
plt.title('ROC')