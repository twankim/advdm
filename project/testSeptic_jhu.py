# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:22:36 2015

@author: twankim
"""

from evalData_jhu import *
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt

saveFlag = True

fileName = 'predictData.csv'
w_sol = np.load('w_sol.npy')
print "----- Making Pos/Neg sample data..."
if saveFlag:
    dataPos, dataNeg, pidPos, pidNeg = makePosNegShock(fileName)
    np.save('shockPosNeg',[dataPos,dataNeg,pidPos,pidNeg])
else:
    dataPos, dataNeg, pidPos, pidNeg = np.load('shockPosNeg.npy')

print "----- Predicting septic shock..."
maxIter = 10
aucSepticMR = np.zeros(maxIter)
aucSepticFeat = np.zeros(maxIter)

for i in range(maxIter):
    print "- iter#: " + str(i)
    print "  Making Train/Test data..."
    Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest, Xtrain2, Xtest2 = makeDataShock(dataPos,dataNeg,pidPos,pidNeg,w_sol)
    print "  Predicting and calculating TPR/FPR..."
    tprMR, fprMR, tprFeat, fprFeat = septicPredict(Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest, Xtrain2, Xtest2)
    aucSepticMR[i] = auc(fprMR, tprMR)
    aucSepticFeat[i] = auc(fprFeat, tprFeat)
    
print "AUC (MRscore+Derived) = " + str(np.mean(aucSepticMR))
print "AUC (Features only) = " + str(np.mean(aucSepticFeat))

plt.figure()
plt.plot(fprMR,tprMR,'b-',fprFeat,tprFeat,'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['MR+derived','Feature'])
plt.title('ROC')
