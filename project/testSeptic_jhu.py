# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:22:36 2015

@author: twankim
"""

import pickle
from evalData import *
from sklearn.metrics import roc_auc_score

f = open('w_sol.pckl')
w_sol = pickle.load(f)
f.close()
[Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest] = makeDataShock(w_sol)

scorePredict = septicPredict(Xtrain, Ytrain, pidTrain, Xtest, Ytest, pidTest)
print "AUC = " + str(roc_auc_score(Ytest,scorePredict))
