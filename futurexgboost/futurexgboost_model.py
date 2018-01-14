# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""
####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
from imp import reload
import futurexgboost.futurexgboost
reload(futurexgboost.futurexgboost)
import InvestBase
reload(InvestBase)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

minp = 0.005
pnumber = 4
label = 'xgbtest'
futuremodel = futurexgboost.futurexgboost.FutureXgboost(minp, pnumber, label)
self = futuremodel

###############################Xgboost strategy:classification#####################
testlen = 60
ntrain = 12
lengths = [1,3,5,9,15,30]
timesteps = 60
day = 2
tr = 0.01
attr = 'ta'
attry = 'roo'
modellabel = 'xgb'
readfile = False
hsma = futuremodel.xgb_cls(testlen, ntrain, lengths, timesteps, day, tr, attr, attry, 
                           modellabel, readfile)
pr = 0.5
fee = 0.0004
hsmaratio = futuremodel.hsmadata_daycode_lsr(hsma, day, pr, fee)
tradestat = InvestBase.tradestat_portfolio(hsmaratio)
plt.plot(hsmaratio.ratio)

###############################Xgboost strategy:regression#####################
testlen = 60
ntrain = 12
lengths = [1,3,5,9,15,30]
timesteps = 60
day = 2
tr = 0.01
attr = 'ta'
attry = 'roo'
modellabel = 'xgb'
readfile = False
feature_sel = 'N'#'SelectFromModel'
max_depth = 10
learning_rate = 1
reg_alpha = 4
reg_lambda = 2
hsma = futuremodel.xgb_reg(testlen, ntrain, lengths, timesteps, day, tr, attr, 
                           attry, feature_sel, max_depth, learning_rate, 
                           reg_alpha, reg_lambda, modellabel, readfile)
r = 0.01
fee = 0.0004
hsmaratio = futuremodel.hsmatraderegressor_r(hsma, day, r, fee)
tradestat = InvestBase.tradestat_portfolio(hsmaratio)
plt.plot(hsmaratio.ratio)

##loop
max_depths = np.array([5, 10, 15, 20])
learning_rates = np.array([0.01, 0.1, 1, 2])
reg_alphas = np.array([0, 1, 2, 4])
reg_lambdas = np.array([0, 1, 2, 4])
r = 0.01
fee = 0.0004
result = futuremodel.xgb_reg_loop(testlen, ntrain, lengths, timesteps, day, tr, attr, 
                attry, max_depths, learning_rates, reg_alphas, reg_lambdas, 
                modellabel, readfile, r, fee)

#######################################################################
###############################Trading Strategies######################
#######################################################################
