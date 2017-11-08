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
import futuresklearn.futuresklearn
reload(futuresklearn.futuresklearn)

code = 'rb000_day'
#code = 'ru000_day'
#code = 'i9000_day'
#code = 'm9000_day'
#code = 'j9000_day'
length = [1, 3, 5, 10, 20, 40]
label = 'sklearntest'
futuremodel = futuresklearn.futuresklearn.FutureSklearn(code, length, label)
self = futuremodel

###############################base strategy#####################
mr = 0.01
cr = 0.015
hsma0 = futuremodel.closer1day_base(mr, cr)
futuremodel.tradestat(hsma0)

###############################bollinger strategy#####################
malen = 5
mr = 0.01
cr = 0.015
hsma0 = futuremodel.closer1day_bollinger(malen, mr, cr)
futuremodel.tradestat(hsma0)

###############################MA strategy#####################
malens = [3, 5, 10, 15, 20, 30]
tradestatlist = futuremodel.MA_optimizer(malens)

malen = 10
hsma = futuremodel.MA(malen)
futuremodel.tradestat(hsma)                
    
malen1 = 10
malen2 = 30
hsma = futuremodel.MAcross(malen1, malen2)
futuremodel.tradestat(hsma) 
 
malen = 10
s = 30
hsma = futuremodel.MA_ADX(malen, s)
futuremodel.tradestat(hsma)   
                
###############################linear regression#############
testlen = 30
ntrain = 12
raw = False
timesteps = 3
day = 10
#feature_sel = "SelectKBest"
feature_sel = None
hsma = futuremodel.linearregressor(testlen, ntrain, raw, timesteps, day, feature_sel)
hsmatrade = futuremodel.hsmatraderegressor(hsma, day)
r = 0.01
hsmatrade = futuremodel.hsmatraderegressor_r(hsma, day, r)


###############################extra trees regression#############
testlen = 20
ntrain = 36
ntrees = 200
nodes = 3
raw = True
timesteps = 10
day = 3
hsma = futuremodel.extratreesregressor(testlen, ntrain, ntrees, nodes)
pr = 0
mr = 0.01
cr = 0.015
hsma = futuremodel.closer1day(hsma, pr, mr, cr)
futuremodel.tradestat(hsma)

###############################linear var#############
testlen = 30
ntrain = 12
raw = True
timesteps = 3
vn = 10
#feature_sel = "SelectKBest"
feature_sel = None
hsma = futuremodel.linearvar(testlen, ntrain, raw, timesteps, vn, feature_sel)
hsmatrade = futuremodel.hsmatraderegressor(hsma, day)
r = 0.01
hsmatrade = futuremodel.hsmatraderegressor_r(hsma, day, r)


