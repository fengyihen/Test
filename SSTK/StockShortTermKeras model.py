# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""
#import fyzinitialize as fi
#import momentum tick as mt
####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test//SSTK")
import os
import StockShortTermKeras as sstk

index = '000905.SH'
market = 'index' # market index SH SZ
datataq = False
day = 5
absratio = False
fee = 0.004
label = 'test1'
sstkModel = sstk.StockShortTerm(index, market, datataq, day, absratio, fee, label)
self = sstkModel

timesteps = 20
sstkModel.hsmaseq(timesteps)

###############################random forest regression###########
testlen = 30
ntrain = 6
r = 0.04

hsma = sstkModel.randomforestregressor(testlen, ntrain, ntrees, nodes)


###classification
minn = 30
idxma = 0
hsmatradeday = sstkModel.hsmatradedayclassifier(hsma, minn, idxma)
tradestat =  sstkModel.tradestat(hsmatradeday)



