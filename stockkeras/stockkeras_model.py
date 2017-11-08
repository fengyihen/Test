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
from stockkeras.stockkeras import StockKerasSequential

dataset = 'marketstk'
absratio = False
day = 2
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'cnn1D' #or 'any'
r = 0.03
fee = 0.004
indexfee = 0.0002
label = 'testn'
stocksklearn = StockKerasSequential(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn


#########################数据预处理################################
self.hsmaseq(timesteps)
self.hsmaseq_raw(timesteps)

###############################extra trees regression#############
raw = True
testlen = 30#10
ntrain = 12
hsma = stocksklearn.extratreesregressor(testlen, ntrain, timesteps, raw)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
#stocksklearn.tradestatlist(hsmatradeday)
