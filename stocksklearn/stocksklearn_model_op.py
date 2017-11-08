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
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'sklearntest'+str(day)
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 1000
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 2
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'sklearntest'+str(day)
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 1000
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 3
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'sklearntest'+str(day)
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 1000
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 5
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'sklearntest'+str(day)
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 1000
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################,
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 10
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'sklearntest'+str(day)
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 1000
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from imp import reload
import stocksklearn.stocksklearn
reload(stocksklearn.stocksklearn)

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'None'#'sklearn' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'indextest1'
stocksklearn = stocksklearn.stocksklearn.StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################base strategy#####################
index = '000905.SH'
#index = '000300.SH'
mr = 0.03
cr = 0.01
hsma0 = stocksklearn.indexcloser1day_base(index, mr, cr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsma0)
stocksklearn.tradestat(hsmatradeday)

###############################extra trees regression#############
index = '000905.SH'
#index = '000300.SH'
testlen = 30
ntrain = 36
ntrees = 200
nodes = 5
hsmaindex0 = stocksklearn.extratreesregressor_index(testlen, ntrain, index, ntrees, nodes)
pr = 0
mr = 0.03
cr = 0.01
hsmaindex = stocksklearn.indexcloser1day(hsmaindex0, pr, mr, cr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)





