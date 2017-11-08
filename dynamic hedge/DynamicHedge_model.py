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
sys.path.append("Test\\dynamic hedge")
import os
import DynamicHedge as DH

index = '000905.SH'
market = 'index' # market index SH SZ
datataq = True
day = 5
absratio = False
fee = 0.004
label = 'test1'
DHModel = DH.DynamicHedge(index, market, datataq, day, absratio, fee, label)
self = DHModel

###############################random forest regression###########
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
hsma = DHModel.randomforestregressor(testlen, ntrain, ntrees, nodes)

###############################extra trees regression#############
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
hsma = DHModel.extratreesregressor(testlen, ntrain, ntrees, nodes)

###############################extra trees regression condition filter0###
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
hsma0 = DHModel.extratreesregressor(testlen, ntrain, ntrees, nodes)
col = 'predratio'
cr = 0.004
hsma = DHModel.conditionfilter0(hsma0, col, cr)

###############################extra trees regression condition###
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
col = 'cci7'
tn = 100
hsma = DHModel.extratreesregressor_conditionrank(testlen, ntrain, ntrees, nodes, col, tn)

###############################extra trees regression cluster###
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
columns = ['roc_1', 'roc_3', 'roc_5', 'roc7']
cmodel = 'KMeans'
ncluster = 10
hsma = DHModel.extratreesregressor_cluster(testlen, ntrain, ntrees, nodes, columns, cmodel, ncluster)

###############################extra trees classifier#############
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
dayr = 0.01
hsma = DHModel.extratreesclassifier(testlen, ntrain, ntrees, nodes, dayr)

##############################GradientBoostingRegressor trees regression##############
testlen = 30
ntrain = 6
ntrees = 20
nodes = 10
hsma = DHModel.GBRTmodel(testlen, ntrain, ntrees, nodes)

###############################support vector regression###########
testlen = 5
ntrain = 6
kernel = 'rbf' 
hsma = DHModel.svrmodel(testlen, ntrain, kernel)

###############################support vector regression###########
testlen = 5
ntrain = 6
batch = 10000
hsma = DHModel.linearsvrmodel(testlen, ntrain, batch)

###############################extra trees regression#############
testlen = 30
ntrain = 6
solver='adam'
hidden_layer_sizes=(20, )
hsma = DHModel.MLPRegressor(testlen, ntrain, solver, *hidden_layer_sizes)

###############################extra trees regression ind#############
testlen = 30
ntrain = 12
ntrees = 20
nodes = 10
hsma = DHModel.extratreesregressor_ind(testlen, ntrain, ntrees, nodes)

###############################support vector regression ind###########
kernel = 'rbf' 
hsma = DHModel.svrmodel_ind(testlen, ntrain, kernel)

###############################KNeighbors regression#############
testlen = 30
ntrain = 6
n_neighbors = 10
hsma = DHModel.knrmodel(testlen, ntrain, n_neighbors)

###############################选股及收益统计#######################
###regression
n = 200
idxma = 0

DHModel.tradestatlist(hsma, n, idxma)

###classification
minn = 30
idxma = 0
hsmatradeday = DHModel.hsmatradedayclassifier(hsma, minn, idxma)
tradestat =  DHModel.tradestat(hsmatradeday)



