# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""
####################################################################
###############################下载数据##############################
####################################################################
import sys
sys.path.append("Test")
from StockData import *

startdate = 20100101
enddate = 20170331
winddata = WindData(startdate, enddate)
self = winddata
self.marketdata_history()
self.indexday()

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
hsmaindex = stocksklearn.indexhighlow1day(hsmaindex0, pr, mr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)
pr = 0
mr = 0.03
cr = 0.01
hsmaindex = stocksklearn.indexcloser1day(hsmaindex0, pr, mr, cr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)

###############################MLP regression####################
index = '000905.SH'
#index = '000300.SH'
testlen = 30
ntrain = 36
hidden_layer_sizes = (256,)
hsmaindex0 = stocksklearn.MLPRegressor_index(testlen, ntrain, index, *hidden_layer_sizes)
pr = 0
mr = 0.03
hsmaindex = stocksklearn.indexhighlow1day(hsmaindex0, pr, mr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)
pr = 0
mr = 0.03
rocn = 10
hsmaindex = stocksklearn.indexhighlow1day_roc(hsmaindex0, pr, mr, rocn)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)
pr = 0
mr = 0.03
cr = 0.01
hsmaindex = stocksklearn.indexcloser1day(hsmaindex0, pr, mr, cr)
hsmatradeday = stocksklearn.hsmaindexregressor(hsmaindex)
stocksklearn.tradestat(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'sklearn' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'test1'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 20
nodes = 20
columns = ['freeturn_1']
cmodel = 'KMeans'
ncluster = 10
hsma = stocksklearn.extratreesregressor_cluster(testlen, ntrain, ntrees, nodes, columns, cmodel, ncluster)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 100
nodes = 20
col1 = 'freeturn_3'
tn1 = 2000
ascending1 = False
col2 = 'CCI_3'
tn2 = 1500
ascending2 = False
hsma = stocksklearn.extratreesregressor_doublerank(testlen, ntrain, ntrees, nodes, col1, tn1, ascending1, col2, tn2, ascending2)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'sklearn' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'test2'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 20
nodes = 5
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'sklearn' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'svctest1'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
kernel = 'rbf'
dayr = 0.02
hsma = stocksklearn.svcstkmodel(testlen, ntrain, kernel, dayr)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'sklearn' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'svctest1_linear'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
kernel = 'linear'
dayr = 0.02
hsma = stocksklearn.svcstkmodel(testlen, ntrain, kernel, dayr)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D' 'None'
fee = 0.004
indexfee = 0.0002
label = 'GBRTtest1'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 200
nodes = 10
hsma = stocksklearn.GBRTregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
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
label = 'GBRTtest2'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 200
nodes = 10
hsma = stocksklearn.GBRTregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cr = None
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cr, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk2016'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
mtype = 'any' #or 'cnn1D'
fee = 0.004
indexfee = 0.0002
label = 'test1_2016'
stocksklearn = StockSklearn(dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 50
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
###################################################################
dataset = 'marketstk'
absratio = False
day = 2
length = [1, 3, 5, 10, 20, 40]
fee = 0.004
indexfee = 0.0002
label = 'test2'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 50
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

###############################MLP regression####################
testlen = 30
ntrain = 12
hidden_layer_sizes = (64,)
hsma = stocksklearn.MLPRegressor(testlen, ntrain, *hidden_layer_sizes)
condition = None
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170120_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)

###############################extratrees svr regression#############
testlen = 30
ntrain = 12
ntrees = 200
nodes = 10
hsma = stocksklearn.extratrees_linearsvr_regressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170120_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)

####################################################################
###############################ALL test##############################
####################################################################
import sys
sys.path.append("Test")
from stocksklearn.stocksklearn import StockSklearn

dataset = 'marketstk'
absratio = False
day = 1
length = [1, 3, 5, 10, 20, 40]
fee = 0.004
indexfee = 0.0002
label = 'lbtest1'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
r0 = 0.03
feature_sel = "all"
varthreshold = 0.2
cv = 10
binn = 10
bq = True
hsma = stocksklearn.logistic_binandwoe(testlen, ntrain, feature_sel, varthreshold, cv, binn, bq, r0)
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
day = 1
length = [1, 3, 5, 10, 20, 40]
fee = 0.004
indexfee = 0.0002
label = 'sklearntest1'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 200
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
fee = 0.004
indexfee = 0.0002
label = 'sklearntest2'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 12
ntrees = 200
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
day = 1
length = [1, 3, 5, 10, 20, 40]
fee = 0.004
indexfee = 0.0002
label = 'sklearntest61'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 6
ntrees = 200
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
fee = 0.004
indexfee = 0.0002
label = 'sklearntest62'
stocksklearn = StockSklearn(dataset, absratio, day, length, fee, indexfee, label)
self = stocksklearn

###############################extra trees regression#############
testlen = 30
ntrain = 6
ntrees = 200
nodes = 10
hsma = stocksklearn.extratreesregressor(testlen, ntrain, ntrees, nodes)
condition = None #'roc1'
n = 10
cta1 = "winpredlvdongbinshorton2zz500day_20170303_test1"
hsmatradeday = stocksklearn.hsmatradedayregressor(condition, hsma, n, cta1)
stocksklearn.tradestatlist(hsmatradeday)



