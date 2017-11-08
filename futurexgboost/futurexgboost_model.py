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
import pandas as pd
import matplotlib.pyplot as plt

minp = 0.005
pnumber = 4
label = 'xgbtest'
futuremodel = futurexgboost.futurexgboost.FutureXgboost(minp, pnumber, label)
self = futuremodel

###############################Xgboost strategy#####################
testlen = 10
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 10
lr = 0.04
tr = 0.01
attr = 'raw'
modellabel = 'xgb'
readfile = True
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.xgb_cls(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, lr, tr,
    attr, modellabel, readfile)
hsmaratio, portfolio = futuremodel.hsmadata_predp_r(hsma, lr)
tradestat = futuremodel.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#######################################################################
###############################Trading Strategies######################
#######################################################################

###############################Keras strategy fixp#####################
#历史回测
testlen = 5  # 等于day
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 10
day = 5
lr = 0.04
p = 0.01
tr = 0.01
activation = 'sigmoid'
attr = 'raw'
modellabel = 'LSTM1cls'
readfile = True
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_classification_fixp(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, day, p, lr, tr,
    activation, attr, modellabel, readfile)
ncode = 10
portfolio = futuremodel.hsmadata_fixp_r(hsma, ncode, p, lr)
tradestat = futuremodel.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_fixp_day5_attrraw_length_t20_tr0.01_timesteps5_tanh_LSTM1cls_kerastest.h5'
hsma = pd.read_hdf(filename, 'hsma')
ncode = 10
portfolio = futuremodel.hsmadata_fixp_r(hsma, ncode, p, lr)
tradestat = futuremodel.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#实盘预测
preddate = 20170922
testlen = 10  # 等于day
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 20
day = 10
lr = 0.04
p = 0.01
tr = 0
activation = 'relu'
attr = 'raw'
modellabel = 'LSTM1cls'
codeprob = self.lstm_classification_fixp_pred(
    preddate, testlen, ntrain, length_t, epochs, batchsize, timesteps, day, p,
    lr, tr, activation, attr, modellabel)
