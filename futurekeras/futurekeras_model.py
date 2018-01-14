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
import InvestBase
from imp import reload
import futurekeras.futurekeras
reload(futurekeras.futurekeras)
import pandas as pd
import matplotlib.pyplot as plt

minp = 0.005
pnumber = 4
label = 'kerastest'
futuremodel = futurekeras.futurekeras.FutureKeras(minp, pnumber, label)
self = futuremodel

###############################Keras strategy#####################
#多品种共同建模
testlen = 60  # 等于day
ntrain = 12
epochs = 20
batchsize = 3000
timesteps = 60
day = 10
tr = 0.01
activation = 'relu'
attr = 'raw'
attry = 'roo'
modellabel = 'LSTM1cls'
#modellabel = 'LSTM2cls'
readfile = False
hsma = futuremodel.lstm_classification_r(testlen, ntrain, epochs,
                                batchsize, timesteps, day, tr, activation,
                                attr, attry, modellabel, readfile)
hsmaratio = futuremodel.hsmadata_daycode_lsr(hsma, day)
tradestat = InvestBase.tradestat_portfolio(hsmaratio)
plt.plot(hsmaratio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_kerastest_LSTM1cls.h5'
hsma = pd.read_hdf(filename, 'hsma')
hsmaratio, portfolio = futuremodel.hsmadata_predp_r(hsma, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)


#每个品种单独建模
testlen = 30  #10
ntrain = 12
epochs = 20
batchsize = 2000
timesteps = 20
ncode = 5
day = 20
lr = 0.04
modellabel = 'LSTM1cls'
#modellabel = 'LSTM2cls'
hsma = futuremodel.lstm_classification_code(
    testlen, ntrain, epochs, batchsize, timesteps, ncode, day, lr, modellabel)
hsmaratio, portfolio = futuremodel.hsmadata_predp2_r(hsma, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################Keras strategy fixma#####################
#多品种共同建模
testlen = 20  # 等于day
ntrain = 12
epochs = 20
batchsize = 3000
timesteps = 20
ncode = 5
day = 20
lr = 0.04
m = 20
tr = 0
modellabel = 'LSTM1cls'
#modellabel = 'LSTM2cls'
hsma = futuremodel.lstm_classification_fixma(testlen, ntrain, epochs,
                                             batchsize, timesteps, ncode, day,
                                             m, lr, tr, modellabel)
hsmaratio, portfolio = futuremodel.hsmadata_fixp_r(hsma, m, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################Keras strategy roc#####################
#多品种共同建模
testlen = 20  # 等于day
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 10
day = 20
tr = 0.02
activation = 'sigmoid'
attr = 'raw'
modellabel = 'LSTM1cls'
readfile = False
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_classification_roc(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, day, tr,
    activation, attr, modellabel, readfile)
ncode = 5
length = 20
portfolio = futuremodel.hsmadata_sma_r(hsma, ncode, length)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################Keras strategy std#####################
#历史回测
testlen = 5  # 等于day
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 5
day = 5
tr = 0.02
activation = 'relu'
attr = 'raw'
modellabel = 'LSTM1cls'
readfile = True
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_classification_std(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, day, tr,
    activation, attr, modellabel, readfile)
ncode = 10
p = 0.01
lr = 0.04
portfolio = futuremodel.hsmadata_fixp_r(hsma, ncode, p, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################Keras strategy breakhl fixlength#####################
#历史回测
testlen = 20  # 等于day
ntrain = 12
length = 20
epochs = 20
batchsize = 3000
timesteps = 5
lr = 0.04
tr = 0.02
activation = 'relu'
attr = 'raw'
modellabel = 'LSTM1cls'
readfile = False
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_cls_breakhlfixlength(
    testlen, ntrain, length, epochs, batchsize, timesteps, lr, tr, activation,
    attr, modellabel, readfile)
ncode = 10
lr = 0.04
portfolio = futuremodel.hsmadata_breakhl_fixlength_r(hsma, ncode, length, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

###############################Keras strategy fixvar###########################
#历史回测
testlen = 20  # 等于day
ntrain = 12
length_t = 20
epochs = 20
batchsize = 3000
timesteps = 20
day = 20
p = 0.01
v = 1 
lr = 0.04
tr = 0.02
activation = 'relu'
attr = 'raw'
yvar = 'fixvar'
modellabel = 'LSTM1cls'
readfile = True
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_classification_fixpv(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, day, p, v, lr, tr,
    activation, attr, yvar, modellabel, readfile)
ncode = 10
portfolio = futuremodel.hsmadata_fixvar_r(hsma, day, ncode, v, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_{}_v{}_day{}_attr{}_length_t{}_tr{}_timesteps{}_{}_{}_{}.h5'.format(
    yvar,v,day,attr,length_t,tr,timesteps,activation,modellabel,self.label)
hsma = pd.read_hdf(filename, 'hsma')
ncode = 10
portfolio = futuremodel.hsmadata_fixvar_r(hsma, day, ncode, v, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
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
p = 0.01
v = 0.5
lr = 0.04
tr = 0.01
activation = 'relu'
attr = 'raw'
yvar = 'fixp'
modellabel = 'LSTM1cls'
readfile = True
#modellabel = 'LSTM2cls'
#readfile = False
hsma = futuremodel.lstm_classification_fixpv(
    testlen, ntrain, length_t, epochs, batchsize, timesteps, day, p, v, lr, tr,
    activation, attr, yvar, modellabel, readfile)
ncode = 10
portfolio = futuremodel.hsmadata_fixp_r(hsma, ncode, p, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#测试结果统计
filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_fixp_p0.01_day5_attrraw_length_t20_tr0.01_timesteps10_relu_LSTM1cls_kerastest.h5'
hsma = pd.read_hdf(filename, 'hsma')
ncode = 10
p = 0.01
lr = 0.04
portfolio = futuremodel.hsmadata_fixp_r(hsma, ncode, p, lr)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#实盘预测
preddate = 20170929
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
activation = 'relu'
attr = 'raw'
modellabel = 'LSTM1cls'
codeprob = self.lstm_classification_fixp_pred(
    preddate, testlen, ntrain, length_t, epochs, batchsize, timesteps, day, p,
    lr, tr, activation, attr, modellabel)
