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
import CTA.CTA
reload(CTA.CTA)
import matplotlib.pyplot as plt

cycle = 'min1'
label = 'CTAtest'
futuremodel = CTA.CTA.CTA(cycle, label)
self = futuremodel

###############################样本内策略测试结果#####################

#样本内测试
strategy = 'RSI_tpr_dayin'
length = 20
rr = 20
wtime = 10
tpr = 0.02
tupleArg = (length, rr, tpr, wtime)
hsmatrade = futuremodel.insample_test(strategy, *tupleArg)
portfolio = futuremodel.portfolio(hsmatrade)
tradestat = futuremodel.traderatiostat(hsmatrade)
plt.plot(portfolio.ratio)

#样本外测试
#strategy = 'RSI_tpr_dayin'
strategy = 'RSI_MA_tpr_dayin'
ncode = 5
ntrain = 12
length = 200
rr = 80
tpr = 0.02
wtime = 10
tupleArg = (length, rr, tpr, wtime)
hsmatrade = futuremodel.out_of_sample_month(strategy, ncode, ntrain, *tupleArg)
portfolio = futuremodel.portfolio(hsmatrade)
tradestat = InvestBase.tradestat_portfolio(portfolio)
plt.plot(portfolio.ratio)

#####################################################################
###############################RSI_reverse_N策略测试结果#############
####################################################################

strategy = 'RSI_ls'
ncode = 10
ntrain = 12
length = 80
rr = 60
tupleArg = (length, rr)
#样本内测试
hsmatrade = futuremodel.insample_test(strategy, *tupleArg)
futuremodel.traderatiostat(hsmatrade)
#样本外测试
hsmatrade = futuremodel.out_of_sample_month(strategy, ncode, ntrain, *tupleArg)
futuremodel.traderatiostat(hsmatrade)

strategy = 'RSI_reverse_ls'
ncode = 10
ntrain = 12
length = 10
rr = 30
tupleArg = (length, rr)
#样本内测试
hsmatrade = futuremodel.insample_test(strategy, *tupleArg)
futuremodel.traderatiostat(hsmatrade)
#样本外测试
hsmatrade = futuremodel.out_of_sample_month(strategy, ncode, ntrain, *tupleArg)
futuremodel.traderatiostat(hsmatrade)

strategy = 'BBANDS_ls'
ncode = 10
ntrain = 12
length = 200
nstd = 2
tupleArg = (length, nstd)
#样本内测试
hsmatrade = futuremodel.insample_test(strategy, *tupleArg)
futuremodel.traderatiostat(hsmatrade)
#样本外测试
hsmatrade = futuremodel.out_of_sample_month(strategy, ncode, ntrain, *tupleArg)
futuremodel.traderatiostat(hsmatrade)



