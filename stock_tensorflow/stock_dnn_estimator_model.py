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
from stock_tensorflow.stock_dnn_estimator import StockDnnEstimator

dataset = 'marketstk'
absratio = False
day = 2
length = [1, 3, 5, 10, 20, 40]
fee = 0.004
label = 'test2'
stockdnnestimator = StockDnnEstimator(dataset, absratio, day, length, fee, label)
self = stockdnnestimator

###############################dnnclassifier_estimator###########
testlen = 30
ntrain = 6
r = 0.02
hidden_units = [10, 20, 20, 20, 20, 20, 10]
steps = 200
hsma = stockdnnestimator.dnnclassifier_estimator(testlen, ntrain, r, hidden_units, steps)
condition = None#'roc1'
minn = 10
hsmatradeday = stockdnnestimator.hsmatradedayclassifier(condition, hsma, minn)

###############################dnnregressor_estimator###########
testlen = 30
ntrain = 12
hidden_units = [30, 30, 30, 30, 30, 20, 10]
steps = 200
hsma = stockdnnestimator.dnnregressor_estimator(testlen, ntrain, hidden_units, steps)
condition = None#'roc1'
n = 10
hsmatradeday = stockdnnestimator.hsmatradedayregressor(condition, hsma, n)

###############################widendnnclassifier_estimator###########
testlen = 30
ntrain = 6
r = 0.02
hidden_units = [10, 20, 20, 20, 10]
steps = 100
hsma = stockdnnestimator.widendeepclassifier_estimator(testlen, ntrain, r, hidden_units, steps)
condition = None#'roc1'
minn = 10
hsmatradeday = stockdnnestimator.hsmatradedayclassifier(condition, hsma, minn)

###############################widendnnregressor_estimator###########
testlen = 30
ntrain = 6
hidden_units = [10, 20, 20, 20, 10]
steps = 100
hsma = stockdnnestimator.widendeepregressor_estimator(testlen, ntrain, hidden_units, steps)
condition = None#'roc1'
n = 10
hsmatradeday = stockdnnestimator.hsmatradedayregressor(condition, hsma, n)


