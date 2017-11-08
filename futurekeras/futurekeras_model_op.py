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
day = 1
length = [1, 3, 5, 10, 20, 40]
timesteps = 10
label = 'sklearntest'+str(day)
futuremodel = futuresklearn.futuresklearn.FutureSklearn(code, day, length, timesteps, label)
self = futuremodel

###############################base strategy#####################
mr = 0.01
cr = 0.015
hsma0 = futuremodel.closer1day_base(mr, cr)
futuremodel.tradestat(hsma0)

###############################MA strategy#####################
malens = [3, 5, 10, 15, 20, 30]
#malens = [10]
tradestatlist = futuremodel.MA_optimizer(malens)

                  
                     






