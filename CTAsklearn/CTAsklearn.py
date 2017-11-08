# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import sys
sys.path.append("Test")
from imp import reload
import FutureMinute
reload(FutureMinute)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree

#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1


class CTAsklearn(FutureMinute.FutureMinute):

    def logistic_dayin(self, testlen, ntrain, llen, slen, tr):

        hsmadata = self.hsmadata(llen, slen)

        filename = 'Test\\CTAsklearn\\testresult\\logistic_dayin_llen{}_slen{}_{}.h5'.format(llen, slen, self.label)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] < dates[i * testlen])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())
            predresult = testdata[['date', 'time', 'code', 'open', 'high', 
                                   'low', 'close', 'vol', 'openint', 
                                   'ratio']].copy()

            X_train = traindata[['roc', 'dayr', 'dayh', 'dayl', 'dayhl', 'smar', 'VAR',
                                 'timepos']]
            y_train_long = traindata['ratio'] > tr
            y_train_short = traindata['ratio'] < -tr
            X_test = testdata[X_train.columns]
            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(X_train, y_train_long)  
            probability = classifier.predict_proba(X_test)
            predresult['prob_long'] = probability[:,1]
            classifier.fit(X_train, y_train_short)  
            probability = classifier.predict_proba(X_test)
            predresult['prob_short'] = probability[:,1]
            
            hsma = pd.concat([hsma, predresult], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def tree_dayin(self, testlen, ntrain, llen, slen, tr):

        hsmadata = self.hsmadata(llen, slen)

        filename = 'Test\\CTAsklearn\\testresult\\tree_dayin_llen{}_slen{}_{}.h5'.format(llen, slen, self.label)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] < dates[i * testlen])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())
            predresult = testdata[['date', 'time', 'code', 'open', 'high', 
                                   'low', 'close', 'vol', 'openint', 
                                   'ratio']].copy()

            X_train = traindata[['roc', 'dayr', 'dayh', 'dayl', 'dayhl', 'smar', 'VAR',
                                 'timepos']]
            y_train_long = traindata['ratio'] > tr
            y_train_short = traindata['ratio'] < -tr
            X_test = testdata[X_train.columns]
            
            #训练并预测模型
            classifier = tree.DecisionTreeClassifier()  # 使用类，参数全是默认的
            classifier.fit(X_train, y_train_long)  
            probability = classifier.predict_proba(X_test)
            predresult['prob_long'] = probability[:,1]
            classifier.fit(X_train, y_train_short)  
            probability = classifier.predict_proba(X_test)
            predresult['prob_short'] = probability[:,1]
            
            hsma = pd.concat([hsma, predresult], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)
        
        
