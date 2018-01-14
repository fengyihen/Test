# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import sys
sys.path.append("Test")
from imp import reload
import FutureDay
reload(FutureDay)
import InvestBase
reload(InvestBase)
import pandas as pd
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest

#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1


class FutureXgboost(FutureDay.Future):

    def xgb_cls(self, testlen, ntrain, lengths, timesteps, day, tr, attr, attry, 
                modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'ta':
            hsmadata_x = self.hsmadata_ta_x(lengths)
        else:
            print('Wrong Attr!')
            
        if attry == 'roc':
            hsmadata_y = self.hsmadata_roc(day)
        elif attry == 'roo':
            hsmadata_y = self.hsmadata_roo(day)
        else:
            print('Wrong Attr_y!')
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'testresult\\futurexgboost\\hsma_xgb_cls_testlen' + \
            str(testlen) + '_attr' + str(attr) + '_attry' + str(attry) + \
            '_tr' + str(tr) + '_timesteps' + str(timesteps) + '_day' + str(day) + \
            '_' + modellabel + '_' + self.label + '.h5'

        if readfile:
            if os.path.exists(filename):
                hsma = pd.read_hdf(filename, 'hsma')
            else:
                hsma = pd.DataFrame()
        else:
            hsma = pd.DataFrame()
        
        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            startdate = dates[i * testlen]
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)
            
            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            testdatax = testdata[traindatax.columns]
            traindatay_long = traindata['ratio'].copy()
            traindatay_long[traindata['ratio'] >= tr] = 1
            traindatay_long[traindata['ratio'] < tr] = 0  
            traindatay_short = traindata['ratio'].copy()
            traindatay_short[traindata['ratio'] <= -tr] = 1
            traindatay_short[traindata['ratio'] > -tr] = 0  
            
            #加入变量筛选
            
            ###建模并预测
            ###xgboost sklearn api
            if modellabel == 'xgb':
                xclas = XGBClassifier(max_depth=10, learning_rate=0.1)  #objective='multi:softmax'
                xclas.fit(traindatax, traindatay_long)  
                testdata['pred_long'] = xclas.predict(testdatax) 
                testdata['prob_long'] = xclas.predict_proba(testdatax)[:,1]
                xclas = XGBClassifier(max_depth=10, learning_rate=0.1)
                xclas.fit(traindatax, traindatay_short)  
                testdata['pred_short'] = xclas.predict(testdatax)
                testdata['prob_short'] = xclas.predict_proba(testdatax)[:,1]
            else:
                pass


            if i == ntrain:
                hsma = testdata[['code', 'date', 'ratio', 'pred_long', 'prob_long',
                                 'pred_short', 'prob_short']].copy()
            else:
                hsma = pd.concat(
                    [hsma, testdata[['code', 'date', 'ratio', 'pred_long',
                                     'prob_long', 'pred_short', 'prob_short']]],
                    ignore_index=True)
            
            if readfile:
                hsma.to_hdf(filename, 'hsma')

        return (hsma)
        
    def xgb_reg(self, testlen, ntrain, lengths, timesteps, day, tr, attr, attry, 
                feature_sel, max_depth, learning_rate, reg_alpha, reg_lambda, 
                modellabel, readfile):        
        
        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'ta':
            hsmadata_x = self.hsmadata_ta_x(lengths)
        else:
            print('Wrong Attr!')
            
        if attry == 'roc':
            hsmadata_y = self.hsmadata_roc(day)
        elif attry == 'roo':
            hsmadata_y = self.hsmadata_roo(day)
        else:
            print('Wrong Attr_y!')
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'testresult\\futurexgboost\\hsma_xgb_cls_testlen' + \
            str(testlen) + '_attr' + str(attr) + '_attry' + str(attry) + \
            '_tr' + str(tr) + '_timesteps' + str(timesteps) + '_day' + str(day) + \
            '_' + modellabel + '_' + self.label + '.h5'

        if readfile:
            if os.path.exists(filename):
                hsma = pd.read_hdf(filename, 'hsma')
            else:
                hsma = pd.DataFrame()
        else:
            hsma = pd.DataFrame()
        
        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            startdate = dates[i * testlen]
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)
            
            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            testdatax = testdata[traindatax.columns]
            traindatay = traindata['ratio']
            
            #在train中做变量筛选, sklearn.feature_selection中的方法
            if feature_sel == "SelectFromModel":
                estimator = XGBRegressor()
                selector = SelectFromModel(estimator)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            elif feature_sel == "SelectKBest":
                selector = SelectKBest(k=traindatax.shape[1]//2)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            else:
                traindatax1, testdatax1 = traindatax, testdatax 
            
            ###建模并预测
            ###xgboost sklearn api
            if modellabel == 'xgb':
                xclas = XGBRegressor(max_depth, learning_rate, 
                                     reg_alpha, reg_lambda)  #objective='multi:softmax'
                xclas.fit(traindatax1, traindatay)
                testdata['predratio'] = xclas.predict(testdatax1) 
            else:
                pass


            if i == ntrain:
                hsma = testdata.copy()
            else:
                hsma = pd.concat([hsma, testdata], ignore_index=True)
            
            if readfile:
                hsma.to_hdf(filename, 'hsma')

        return (hsma)
    
    def xgb_reg_loop(self, testlen, ntrain, lengths, timesteps, day, tr, attr, 
                attry, feature_sel, max_depths, learning_rates, reg_alphas, 
                reg_lambdas, modellabel, readfile, r, fee): 
        
        result = pd.DataFrame()
        for max_depth in max_depths:
            for learning_rate in learning_rates:
                for reg_alpha in reg_alphas:
                    for reg_lambda in reg_lambdas:
                        hsma = self.xgb_reg(testlen, ntrain, lengths, timesteps, 
                            day, tr, attr, attry, feature_sel, max_depth, 
                            learning_rate, reg_alpha, reg_lambda, modellabel, 
                            readfile)
                        hsmaratio = self.hsmatraderegressor_r(hsma, day, r, fee)
                        if hsmaratio.shape[0] > 0:
                            temp = pd.DataFrame({'max_depth' : max_depth,
                                                'learning_rate' : learning_rate,
                                                'reg_alpha' : reg_alpha,
                                                'reg_lambda' : reg_lambda,
                                                'ratio' : hsmaratio.ratio.iloc[hsmaratio.shape[0]-1]
                                                }, index=[0])
                            result = pd.concat([result, temp], ignore_index=True)
                            result.sort_values('ratio', axis=0, ascending=False, inplace=True)
                            print(result.head())
        
        return result
        
        
        