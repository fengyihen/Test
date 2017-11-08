# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import sys
sys.path.append("Test")
from imp import reload
import Future
reload(Future)
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1


class FutureXgboost(Future.Future):

    def xgb_cls(self, testlen, ntrain, length_t, epochs, batchsize, timesteps,
                ncode, lr, tr, attr, modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        elif attr == 'rawcci':
            hsmadata_x = self.hsmadata_rawcci_x(timesteps, length_t)
        else:
            print('Wrong Attr!')
        hsmadata_y = self.hsmadata_bestp(testlen, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'Test\\futurexgboost\\testresult\\hsma_xgb_cls_testlen' + \
            str(testlen) + '_attr' + str(attr) + '_length_t' + str(
            length_t) + '_tr' + str(tr) + '_timesteps' + str(
            timesteps) + '_' + modellabel + '_' + self.label + '.h5'

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
                & (hsmadata['date'] <= dates[(i - 1) * testlen - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            preddate = dates[i * testlen]
            startdate = min(testdata.date[testdata.date > preddate])
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)

            ###选取盈利潜力最大的10个品种训练并预测
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)
            
            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'bestp', 'bestp_r'],
                                        1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatay = traindata['bestp'].values

            ###建模并预测
            ###改造成sklean api
            if modellabel == 'xgb':
                xclas = XGBClassifier(objective='multi:softmax')  
                xclas.fit(traindatax, traindatay)  
                testdata['predp'] = xclas.predict(testdatax) 
            else:
                pass

            for code in testdata.code.unique():
                testdata.loc[testdata.code == code, 'predp'] = testdata.loc[
                    testdata.code == code, 'predp'].iloc[0]

            testdata = testdata[testdata.date > dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'predp']].copy()
            else:
                hsma = pd.concat(
                    [hsma, testdata[['code', 'date', 'predp']]],
                    ignore_index=True)
                
            hsma.to_hdf(filename, 'hsma')

        return (hsma)
        
        
        
        
        