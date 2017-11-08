# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.regularizers import l2, activity_l2
from keras.layers import Embedding


class StockShortTerm(object):
    
    #所有股票所有日期上的技术指标统计，排序后
    
    def __init__(self, index, market, datataq, day, absratio, fee, label):
        self.index = index
        self.market = market # market全市场 index指数成分股 SH指数成分股中的上海股票 SZ指数成分股中的深圳股票
        self.datataq = datataq        
        self.day = day
        self.absratio = absratio
        self.fee = fee
        self.label = label
        self.hsmadata = self.hsmadata()
    
    def hsmadata(self):        
        
        if self.datataq:
            if self.market == 'market':
                LTTAQday = pd.read_csv("strategy data//LTTAQdayMarketMAshort.csv")
                LTdayall = pd.read_csv("strategy data//LTdayallMarket.csv")
            else:
                LTTAQday = pd.read_csv("strategy data//LTTAQdayMAshort.csv")
                LTdayall = pd.read_csv("strategy data//LTdayall.csv")
            LTdayall = LTdayall[LTdayall['coratio'] < 0.09]
            LTdayall = LTdayall[["code","date","ooratio" + str(self.day)]]
            LTdayall = LTdayall.rename(columns = {"ooratio" + str(self.day):'closeratio'})
            hsma = pd.merge(LTTAQday,LTdayall)
        else:
            if self.market == 'market':
                LTdayall = pd.read_csv("strategy data//LTdayallMarket.csv")
            else:
                LTdayall = pd.read_csv("strategy data//LTdayall.csv")
            LTdayall = LTdayall[LTdayall['coratio'] < 0.09]
            LTdayall = LTdayall[["code","date","ooratio" + str(self.day)]]
            LTdayall = LTdayall.rename(columns = {"ooratio" + str(self.day):'closeratio'})
            hsma = LTdayall.copy()
        
        if not self.absratio:
            LTindexoo = pd.read_csv("strategy data//LTindexoo_000905.SH.csv")
            LTindexoo = LTindexoo[['date', "ooratio" + str(self.day)]]
            hsma = pd.merge(hsma,LTindexoo)
            hsma['closeratio'] = hsma['closeratio'] - hsma["ooratio" + str(self.day)]
            hsma = hsma.drop(["ooratio" + str(self.day)],1)

        if self.market == 'market':
            LTdayTTR = pd.read_csv("strategy data//LTdayTTR0Market_length7_sd2.csv")      
        else:
            LTdayTTR = pd.read_csv("strategy data//LTdayTTR0_length7_sd2.csv") 
        LTdayTTR = LTdayTTR[["code","date","adx","mfi","rsi","volatility","roc","cci"]]
        LTdayTTR.columns = ["code","date","adx7","mfi7","rsi7","volatility7","roc7","cci7"]
        hsma = pd.merge(hsma,LTdayTTR)
        hsma = hsma[hsma['volatility7'] > 0]
        
        if self.market == 'market':
            LTdayTTR = pd.read_csv("strategy data//LTdayTTR0Market_length14_sd2.csv")      
        else:
            LTdayTTR = pd.read_csv("strategy data//LTdayTTR0_length14_sd2.csv") 
        LTdayTTR = LTdayTTR[["code","date","adx","mfi","rsi","volatility","roc","cci"]]
        LTdayTTR.columns = ["code","date","adx14","mfi14","rsi14","volatility14","roc14","cci14"]
        hsma = pd.merge(hsma,LTdayTTR)
        
        if self.market == 'market':
            LTdayvolr = pd.read_csv("strategy data//LTdayvolrMarket.csv")      
        else:
            LTdayvolr = pd.read_csv("strategy data//LTdayvolr.csv") 
        hsma = pd.merge(hsma,LTdayvolr)
        
        if self.market == 'market':
            LTdayroc = pd.read_csv("strategy data//LTdayroc_shortMarket.csv")     
        else:
            LTdayroc = pd.read_csv("strategy data//LTdayroc_short.csv") 
        hsma = pd.merge(hsma,LTdayroc)

        LTindex = pd.read_csv("strategy data//LTindexroc_000905.SH.csv")   
        hsma = pd.merge(hsma,LTindex)
        
        if self.market == 'SH':
            
            LTTRANSACTIONday = pd.read_csv("strategy data//LTTRANSACTIONday.csv")        
            hsma = pd.merge(hsma,LTTRANSACTIONday)

        if self.market == 'SZ':
            
            LTTRADEday = pd.read_csv("strategy data//LTTRADEday.csv")        
            hsma = pd.merge(hsma,LTTRADEday)
        
        hsma = hsma.dropna()
 
        return(hsma) 
        
    def hsmaseq(self, timesteps):
        
        traindata = self.hsmadata

        codelist = pd.Series(traindata['code'].unique())
        
        flag = 0
        for code in codelist:
            traindatac = traindata[traindata['code'] == code].copy()
            traindatac = traindatac.sort_values(by='date')

            if traindatac.shape[0] < timesteps:
                continue
            
            traindatacx = traindatac.drop(['code', 'date', 'closeratio'], 1)
            traindatacy = traindatac[['code','date','closeratio']]
            
            for i in range(traindatac.shape[0] - timesteps + 1):
                x_traintemp = np.array(traindatacx.iloc[i:(i+timesteps), ])
                y_traintemp = np.array(traindatacy.iloc[(i+timesteps-1), ])
                y_traintemp[0] = int(y_traintemp[0][0:6])
                
                x_temp = np.empty((1, timesteps, x_traintemp.shape[1]))
                x_temp[0, :, :] = x_traintemp
                
                y_temp = np.empty((1,3))
                y_temp[0, :] = y_traintemp
                
                if flag == 0:
                    x_train = x_temp
                    y_train = y_temp
                    flag = 1
                else:
                    x_train = np.concatenate((x_train,x_temp))                    
                    y_train = np.concatenate((y_train, y_temp))
                    
        np.savez('Test//SSTK//hsmaseq_' + str(timesteps) + '.npz', x=x_train,y=y_train)
        
    def kerasfnn_classifier(self, testlen, ntrain, r):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train = np.array(traindata.drop(['code', 'date', 'closeratio'], 1))
            y_train = np.array(traindata['closeratio'] > r, dtype=np.int8)
            
            x_test = np.array(testdata.drop(['code', 'date', 'closeratio'], 1))
            #y_test = np.array(testdata['closeratio'] > 0, dtype=np.int8)
            
            ###FNN model
            model = Sequential()
            model.add(Dense(32, input_dim=x_train.shape[1], W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            #model.add(Dense(32, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            #model.add(Activation('relu'))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            #model.add(Activation('sigmoid'))
            model.add(Activation('sigmoid'))
            
            model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_train, nb_epoch=100, batch_size=10000)
            
            testdata['predratio'] = model.predict_on_batch(x_test)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)        
        
    def kerasfnn_regressor(self, testlen, ntrain):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train = np.array(traindata.drop(['code', 'date', 'closeratio'], 1))
            y_train = np.array(traindata['closeratio'])
            
            x_test = np.array(testdata.drop(['code', 'date', 'closeratio'], 1))
            #y_test = np.array(testdata['closeratio'] > 0, dtype=np.int8)
            
            ###FNN model
            model = Sequential()
            model.add(Dense(32, input_dim=x_train.shape[1], W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            model.add(Activation('relu')) #relu
            model.add(Dropout(0.5))
            #model.add(Dense(16, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            #model.add(Activation('tanh'))
            #model.add(Dropout(0.5))
            #model.add(Dense(8, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            #model.add(Activation('tanh'))
            #model.add(Dropout(0.5))
            #model.add(Dense(16, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            #model.add(Activation('tanh'))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))#sigmoid tanh
            
            model.compile(optimizer='rmsprop', loss='mean_squared_error',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_train, nb_epoch=100, batch_size=100000)
            
            testdata['predratio'] = model.predict(x_test)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)        

    def keraslstm_classifier(self, testlen, ntrain, timesteps, r):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train, y_train = self.sequencedata(traindata, timesteps)
            y_traint = np.array(y_train[:, 2] > r, dtype=np.int8)
            
            x_test, y_test = self.sequencedata(testdata, timesteps)
            y_test = pd.DataFrame(y_test)
            y_test.columns = ['code', 'date', 'closeratio']
            
            ###FNN model
            model = Sequential()
            model.add(LSTM(32, input_shape=(timesteps, x_train.shape[2]), W_regularizer=l2(0.01)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            #model.add(Dense(32, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
            #model.add(Activation('relu'))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))
            
            model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_traint, nb_epoch=100, batch_size=1000)
            
            y_test['predratio'] = model.predict_on_batch(x_test)
            y_test['predclass'] = model.predict_classes(x_test)
            
            hsma = pd.concat([hsma, y_test], ignore_index = True)

        return(hsma) 
        
    def keraslstm_regressor1(self, testlen, ntrain, timesteps):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train, y_train = self.sequencedata(traindata, timesteps)
            y_traint = np.array(y_train[:, 2])
            
            x_test, y_test = self.sequencedata(testdata, timesteps)
            y_test = pd.DataFrame(y_test)
            y_test.columns = ['code', 'date', 'closeratio']
            
            ###FNN model
            model = Sequential()
            model.add(LSTM(32, input_shape=(timesteps, x_train.shape[2]), W_regularizer=l2(0.01)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            #model.add(LSTM(32, W_regularizer=l2(0.01)))
            #model.add(Activation('tanh'))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))
            
            model.compile(optimizer='rmsprop', loss='mean_squared_error',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_traint, nb_epoch=100, batch_size=1000)
            
            y_test['predratio'] = model.predict(x_test)
            
            hsma = pd.concat([hsma, y_test], ignore_index = True)

        return(hsma) 
        
    def keraslstm_regressor1_up1(self, testlen, ntrain, timesteps):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        hsmadata = hsmadata[hsmadata['roc_1'] > 0]        


        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train, y_train = self.sequencedata(traindata, timesteps)
            y_traint = np.array(y_train[:, 2])
            
            x_test, y_test = self.sequencedata(testdata, timesteps)
            y_test = pd.DataFrame(y_test)
            y_test.columns = ['code', 'date', 'closeratio']
            
            ###FNN model
            model = Sequential()
            model.add(LSTM(32, input_shape=(timesteps, x_train.shape[2]), W_regularizer=l2(0.01)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            #model.add(LSTM(32, W_regularizer=l2(0.01)))
            #model.add(Activation('tanh'))
            #model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))
            
            model.compile(optimizer='rmsprop', loss='mean_squared_error',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_traint, nb_epoch=100, batch_size=1000)
            
            y_test['predratio'] = model.predict(x_test)
            
            hsma = pd.concat([hsma, y_test], ignore_index = True)

        return(hsma) 
        
    def keraslstm_regressorn(self, testlen, ntrain, timesteps):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(dates[i*testlen])
            
            x_train, y_train = self.sequencedata(traindata, timesteps)
            y_traint = np.array(y_train[:, 2])
            
            x_test, y_test = self.sequencedata(testdata, timesteps)
            y_test = pd.DataFrame(y_test)
            y_test.columns = ['code', 'date', 'closeratio']
            
            ###FNN model
            model = Sequential()
            model.add(LSTM(32, input_shape=(timesteps, x_train.shape[2]), return_sequences=True, W_regularizer=l2(0.01)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            model.add(LSTM(32, W_regularizer=l2(0.01)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('linear'))
            
            model.compile(optimizer='rmsprop', loss='mean_squared_error',
                          metrics=['accuracy'])
            
            model.fit(x_train, y_traint, nb_epoch=100, batch_size=1000)
            
            y_test['predratio'] = model.predict(x_test)
            
            hsma = pd.concat([hsma, y_test], ignore_index = True)

        return(hsma) 

    def hsmatrade(self, hsma, n):
        #根据指标选股          
        dates = pd.Series(hsma['date'].unique()).sort_values()

        if not self.absratio:
            LTindexoo = pd.read_csv("strategy data//LTindexoo_000905.SH.csv")
            LTindexoo = LTindexoo[['date', "ooratio" + str(self.day)]]
            hsma1 = pd.merge(hsma,LTindexoo)
            hsma1['closeratio'] = hsma1['closeratio'] + hsma1["ooratio" + str(self.day)]
            hsma1 = hsma1.drop(["ooratio" + str(self.day)],1)
        else:
            hsma1 = hsma
        
        hsmatrade = pd.DataFrame()
        for d in dates:
            hsmad = hsma1[hsma1['date'] == d].copy()
            hsmad['rank'] = hsmad['predratio'].rank(ascending=False)
            hsmatrade = pd.concat([hsmatrade,hsmad[hsmad['rank'] <= n]], ignore_index = True)

        hsmatrade['closeratio'] = hsmatrade['closeratio'] - self.fee
        print(hsmatrade['closeratio'].describe())
        
        return(hsmatrade)
        
    def hsmatradeday(self, hsma, n, idxma, stockFF):
        
        hsmatrade = self.hsmatrade(hsma, n)
        
        hsmatradeday = hsmatrade.groupby(['date'], as_index=False)[['closeratio']].mean()
        hsmatradeday['stockcloseratio'] = hsmatradeday['closeratio'] / self.day

        stockFF = pd.read_csv("strategy data\\" + stockFF + ".csv")
        stockFF = stockFF[['date','closeratio']]
        stockFF['FFcloseratio'] = stockFF['closeratio'].shift(-1)
        stockFF = stockFF.drop(["closeratio"],1)
        stockFF = stockFF.dropna()

        hsmatradeday = pd.merge(hsmatradeday,stockFF)
        
        #均线以下的交易日股票不入场，同时相应缩减对冲的期指头寸
        if idxma != 0:
            LTindex = pd.read_csv("strategy data\\LTindex_000905.SH.csv")
            LTindex = LTindex[['date', 'ma_1', 'ma_' + str(idxma)]]
            LTindex.columns = ['date', 'idxclose', 'idxma']
            hsmatradeday = pd.merge(hsmatradeday, LTindex)
            hsmatradeday['position'] = 0
            
            for i in hsmatradeday.index:
                if hsmatradeday.ix[i, 'idxclose'] < hsmatradeday.ix[i, 'idxma']:   
                    hsmatradeday.ix[i, 'stockcloseratio'] = 0
                else:
                    hsmatradeday.ix[i:min(max(hsmatradeday.index), i+self.day-1), 'position'] += 1/self.day                     

            hsmatradeday['FFcloseratio'] = hsmatradeday['FFcloseratio'] * hsmatradeday['position']
            
        hsmatradeday['stockratio'] = hsmatradeday['stockcloseratio'].cumsum()
        hsmatradeday['FFratio'] = hsmatradeday['FFcloseratio'].cumsum()

        hsmatradeday['ratio'] = (hsmatradeday['stockcloseratio'] + hsmatradeday['FFcloseratio']) / 2
        hsmatradeday['cumratio'] = (hsmatradeday['stockratio'] + hsmatradeday['FFratio']) / 2
        
        print(hsmatradeday['ratio'].describe())
        plt.plot(hsmatradeday['cumratio'])
        return(hsmatradeday)

    def hsmatradeclassifier(self, hsma, minn):
        #根据指标选股          
        hsma = hsma[hsma['predratio'] > 0.5]
        dates = pd.Series(hsma['date'].unique()).sort_values()
        
        hsmatrade = pd.DataFrame()
        for d in dates:
            hsmad = hsma[(hsma['date'] == d) & (hsma['predratio'] == 1)].copy()
            if hsmad.shape[0] == 0:
                continue
            hsmad['closeratio'] = hsmad['closeratio'] - self.fee
            if hsmad.shape[0] < minn:
                hsmad['closeratio'] = hsmad['closeratio'] * (1 / minn)
            else:
                hsmad['closeratio'] = hsmad['closeratio'] * (1 / hsmad.shape[0])
            hsmatrade = pd.concat([hsmatrade,hsmad], ignore_index = True)
        
        print(hsmatrade['closeratio'].describe())
        
        return(hsmatrade) 

    def hsmatradedayclassifier(self, hsma, minn):
        
        hsmatrade = self.hsmatradeclassifier(hsma, minn)
        
        hsmatradeday = hsmatrade.groupby(['date'], as_index=False)[['closeratio']].sum()
        hsmatradeday['ratio'] = hsmatradeday['closeratio'] / self.day

       
        hsmatradeday['cumratio'] = hsmatradeday['ratio'].cumsum()
   
        print(hsmatradeday['ratio'].describe())
        plt.plot(hsmatradeday['cumratio'])
        return(hsmatradeday)        
        
        
    def hsmatradedayclassifier_hedge(self, hsma, minn, idxma):
        
        hsmatrade = self.hsmatradeclassifier(hsma, minn)
        
        hsmatradeday = hsmatrade.groupby(['date'], as_index=False)[['closeratio']].sum()
        hsmatradeday['stockcloseratio'] = hsmatradeday['closeratio'] / self.day

        stockFF = pd.read_csv("strategy data\\stockFF.csv")
        stockFF = stockFF[['date','closeratio']]
        stockFF = stockFF.rename(columns = {"closeratio":'FFcloseratio'})

        hsmatradeday = pd.merge(hsmatradeday,stockFF)
        
        #均线以下的交易日股票不入场，同时相应缩减对冲的期指头寸
        if idxma != 0:
            LTindex = pd.read_csv("strategy data\\LTindex_000905.SH.csv")
            LTindex = LTindex[['date', 'ma_1', 'ma_' + str(idxma)]]
            LTindex.columns = ['date', 'idxclose', 'idxma']
            hsmatradeday = pd.merge(hsmatradeday, LTindex)
            hsmatradeday['position'] = 0
            
            for i in hsmatradeday.index:
                if hsmatradeday.ix[i, 'idxclose'] < hsmatradeday.ix[i, 'idxma']:   
                    hsmatradeday.ix[i, 'stockcloseratio'] = 0
                else:
                    hsmatradeday.ix[i:min(max(hsmatradeday.index), i+self.day-1), 'position'] += 1/self.day                     

            hsmatradeday['FFcloseratio'] = hsmatradeday['FFcloseratio'] * hsmatradeday['position']
            
        hsmatradeday['stockratio'] = hsmatradeday['stockcloseratio'].cumsum()
        hsmatradeday['FFratio'] = hsmatradeday['FFcloseratio'].cumsum()

        hsmatradeday['ratio'] = (hsmatradeday['stockcloseratio'] + hsmatradeday['FFcloseratio']) / 2
        hsmatradeday['cumratio'] = (hsmatradeday['stockratio'] + hsmatradeday['FFratio']) / 2
        
        print(hsmatradeday['ratio'].describe())
        plt.plot(hsmatradeday['cumratio'])
        return(hsmatradeday)        
        
    def tradestat(self, hsmatradeday, stockFF):
        ###对收益曲线的统计
        tradestat = pd.DataFrame({'startdate' : [min(hsmatradeday['date'])], 'enddate' : [max(hsmatradeday['date'])]})
        tradestat['ratio'] = hsmatradeday.ix[hsmatradeday.shape[0] - 1,'cumratio']
        
        tradestat['meandayratio'] = hsmatradeday['ratio'].mean()
        
        mdd = 0
        mdddate = 0
        hsmatradeday['year'] = 0
        for i in hsmatradeday.index:
            hsmatradeday.ix[i, 'year'] = int(str(hsmatradeday.ix[i,'date'])[0:4])
            mdd1 = hsmatradeday.ix[i, 'cumratio'] - min(hsmatradeday.ix[i:, 'cumratio'])
            if mdd1 > mdd:
                mdd = mdd1
                mdddate = hsmatradeday.ix[i, 'date']
            
        for year in range(2010, 2017):
            temp = hsmatradeday[hsmatradeday['year'] == year]
            temp.index = range(0, temp.shape[0])
            tradestat[str(year) + 'ratio'] = sum(temp['ratio'])
            
        stockFF = pd.read_csv("strategy data\\" + stockFF + ".csv")
        totdays = sum((stockFF['date'] >= min(hsmatradeday['date'])) & (stockFF['date'] <= max(hsmatradeday['date'])))
        tradestat['yearratio'] = tradestat['ratio'] / totdays * 252
        tradestat['mdd'] = mdd
        tradestat['mdddate'] = mdddate
        tradestat['RRR'] = tradestat['yearratio'] / tradestat['mdd']
        
        tradestat['sharpratio'] = hsmatradeday['ratio'].mean() / hsmatradeday['ratio'].std() * 252**0.5
        
        print(tradestat)
        return(tradestat)

    def tradestatlist(self, hsma, n, idxma=0):#n=200,100,50

        tradestatlist = pd.DataFrame()
        stockFF = "stockFFDHtaq_test1"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        stockFF = "stockFFDHtaqpbsr_test1"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        stockFF = "stockFFDHmbttshorton2_test1"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        stockFF = "stockFFDHmbttshorton2_test2"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        stockFF = "stockFFDHlvdongbinshorton2_test1"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        stockFF = "stockFFDHlvdongbinshorton2_test2"
        hsmatradeday = self.hsmatradeday(hsma, n, idxma, stockFF)
        tradestat =  self.tradestat(hsmatradeday, stockFF)
        tradestat['stockFF'] = stockFF
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)       

        tradestatlist.to_csv("Test\\dynamic hedge\\testresult\\DynamicHedge_" + self.label + ".csv",index=False)

            
    def extratreestradelist(self, testlen, ntrain, ntrees, n):
        hsma = self.wsddataall    
        dates = pd.Series(hsma['date'].unique()).sort_values()
        ntest = len(dates) // testlen       
        
        i = ntest
        traindata = hsma[(hsma['date'] >= dates[(i-ntrain)*testlen]) & (hsma['date'] < dates[i*testlen])].copy()
        testdata = hsma[hsma['date'] >= dates[i*testlen - 1]].copy()        

        traindata = traindata.dropna(axis=1)
        traindata = traindata.iloc[:, 16:]
        traindatax = traindata.drop(['dayration'], 1)
        traindatay = traindata['dayration']            
        testdatax = testdata[traindatax.columns]   

        treemodel = ExtraTreesRegressor(n_estimators=ntrees)
        treemodel.fit(traindatax, traindatay)
        testdata['preddayratio'] = treemodel.predict(testdatax)
        
        temp = testdata[testdata['date'] == self.enddate]
        temp = temp.sort_values(by='preddayratio')
        temp.index = range(0, temp.shape[0])
        hsmashort = temp[0:n].copy()
        hsmashort['ls'] = -1
        hsmalong = temp[(temp.shape[0]-n):temp.shape[0]].copy()
        hsmalong['ls'] = 1
        hsmals = pd.concat([hsmalong, hsmashort]) 
        hsmals = hsmals[['date','code','rawcode','ls']]
        hsmals.to_csv("momentumhedge_wsd\\hedgelist\\extratreestradelist_" + self.label + "_" + self.enddate + ".csv",index=False)

        return(hsmals)
        
        
        
        
        