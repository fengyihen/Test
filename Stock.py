# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import talib
import matplotlib.pyplot as plt


class Stock():
    
    #所有股票所有日期上的技术指标统计，排序后
    
    def __init__(self, dataset, absratio, day, length, timesteps, mtype, fee, indexfee, label):
        self.dataset = dataset  
        self.absratio = absratio
        self.day = day
        self.length = length
        self.timesteps = timesteps
        self.mtype = mtype
        self.fee = fee
        self.indexfee = indexfee
        self.label = label
        if mtype == 'cnn1D':
            self.hsmadata, self.ratio_mean = self.hsmadata_cnn1D()
        elif mtype == 'None':
            self.hsmadata, self.ratio_mean = 0, 0
        else:
            self.hsmadata, self.ratio_mean = self.hsmadata_ta()
        self.hsmadata_index = self.hsmadata_index()
    
    def hsmadata_ta(self):        
        
        marketdata = pd.read_csv("raw data\\stock\\" + self.dataset + ".csv")
        codelist = pd.Series(marketdata['code'].unique()).sort_values()
 
        hsmadata = pd.DataFrame()
        for code in codelist:
            hsma0 = marketdata[marketdata['code'] == code].copy()  
            hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date')
            
            hsma0['ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
            hsma0['ocratio'] = hsma0['open'].shift(-1)/hsma0['close'] - 1
            
            hsma = hsma0[['date', 'code', 'ratio']].copy()
            for l in self.length:                
                hsma['ROC_' + str(l)] = talib.ROC(hsma0.close.values, timeperiod = l)
                if l == 1:
                    hsma['freeturn_1'] = hsma0['free_turn']
                else:
                    hsma['freeturn_' + str(l)] = talib.SUM(hsma0.free_turn.values, timeperiod = l)
                if l > 2:
                    hsma['CCI_' + str(l)] = talib.CCI(hsma0.high.values,hsma0.low.values,hsma0.close.values, timeperiod = l)
                    hsma['ADX_' + str(l)] = talib.ADX(hsma0.high.values,hsma0.low.values,hsma0.close.values, timeperiod = l)
                    hsma['MFI_' + str(l)] = talib.MFI(hsma0.high.values,hsma0.low.values,hsma0.close.values,hsma0.amt.values, timeperiod = l)
                    hsma['RSI_' + str(l)] = talib.RSI(hsma0.close.values, timeperiod = l)
                    hsma['VAR_' + str(l)] = talib.VAR(hsma0.close.values, timeperiod = l)
                    
            hsma = hsma[hsma0['ocratio'] < 0.09]
            hsmadata = pd.concat([hsmadata, hsma], ignore_index = True)
            
        hsmadata = hsmadata.dropna()
        
        hsmamean = pd.DataFrame()
        for i in range(3, hsmadata.shape[1]):
            temp = hsmadata.ix[:, [0, i]]
            temp = temp.groupby(['date'], as_index=False)[[temp.columns[1]]].mean()
            temp = temp.sort_values(by='date')
            temp = temp.rename(columns = {temp.columns[1]:(temp.columns[1] + '_mean')})
            if hsmamean.shape[0] == 0:
                hsmamean = temp
            else:
                hsmamean = pd.merge(hsmamean, temp)
        
        hsmadata = pd.merge(hsmadata, hsmamean)
        
        if not self.absratio:
            temp = hsmadata[['date', 'ratio']]
            ratio_mean = temp.groupby(['date'], as_index=False)[['ratio']].mean()
            ratio_mean = ratio_mean.sort_values(by='date')
            ratio_mean = ratio_mean.rename(columns = {'ratio':('ratio_mean')})
            hsmadata = pd.merge(hsmadata, ratio_mean)
            hsmadata.ratio = hsmadata.ratio - hsmadata.ratio_mean
            hsmadata = hsmadata.drop(['ratio_mean'], 1)
        else:
            ratio_mean = 0
        
        return hsmadata, ratio_mean 

    def hsmadata_cnn1D(self):
        
        marketdata = pd.read_csv("raw data\\stock\\" + self.dataset + ".csv")
        codelist = pd.Series(marketdata['code'].unique()).sort_values()
 
        hsmadata = pd.DataFrame()
        for code in codelist:
            hsma0 = marketdata[marketdata['code'] == code].copy()  
            hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date')
            
            hsma0['ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
            hsma0['ocratio'] = hsma0['open'].shift(-1)/hsma0['close'] - 1
            hsma0['openr'] = talib.ROC(hsma0.open.values, timeperiod = 1)
            hsma0['highr'] = talib.ROC(hsma0.high.values, timeperiod = 1)
            hsma0['lowr'] = talib.ROC(hsma0.low.values, timeperiod = 1)
            hsma0['closer'] = talib.ROC(hsma0.close.values, timeperiod = 1)
            hsma0['amtr'] = talib.ROC(hsma0.amt.values, timeperiod = 1)
            
            hsma = hsma0[['date', 'code', 'ratio']].copy()
            for l in range(self.timesteps):                
                hsma['openr_' + str(l)] = hsma0['openr'].shift(l)
                hsma['highr_' + str(l)] = hsma0['highr'].shift(l)
                hsma['lowr_' + str(l)] = hsma0['lowr'].shift(l)
                hsma['closer_' + str(l)] = hsma0['closer'].shift(l)
                hsma['amtr_' + str(l)] = hsma0['amtr'].shift(l)
                hsma['freeturn_' + str(l)] = hsma0['free_turn'].shift(l)
                    
            hsma = hsma[hsma0['ocratio'] < 0.09]
            hsmadata = pd.concat([hsmadata, hsma], ignore_index = True)
            
        hsmadata = hsmadata.dropna()
        
        hsmamean = pd.DataFrame()
        for i in range(3, hsmadata.shape[1]):
            temp = hsmadata.ix[:, [0, i]]
            temp = temp.groupby(['date'], as_index=False)[[temp.columns[1]]].mean()
            temp = temp.sort_values(by='date')
            temp = temp.rename(columns = {temp.columns[1]:(temp.columns[1] + '_mean')})
            if hsmamean.shape[0] == 0:
                hsmamean = temp
            else:
                hsmamean = pd.merge(hsmamean, temp)
        
        hsmadata = pd.merge(hsmadata, hsmamean)
        
        if not self.absratio:
            temp = hsmadata[['date', 'ratio']]
            ratio_mean = temp.groupby(['date'], as_index=False)[['ratio']].mean()
            ratio_mean = ratio_mean.sort_values(by='date')
            ratio_mean = ratio_mean.rename(columns = {'ratio':('ratio_mean')})
            hsmadata = pd.merge(hsmadata, ratio_mean)
            hsmadata.ratio = hsmadata.ratio - hsmadata.ratio_mean
            hsmadata = hsmadata.drop(['ratio_mean'], 1)
        else:
            ratio_mean = 0
        
        return hsmadata, ratio_mean 
        
    def hsmadata_index(self):
        
        hsmadata = pd.DataFrame()
        for index in ['000300.SH', '000905.SH']:
                    
            hsma0 = pd.read_csv("raw data\\stock\\indexday_" + index + ".csv")        
            hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
            
            hsma0['ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1                     
            hsma0['ocratio'] = hsma0['open'].shift(-1)/hsma0['close'] - 1
            hsma0['amt'] = hsma0['amt'] + 0.0
            hsma0['vol'] = hsma0['vol'] + 0.0
            
            hsma = hsma0[['date', 'code', 'ratio']].copy()
            for l in self.length:                
                hsma['ROC_' + str(l)] = talib.ROC(hsma0.close.values, timeperiod = l)
                if l > 2:
                    hsma['CCI_' + str(l)] = talib.CCI(hsma0.high.values,hsma0.low.values,hsma0.close.values, timeperiod = l)
                    hsma['ADX_' + str(l)] = talib.ADX(hsma0.high.values,hsma0.low.values,hsma0.close.values, timeperiod = l)
                    hsma['MFI_' + str(l)] = talib.MFI(hsma0.high.values,hsma0.low.values,hsma0.close.values,hsma0.amt.values, timeperiod = l)
                    hsma['RSI_' + str(l)] = talib.RSI(hsma0.close.values, timeperiod = l)
                    hsma['VAR_' + str(l)] = talib.VAR(hsma0.close.values, timeperiod = l)
                    
            hsma = hsma[abs(hsma0['ocratio']) < 0.09]
            hsmadata = pd.concat([hsmadata, hsma], ignore_index = True)
            
        hsmadata = hsmadata.dropna()
        
        return hsmadata
       
    def binandwoe_traintest(self, traindatax, traindatay, testdatax, binn, bq, r0):
        #进行粗分类和woe转换
        #进行粗分类（bin）时，bq=True对连续变量等分位数分段，bp=False对连续变量等宽分段
        #先对X_train进行粗分类和woe转换，然后根据X_train的分类结果对X_test进行粗分类和woe转换
        traindatax = traindatax.copy()
        traindatay = traindatay.copy()
        testdatax = testdatax.copy()
        for col in traindatax.columns:            
   
            #按等分位数还是等宽分类
            if bq == True:
                arrayA = np.arange(0,100,100/binn)
                arrayB = np.array([100]);
                arrayA = np.concatenate((arrayA,arrayB)) 
                breakpoints = np.unique(np.percentile(traindatax[col],arrayA))
                if len(breakpoints) == 2:
                    breakpoints = np.array([breakpoints[0], np.mean(breakpoints), breakpoints[1]])
            else:
                minvalue = traindatax[col].min()
                maxvalue = traindatax[col].max()
                breakpoints = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/binn) 
                breakpoints = np.append(breakpoints, maxvalue)
            #分段并标识为相应标签labels    
            labels = np.arange(len(breakpoints) - 1)
            traindatax[col] = pd.cut(traindatax[col],bins=breakpoints,right=True,labels=labels,include_lowest=True)
            traindatax[col] = traindatax[col].astype('object')
            testdatax[col] = pd.cut(testdatax[col],bins=breakpoints,right=True,labels=labels,include_lowest=True)
            testdatax[col] = testdatax[col].astype('object')
            
            #woe转换
            #对test中出现但没在train中出现的值，woe取值为0
            xtrainunique = traindatax[col].unique()
            xtestunique = testdatax[col].unique()
            for cat in xtestunique:
                if not any(xtrainunique == cat):
                    testdatax[col] = testdatax[col].replace({cat:0})
           
            #对train中数据做woe转换，并对test中数据做相同的转换
            for cat in xtrainunique:
                #计算单个分类的woe  
                nob = max(1, sum((traindatay == 1) & (traindatax[col] == cat)))
                tnob = sum(traindatay == 1)
                nog = max(1, sum((traindatay == 0) & (traindatax[col] == cat)))
                tnog = sum(traindatay == 0)
                woei = np.log((nob/tnob)/(nog/tnog))
                traindatax[col] = traindatax[col].replace({cat:woei})
                if any(xtestunique == cat):
                    testdatax[col] = testdatax[col].replace({cat:woei})

                    
        return traindatax, testdatax
        
    def conditionrank(self, traindata, col, tn, ascending):
        dates = pd.Series(traindata['date'].unique()).sort_values()
        
        traindatac = pd.DataFrame()
        for d in dates:
            temp = traindata[traindata['date'] == d].copy()
            temp['rank'] = temp[col].rank(ascending=ascending)
            traindatac = pd.concat([traindatac,temp[temp['rank'] <= tn]], ignore_index = True)
        
        return(traindatac)
        
    def hsmatraderegressor_clear(self, condition, hsma, n, cr):
        
        hsmatrade = pd.DataFrame()
      
        dates = pd.Series(hsma['date'].unique()).sort_values()    
    
        for d in dates:
            hsmad = hsma[hsma['date'] == d].copy()
            if condition == 'ROC_1':
                hsmad = hsmad[hsmad.ROC_1 < 0.04]
            elif condition == 'ADX_3':
                hsmad = hsmad[hsmad.ADX_3 > 30]
            elif condition == 'ADX_10':
                hsmad = hsmad[hsmad.ADX_10 > 30]
            else:
                hsmad = hsmad           
            if hsmad.shape[0] == 0:
                continue
            hsmad['rank'] = hsmad['predratio'].rank(ascending=False)
            hsmatrade = pd.concat([hsmatrade,hsmad[hsmad['rank'] <= n]], ignore_index = True)

        if not self.absratio:
            hsmatrade = pd.merge(hsmatrade, self.ratio_mean)
            hsmatrade.ratio = hsmatrade.ratio + hsmatrade.ratio_mean
            
        hsmatrade['ratio'] = hsmatrade['ratio'] - self.fee
        
        hsmatrade.ix[hsmatrade['predratio'] < cr, 'ratio'] = 0
            
        print(hsmatrade['ratio'].describe())
        
        return(hsmatrade)
        
    def hsmatraderegressor(self, condition, hsma, n):
        
        hsmatrade = pd.DataFrame()
      
        dates = pd.Series(hsma['date'].unique()).sort_values()    
    
        for d in dates:
            hsmad = hsma[hsma['date'] == d].copy()
            if condition == 'ROC_1':
                hsmad = hsmad[hsmad.ROC_1 < 0.04]
            elif condition == 'ADX_3':
                hsmad = hsmad[hsmad.ADX_3 > 30]
            elif condition == 'ADX_10':
                hsmad = hsmad[hsmad.ADX_10 > 30]
            else:
                hsmad = hsmad           
            if hsmad.shape[0] == 0:
                continue
            hsmad['rank'] = hsmad['predratio'].rank(ascending=False)
            hsmatrade = pd.concat([hsmatrade,hsmad[hsmad['rank'] <= n]], ignore_index = True)

        if not self.absratio:
            hsmatrade = pd.merge(hsmatrade, self.ratio_mean)
            hsmatrade.ratio = hsmatrade.ratio + hsmatrade.ratio_mean
            
        hsmatrade['ratio'] = hsmatrade['ratio'] - self.fee
            
        print(hsmatrade['ratio'].describe())
        
        return(hsmatrade)
        
    def hsmatradedayregressor(self, condition, hsma, n, cr=None, cta1=None, cta2=None):
        
        if cr == None:
            hsmatrade = self.hsmatraderegressor(condition, hsma, n)
        else:
            hsmatrade = self.hsmatraderegressor_clear(condition, hsma, n, cr)
            
        hsmatradeday = hsmatrade.groupby(['date'], as_index=False)[['ratio']].mean()
        number = hsmatrade.groupby(['date'], as_index=False)[['ratio']].size()
        number.index = range(len(number))
        hsmatradeday['number'] = number
        hsmatradeday['dayratio'] = hsmatradeday['ratio'] / self.day * (hsmatradeday['number'] / n)       
        hsmatradeday['cumratio'] = hsmatradeday['dayratio'].cumsum()

        print('dayratio:\n', hsmatradeday['dayratio'].describe())
        
        plt.plot(hsmatradeday['cumratio'], label='stock')
        plt.legend(loc='upper left')
        
        if not self.absratio:
            index300 = pd.read_csv("raw data\\stock\\indexday_000300.SH.csv")
            index300 = index300[(index300.date >= hsmatradeday.date.min()) & (index300.date <= hsmatradeday.date.max())]
            index300.index = range(index300.shape[0])
            index300['300dayratio'] = 1 - index300.close / index300.close.shift(1) 
            index300.ix[0, '300dayratio'] = 1 - index300.close[0] / index300.open[0]
            index300['300ratio'] = index300['300dayratio'].cumsum()
            hsmatradeday = pd.merge(hsmatradeday, index300[['date', '300dayratio', '300ratio']])
            hsmatradeday['hedge300dayratio'] = (hsmatradeday.dayratio + hsmatradeday['300dayratio']) / 2            
            hsmatradeday['hedge300ratio'] = (hsmatradeday.cumratio + hsmatradeday['300ratio']) / 2
            print('hedge300dayratio:\n', hsmatradeday['hedge300dayratio'].describe())
            plt.plot(hsmatradeday['hedge300ratio'], label='hedge300')
            plt.legend(loc='upper left')

            index500 = pd.read_csv("raw data\\stock\\indexday_000905.SH.csv")
            index500 = index500[(index500.date >= hsmatradeday.date.min()) & (index500.date <= hsmatradeday.date.max())]
            index500.index = range(index500.shape[0])
            index500['500dayratio'] = 1 - index500.close / index500.close.shift(1) 
            index500.ix[0, '500dayratio'] = 1 - index500.close[0] / index500.open[0]
            index500['500ratio'] = index500['500dayratio'].cumsum()
            hsmatradeday = pd.merge(hsmatradeday, index500[['date', '500dayratio', '500ratio']])
            hsmatradeday['hedge500dayratio'] = (hsmatradeday.dayratio + hsmatradeday['500dayratio']) / 2            
            hsmatradeday['hedge500ratio'] = (hsmatradeday.cumratio + hsmatradeday['500ratio']) / 2
            print('hedge500dayratio:\n', hsmatradeday['hedge500dayratio'].describe())
            plt.plot(hsmatradeday['hedge500ratio'], label='hedge500') 
            plt.legend(loc='upper left')
            
            if os.path.exists("Test\\strategy data\\" + cta1 + ".csv"):
                cta = pd.read_csv("Test\\strategy data\\" + cta1 + ".csv")
                cta['cta1dayratio'] = cta.net.diff(1)/(cta.open.shift(1) * 100)
                cta = cta[(cta.date >= hsmatradeday.date.min()) & (cta.date <= hsmatradeday.date.max())]
                cta.index = range(cta.shape[0])
                cta['cta1ratio'] = cta['cta1dayratio'].cumsum()
                hsmatradeday = pd.merge(hsmatradeday, cta[['date', 'cta1dayratio', 'cta1ratio']])
                hsmatradeday['hedgecta1dayratio'] = (hsmatradeday.dayratio + hsmatradeday['cta1dayratio']) / 2            
                hsmatradeday['hedgecta1ratio'] = (hsmatradeday.cumratio + hsmatradeday['cta1ratio']) / 2
                print('hedgecta1dayratio:\n', hsmatradeday['hedgecta1dayratio'].describe())
                plt.plot(hsmatradeday['hedgecta1ratio'], label='hedgecta1')  
                plt.legend(loc='upper left')
                
        return(hsmatradeday)

    def hsmatradeclassifier(self, condition, hsma):
        
        hsmatrade = hsma[hsma.predratio == 1].copy()
       
        if condition == 'roc1':
            hsmatradec = hsmatrade[hsmatrade.ROC_1 > 0]
        elif condition == 'adx10':
            hsmatradec = hsmatrade[hsmatrade.ADX_10_mean > 30]
        else:
            hsmatradec = hsmatrade

        if (not self.absratio) and (hsmatradec.shape[0] > 0):
            hsmatradec = pd.merge(hsmatradec, self.ratio_mean)
            hsmatradec.ratio = hsmatradec.ratio + hsmatradec.ratio_mean   

        hsmatradec.ratio = hsmatradec.ratio - self.fee
            
        print(hsmatradec['ratio'].describe())
        
        return(hsmatradec)
        
    def hsmatradedayclassifier(self, condition, hsma, minn):
        
        hsmatradec = self.hsmatradeclassifier(condition, hsma)
        
        hsmatradeday = hsmatradec.groupby(['date'], as_index=False)[['ratio']].mean()
        number = hsmatradec.groupby(['date'], as_index=False)[['ratio']].size()
        number.index = range(len(number))
        hsmatradeday['number'] = number
        hsmatradeday['tradenumber'] = hsmatradeday['number']
        hsmatradeday.ix[hsmatradeday.number > minn, 'tradenumber'] = minn
        hsmatradeday['dayratio'] = hsmatradeday['ratio'] / self.day * (hsmatradeday['tradenumber'] / minn)       
        hsmatradeday['cumratio'] = hsmatradeday['dayratio'].cumsum()
      
        print(hsmatradeday['dayratio'].describe())
        plt.plot(hsmatradeday['cumratio'])
        
        return(hsmatradeday) 

    def hsmaindexregressor(self, hsmaindex):

        hsmai = hsmaindex.copy()
        
        hsmai['dayratio'] = (hsmai['ratio'] - self.indexfee)/self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()
        
        return(hsmai)

    def hsmaindexregressor_short(self, hsmaindex):

        hsmai = hsmaindex.copy()
        
        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio < 0, 'dayratio'] = -hsmai.ix[hsmai.predratio < 0, 'dayratio']
        hsmai.ix[hsmai.predratio >= 0, 'dayratio'] = 0
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee)/self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()
        
        return(hsmai)
        
    def hsmaindexclassifier(self, hsmaindex):
        

        hsmai = hsmaindex.copy()
        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio == 0, 'dayratio'] = -hsmai.ix[hsmai.predratio == 0, 'dayratio']
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee)/self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()
        
        return(hsmai)

    def hsmaindexclassifier_short(self, hsmaindex):
        

        hsmai = hsmaindex.copy()
        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio == 0, 'dayratio'] = -hsmai.ix[hsmai.predratio == 0, 'dayratio']
        hsmai.ix[hsmai.predratio == 1, 'dayratio'] = 0
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee)/self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()
        
        return(hsmai)
               
    def traintestanalysisclassifier(self, hsma, testlen, ntrain):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        tttable = pd.DataFrame()
        for i in range(ntrain, ntest):
            temp = hsma[(hsma['date'] >= dates[i*testlen]) & (hsma['date'] < dates[(i+1)*testlen])].copy()        
            temp1 = temp[temp.predratio == 1]
            
            table = pd.DataFrame({'i': i, 'startdate': [dates[i*testlen]], 'enddate': [dates[(i+1)*testlen]]},
                                 columns=['i', 'startdate', 'enddate'])
            table['meanratio_all'] = temp.ratio.mean()
            table['meanratio_pred'] = temp1.ratio.mean()            
            table['number_pred'] = temp1.shape[0] 
            
            tttable = pd.concat([tttable, table], ignore_index = True)

        return(tttable)
        
    def tradestat(self, hsmatradeday):
        ###对收益曲线的统计
        tradestat = pd.DataFrame({'startdate' : [min(hsmatradeday['date'])], 'enddate' : [max(hsmatradeday['date'])]})
        tradestat['ratio'] = hsmatradeday.ix[hsmatradeday.shape[0] - 1,'cumratio']
        
        tradestat['meandayratio'] = hsmatradeday['dayratio'].mean()
        
        mdd = 0
        mdddate = 0
        hsmatradeday['year'] = 0
        for i in hsmatradeday.index:
            hsmatradeday.ix[i, 'year'] = int(str(hsmatradeday.ix[i,'date'])[0:4])
            mdd1 = hsmatradeday.ix[i, 'cumratio'] - min(hsmatradeday.ix[i:, 'cumratio'])
            if mdd1 > mdd:
                mdd = mdd1
                mdddate = hsmatradeday.ix[i, 'date']
            
        for year in range(2008, 2018):
            temp = hsmatradeday[hsmatradeday['year'] == year]
            if temp.shape[0] == 0:
                continue
            temp.index = range(0, temp.shape[0])
            tradestat[str(year) + 'ratio'] = sum(temp['dayratio'])
            
        tradestat['yearratio'] = tradestat['ratio'] / hsmatradeday.shape[0] * 252
        tradestat['mdd'] = mdd
        tradestat['mdddate'] = mdddate
        tradestat['RRR'] = tradestat['yearratio'] / tradestat['mdd']
        
        tradestat['sharpratio'] = hsmatradeday['dayratio'].mean() / hsmatradeday['dayratio'].std() * 252**0.5
        
        print(tradestat)
        return(tradestat)

    def tradestatlist(self, hsmatradeday):#n=200,100,50

        tradestatlist = pd.DataFrame()
        
        tradestat =  self.tradestat(hsmatradeday)
        tradestat['Hedge'] = 'NoHedge'
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        temp = hsmatradeday[['date','hedge300dayratio','hedge300ratio']]
        temp = temp.rename(columns = {'hedge300dayratio':'dayratio', 'hedge300ratio':'cumratio'})
        tradestat =  self.tradestat(temp)
        tradestat['Hedge'] = 'Hedge300'
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        temp = hsmatradeday[['date','hedge500dayratio','hedge500ratio']]
        temp = temp.rename(columns = {'hedge500dayratio':'dayratio', 'hedge500ratio':'cumratio'})
        tradestat =  self.tradestat(temp)
        tradestat['Hedge'] = 'Hedge500'
        tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        if any(hsmatradeday.columns == 'hedgecta1ratio'):
            temp = hsmatradeday[['date','hedgecta1dayratio','hedgecta1ratio']]
            temp = temp.rename(columns = {'hedgecta1dayratio':'dayratio', 'hedgecta1ratio':'cumratio'})
            tradestat =  self.tradestat(temp)
            tradestat['Hedge'] = 'HedgeCTA1'
            tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        print(tradestatlist)
        tradestatlist.to_csv("Test\\testresult\\" + self.label + ".csv",index=False)

            

            