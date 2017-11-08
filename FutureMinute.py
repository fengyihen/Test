# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import datetime
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt


class FutureMinute():

    #所有股票所有日期上的技术指标统计，排序后

    def __init__(self, cycle, label):
        self.feelist = {
            'i9000': 0.00024,
            'jd000': 0.0006,
            'jm000': 0.00024,
            'l9000': 8,
            'y9000': 10,
            'pp000': 0.00024,
            'm9000': 6,
            'j9000': 0.00024,
            'p9000': 10,
            'rb000': 0.0004,
            'ru000': 0.0002,
            'cu000': 0.0002,
            'al000': 12,
            'au000': 40,
            'ag000': 0.0002,
            'zn000': 12,
            'ni000': 4,
            'pb000': 0.00016,
            'hc000': 0.0004,
            'cf000': 18,
            'rm000': 6,
            'sr000': 12,
            'fg000': 12,
            'ma000': 8,
            'oi000': 10,
            'zc000': 16,
            'ta000': 12
        }
        self.handlist = {
            'i9000': 100,
            'jd000': 10,
            'jm000': 60,
            'l9000': 5,
            'y9000': 10,
            'pp000': 5,
            'm9000': 10,
            'j9000': 100,
            'p9000': 10,
            'rb000': 10,
            'ru000': 10,
            'cu000': 5,
            'al000': 5,
            'au000': 1000,
            'ag000': 15,
            'zn000': 5,
            'ni000': 1,
            'pb000': 5,
            'hc000': 10,
            'cf000': 5,
            'rm000': 10,
            'sr000': 10,
            'fg000': 20,
            'ma000': 10,
            'oi000': 10,
            'zc000': 100,
            'ta000': 5
        }
        self.minpoint = {
            'i9000': 0.5,
            'jd000': 1,
            'jm000': 0.5,
            'l9000': 5,
            'y9000': 2,
            'pp000': 1,
            'm9000': 1,
            'j9000': 0.5,
            'p9000': 2,
            'rb000': 1,
            'ru000': 5,
            'cu000': 10,
            'al000': 5,
            'au000': 0.05,
            'ag000': 1,
            'zn000': 5,
            'ni000': 10,
            'pb000': 5,
            'hc000': 1,
            'cf000': 5,
            'rm000': 1,
            'sr000': 1,
            'fg000': 1,
            'ma000': 1,
            'oi000': 2,
            'zc000': 0.2,
            'ta000': 2
        }
        self.cycle = cycle
        self.label = label
        self.hsmaall = self.hsmaall(cycle)

    def hsma0(self, code, cycle):

        hsma0 = pd.read_csv("raw data\\stock\\futures\\min1\\" + code + "_" + cycle + ".csv")
        hsma0.columns = [
            'datetime', 'open', 'high', 'low', 'close', 'vol', 'openint'
        ]
        hsma0['datetime'] = hsma0['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M"))
        hsma0['date'] = hsma0['datetime'].apply(lambda x: x.year*10000 + x.month*100 + x.day)
        hsma0['time'] = hsma0['datetime'].apply(lambda x: x.hour*100 + x.minute)
        hsma0['code'] = code
        hsma0 = hsma0[['date', 'time', 'code', 'open', 'high', 'low', 'close', 'vol', 'openint']]
        hsma0 = hsma0[hsma0['date'] >= 20100000]

        hsma0['open'] = hsma0['open'] + 0.0
        hsma0['high'] = hsma0['high'] + 0.0
        hsma0['low'] = hsma0['low'] + 0.0
        hsma0['close'] = hsma0['close'] + 0.0
        hsma0['vol'] = hsma0['vol'] + 0.0
        hsma0['openint'] = hsma0['openint'] + 0.0

        hsma0.dropna(inplace=True)
        return hsma0

    def hsmaall(self, cycle):

        hsmaall = pd.DataFrame()
        for code in self.feelist.keys():
            if os.path.exists("raw data\\stock\\futures\\min1\\" + code + "_" + cycle + ".csv"):
                print(code)
                hsma0 = self.hsma0(code, cycle)
                hsmaall = pd.concat([hsmaall, hsma0], ignore_index=True)

        return hsmaall

    def hsmadata(self, llen, slen):  #code 只能是商品

        hsmadata = pd.DataFrame()
        for code in self.hsmaall.code.unique():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0['ratio'] = 0
            hsma0['roc'] = talib.ROC(hsma0.close.values, timeperiod=slen)
            hsma0['dayr'] = hsma0.close / hsma0.open - 1
            hsma0['dayh'] = hsma0.close / hsma0.high - 1
            hsma0['dayl'] = hsma0.close / hsma0.low - 1
            hsma0['dayhl'] = hsma0.high / hsma0.low - 1
            hsma0['smar'] =  hsma0['close'] / talib.SMA(hsma0.close.values, timeperiod=llen) - 1
            hsma0['VAR'] = talib.STDDEV(hsma0.close.values, timeperiod=llen) / hsma0['close'] - 1
            hsma0['timepos'] = 0

            for d in hsma0['date'].unique():
                hsmad = hsma0[hsma0.date == d]
                hsma0.loc[hsma0.date == d, 'ratio'] = (hsmad.close.iloc[hsmad.shape[0]-1] / hsmad.close - 1).values
                hsma0.loc[hsma0.date == d, 'timepos'] = ((pd.Series(range(hsmad.shape[0])) + 1) / hsmad.shape[0]).values              

            hsma0 = hsma0.dropna()
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmatrade(self, hsma, wtime, tpr):
        
        if not 'LS' in hsma.columns:
            hsma['LS'] = 'N'
            hsma.loc[(hsma.prob_long > 0.5) & (hsma.prob_short < 0.5), 'LS'] = 'L'
            hsma.loc[(hsma.prob_long < 0.5) & (hsma.prob_short > 0.5), 'LS'] = 'S'
            
        hsmatrade = pd.DataFrame()
        for code in hsma['code'].unique():   
            hsmacode = hsma[hsma['code'] == code]
            for d in hsmacode['date'].unique():
                if not any((hsmacode['date'] == d) & (hsmacode['LS'] != 'N')):
                    continue
                hsmad = hsmacode[hsmacode['date'] == d].copy()  
                
                #开仓
                idx = np.where((hsmad['date'] == d) & (hsmad['LS'] != 'N'))[0]
                idx = idx[idx >= wtime]
                if len(idx) == 0:
                    continue
                inpos = idx[0] + 1                
                #最后一分钟不入场
                if inpos >= hsmad.shape[0]:
                    continue 
                ls = hsmad['LS'].iloc[idx[0]]
                opentime = hsmad['time'].iloc[inpos]
                openprice = hsmad['open'].iloc[inpos]
                
                #平仓
                if ls == 'L':
                    if not any(hsmad.close.iloc[inpos:] < openprice * (1-tpr)):
                        outpos = hsmad.shape[0] - 1        
                    else:
                       idx = np.where(hsmad.close.iloc[inpos:] < openprice * (1-tpr))[0]
                       outpos = min(hsmad.shape[0] - 1, idx[0] + inpos + 1)
                if ls == 'S':
                    if not any(hsmad.close.iloc[inpos:] > openprice * (1+tpr)):
                        outpos = hsmad.shape[0] - 1        
                    else:
                       idx = np.where(hsmad.close.iloc[inpos:] > openprice * (1+tpr))[0]
                       outpos = min(hsmad.shape[0] - 1, idx[0] + inpos + 1)                       
                closetime = hsmad['time'].iloc[outpos]
                closeprice = hsmad['open'].iloc[outpos]
                if ls == 'L':
                    dayratio = closeprice / openprice - 1 - self.feelist[code]
                else:
                    dayratio = 1 - closeprice / openprice - self.feelist[code]
                    
                temp = pd.DataFrame({'code' : code, 'date' : d, 'LS': ls,
                                     'opentime' : opentime,
                                     'closetime' : closetime,
                                     'openprice' : openprice,
                                     'closeprice' : closeprice,
                                     'dayratio' : dayratio
                                     },index=[0])
                hsmatrade = pd.concat([hsmatrade, temp])
        
        print(hsmatrade.dayratio.mean())
        return hsmatrade

    def portfolio(self, hsmatrade):
        portfolio = hsmatrade.groupby('date')[['date','traderatio']].mean()
        portfolio['ratio'] = portfolio['traderatio'].cumsum() 
        portfolio.index = range(portfolio.shape[0])
        return portfolio
        
    def traderatiostat(self, hsmatrade):
        tradeL = hsmatrade[hsmatrade.LS == 'L']
        tradeS = hsmatrade[hsmatrade.LS == 'S']
        tradestat = pd.DataFrame({'LS' :'ALL',
                                  'startDate' : hsmatrade.opendate.min(),
                                 'endDate' : hsmatrade.closedate.max(),
                                 'meanRatio': hsmatrade.traderatio.mean(),
                                 'sumRatio': hsmatrade.traderatio.sum(),
                                 'tradeCount': hsmatrade.shape[0],
                                 'meanTime' : hsmatrade.tradetime.mean()
                                 }, index=[0])
        tradestatL = pd.DataFrame({'LS' : 'L',
                                   'startDate' : tradeL.opendate.min(),
                                 'endDate' : tradeL.closedate.max(),
                                 'meanRatio': tradeL.traderatio.mean(),
                                 'sumRatio': tradeL.traderatio.sum(),                                 
                                 'tradeCount': tradeL.shape[0],
                                 'meanTime' : tradeL.tradetime.mean()
                                 }, index=[0])
        tradestatS = pd.DataFrame({'LS' : 'S',
                                   'startDate' : tradeS.opendate.min(),
                                 'endDate' : tradeS.closedate.max(),
                                 'meanRatio': tradeS.traderatio.mean(),
                                 'sumRatio': tradeS.traderatio.sum(),                                 
                                 'tradeCount': tradeS.shape[0],
                                 'meanTime' : tradeS.tradetime.mean()
                                 }, index=[0])
        tradestat = pd.concat([tradestat,tradestatL,tradestatS])
        
        hsmatrade['year'] = hsmatrade.opendate // 10000
        hsmatrade['month'] = hsmatrade.opendate // 100        
        
        yearmean = hsmatrade.groupby('year')[['traderatio']].mean()
        yearcount = hsmatrade.groupby('year')[['traderatio']].count() 
        yearsum = hsmatrade.groupby('year')[['traderatio']].sum() 
        yearratio = pd.merge(yearmean, yearcount, left_index=True, right_index=True)
        yearratio = pd.merge(yearratio, yearsum, left_index=True, right_index=True)
        yearratio.columns = ['meanRatio', 'Count', 'sumRatio']
        
        monthmean = hsmatrade.groupby('month')[['traderatio']].mean()
        monthcount = hsmatrade.groupby('month')[['traderatio']].count()   
        monthsum = hsmatrade.groupby('month')[['traderatio']].sum()
        monthratio = pd.merge(monthmean, monthcount, left_index=True, right_index=True)
        monthratio = pd.merge(monthratio, monthsum, left_index=True, right_index=True)
        monthratio.columns = ['meanRatio', 'Count', 'sumRatio']
        
        codemean = hsmatrade.groupby('code')[['traderatio']].mean()
        codecount = hsmatrade.groupby('code')[['traderatio']].count() 
        codesum = hsmatrade.groupby('code')[['traderatio']].sum() 
        coderatio = pd.merge(codemean, codecount, left_index=True, right_index=True)
        coderatio = pd.merge(coderatio, codesum, left_index=True, right_index=True)
        coderatio.columns = ['meanRatio', 'Count', 'sumRatio']
        coderatio = coderatio.sort_values('sumRatio')
        
        print(monthratio)
        print(coderatio)
        print(yearratio)
        print(tradestat)
        