# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt


class Future():

    #所有股票所有日期上的技术指标统计，排序后

    def __init__(self, minp, pnumber, label):
        self.feelist = {
            'i9000_day': 0.00024,
            'jd000_day': 0.0006,
            'jm000_day': 0.00024,
            'l9000_day': 8,
            'y9000_day': 10,
            'pp000_day': 0.00024,
            'm9000_day': 6,
            'j9000_day': 0.00024,
            'p9000_day': 10,
            'rb000_day': 0.0004,
            'ru000_day': 0.0002,
            'cu000_day': 0.0002,
            'al000_day': 12,
            'au000_day': 40,
            'ag000_day': 0.0002,
            'zn000_day': 12,
            'ni000_day': 4,
            'pb000_day': 0.00016,
            'hc000_day': 0.0004,
            'cf000_day': 18,
            'rm000_day': 6,
            'sr000_day': 12,
            'fg000_day': 12,
            'ma000_day': 8,
            'oi000_day': 10,
            'zc000_day': 16,
            'ta000_day': 12
        }
        self.handlist = {
            'i9000_day': 100,
            'jd000_day': 10,
            'jm000_day': 60,
            'l9000_day': 5,
            'y9000_day': 10,
            'pp000_day': 5,
            'm9000_day': 10,
            'j9000_day': 100,
            'p9000_day': 10,
            'rb000_day': 10,
            'ru000_day': 10,
            'cu000_day': 5,
            'al000_day': 5,
            'au000_day': 1000,
            'ag000_day': 15,
            'zn000_day': 5,
            'ni000_day': 1,
            'pb000_day': 5,
            'hc000_day': 10,
            'cf000_day': 5,
            'rm000_day': 10,
            'sr000_day': 10,
            'fg000_day': 20,
            'ma000_day': 10,
            'oi000_day': 10,
            'zc000_day': 100,
            'ta000_day': 5
        }
        self.minpoint = {
            'i9000_day': 0.5,
            'jd000_day': 1,
            'jm000_day': 0.5,
            'l9000_day': 5,
            'y9000_day': 2,
            'pp000_day': 1,
            'm9000_day': 1,
            'j9000_day': 0.5,
            'p9000_day': 2,
            'rb000_day': 1,
            'ru000_day': 5,
            'cu000_day': 10,
            'al000_day': 5,
            'au000_day': 0.05,
            'ag000_day': 1,
            'zn000_day': 5,
            'ni000_day': 10,
            'pb000_day': 5,
            'hc000_day': 1,
            'cf000_day': 5,
            'rm000_day': 1,
            'sr000_day': 1,
            'fg000_day': 1,
            'ma000_day': 1,
            'oi000_day': 2,
            'zc000_day': 0.2,
            'ta000_day': 2
        }
        self.minp = minp
        self.pnumber = pnumber
        self.label = label
        self.hsmaall = self.hsmaall()

    def hsma0(self, code):

        hsma0 = pd.read_csv("raw data\\stock\\futures\\day\\" + code + ".csv")
        hsma0.columns = [
            'date', 'open', 'high', 'low', 'close', 'vol', 'openint'
        ]
        hsma0['code'] = code
        temp = hsma0['date'].map(lambda x: x.split('-'))
        hsma0['date'] = temp.map(
            lambda x: int(x[0]) * 10000 + int(x[1]) * 100 + int(x[2]))
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        hsma0 = hsma0[hsma0['date'] > 20080000].copy()

        hsma0['open'] = hsma0['open'] + 0.0
        hsma0['high'] = hsma0['high'] + 0.0
        hsma0['low'] = hsma0['low'] + 0.0
        hsma0['close'] = hsma0['close'] + 0.0
        hsma0['preopen'] = hsma0['open'].shift(1)
        hsma0['preclose'] = hsma0['close'].shift(1)
        hsma0['nextopen'] = hsma0['open'].shift(-1)
        hsma0['vol'] = hsma0['vol'] + 0.0
        hsma0['openint'] = hsma0['openint'] + 0.0
        hsma0.loc[hsma0.index[hsma0.shape[0] - 1], 'nextopen'] = hsma0.loc[
            hsma0.index[hsma0.shape[0] - 1], 'close']

        hsma0.dropna(inplace=True)
        return hsma0

    def hsmaall(self):

        hsmaall = pd.DataFrame()
        for code in self.feelist.keys():
            print(code)
            hsma0 = self.hsma0(code)
            hsmaall = pd.concat([hsmaall, hsma0], ignore_index=True)

        return hsmaall

    def hsmadata_ta_x(self):  #code 只能是商品

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code]
            temp = hsma0[[
                'date',
                'code',
            ]].copy()
            for l in self.length:
                temp['ROC_' + str(l)] = talib.ROC(
                    hsma0.close.values, timeperiod=l)
                if l > 2:
                    temp['CCI_' + str(l)] = talib.CCI(
                        hsma0.high.values,
                        hsma0.low.values,
                        hsma0.close.values,
                        timeperiod=l)
                    temp['ADX_' + str(l)] = talib.ADX(
                        hsma0.high.values,
                        hsma0.low.values,
                        hsma0.close.values,
                        timeperiod=l)
                    temp['MFI_' + str(l)] = talib.MFI(
                        hsma0.high.values,
                        hsma0.low.values,
                        hsma0.close.values,
                        hsma0.vol.values,
                        timeperiod=l)
                    temp['RSI_' + str(l)] = talib.RSI(
                        hsma0.close.values, timeperiod=l)
                    temp['VAR_' + str(l)] = talib.VAR(
                        hsma0.close.values, timeperiod=l)

            temp = temp.dropna()
            hsmadata = pd.concat([hsmadata, temp], ignore_index=True)

        return hsmadata

    def hsmadata_rawcci_x(self, timesteps, length_t):  #code 只能是商品

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()

            hsma0['closer'] = hsma0['close'] / hsma0['preclose'] - 1
            hsma0['volr'] = hsma0['vol'] / hsma0['vol'].shift(1) - 1
            hsma0['CCI'] = talib.CCI(
                hsma0.high.values,
                hsma0.low.values,
                hsma0.close.values,
                timeperiod=length_t)
            hsma0 = hsma0.dropna()

            temp = hsma0[['date', 'code', 'closer', 'volr', 'CCI']].copy()
            temp.columns = ['date', 'code', 'closer_0', 'volr_0', 'CCI_0']

            for i in range(1, timesteps):
                temp['closer_' + str(i)] = hsma0['closer'].shift(i)
                temp['volr_' + str(i)] = hsma0['volr'].shift(i)
                temp['CCI_' + str(i)] = hsma0['CCI'].shift(i)

            temp = temp.dropna()
            hsmadata = pd.concat([hsmadata, temp], ignore_index=True)

        return hsmadata

    def hsmadata_raw_x(self, timesteps):  #code 只能是商品

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()

            hsma0['openr'] = hsma0['open'] / hsma0['preclose'] - 1
            hsma0['highr'] = hsma0['high'] / hsma0['preclose'] - 1
            hsma0['lowr'] = hsma0['low'] / hsma0['preclose'] - 1
            hsma0['closer'] = hsma0['close'] / hsma0['preclose'] - 1
            hsma0['volr'] = hsma0['vol'] / hsma0['vol'].shift(1) - 1
            hsma0[
                'openintr'] = hsma0['openint'] / hsma0['openint'].shift(1) - 1
            hsma0 = hsma0.dropna()

            temp = hsma0[[
                'date', 'code', 'openr', 'highr', 'lowr', 'closer', 'volr',
                'openintr'
            ]].copy()
            temp.columns = [
                'date', 'code', 'openr_0', 'highr_0', 'lowr_0', 'closer_0',
                'volr_0', 'openintr_0'
            ]

            for i in range(1, timesteps):
                temp['openr_' + str(i)] = hsma0['openr'].shift(i)
                temp['highr_' + str(i)] = hsma0['highr'].shift(i)
                temp['lowr_' + str(i)] = hsma0['lowr'].shift(i)
                temp['closer_' + str(i)] = hsma0['closer'].shift(i)
                temp['volr_' + str(i)] = hsma0['volr'].shift(i)
                temp['openintr_' + str(i)] = hsma0['openintr'].shift(i)

            temp = temp.dropna()
            hsmadata = pd.concat([hsmadata, temp], ignore_index=True)

        return hsmadata

    def hsmadata_raw_x2(self, timesteps):  #code 只能是商品

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()

            hsma0['openr'] = hsma0['open'] / hsma0['open'].shift(1) - 1
            hsma0['highr'] = hsma0['high'] / hsma0['high'].shift(1) - 1
            hsma0['lowr'] = hsma0['low'] / hsma0['low'].shift(1) - 1
            hsma0['closer'] = hsma0['close'] / hsma0['preclose'] - 1
            hsma0['volr'] = hsma0['vol'] / hsma0['vol'].shift(1) - 1
            hsma0[
                'openintr'] = hsma0['openint'] / hsma0['openint'].shift(1) - 1
            hsma0 = hsma0.dropna()

            temp = hsma0[[
                'date', 'code', 'openr', 'highr', 'lowr', 'closer', 'volr',
                'openintr'
            ]].copy()
            temp.columns = [
                'date', 'code', 'openr_0', 'highr_0', 'lowr_0', 'closer_0',
                'volr_0', 'openintr_0'
            ]

            for i in range(1, timesteps):
                temp['openr_' + str(i)] = hsma0['openr'].shift(i)
                temp['highr_' + str(i)] = hsma0['highr'].shift(i)
                temp['lowr_' + str(i)] = hsma0['lowr'].shift(i)
                temp['closer_' + str(i)] = hsma0['closer'].shift(i)
                temp['volr_' + str(i)] = hsma0['volr'].shift(i)
                temp['openintr_' + str(i)] = hsma0['openintr'].shift(i)

            temp = temp.dropna()
            hsmadata = pd.concat([hsmadata, temp], ignore_index=True)

        return hsmadata

    def hsmadata_breakhl_fixlength(self, testlen, length, lr):  #

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0['highd'] = talib.MAX(
                hsma0.high.shift(1).values, timeperiod=length)
            hsma0['lowd'] = talib.MAX(
                hsma0.low.shift(1).values, timeperiod=length)
            hsma0 = hsma0.dropna()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['r'] = 0
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]

            #做空
            for i in range(0, hsma0.shape[0] - testlen):
                #做空
                rlist = 1 - hsma0.loc[(i + 1):(i + testlen), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + testlen), 'lowd']) - fee
                #开盘价入场
                temp = hsma0.loc[(i + 1):(i + testlen), 'open'] < hsma0.loc[(
                    i + 1):(i + testlen), 'lowd']
                rlist[temp] = 1 - hsma0.loc[(i + 1):(i + testlen), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + testlen), 'open']) - fee
                #波动不足，不交易
                temp = hsma0.loc[(i + 1):(i + testlen), 'low'] > hsma0.loc[(
                    i + 1):(i + testlen), 'lowd']
                rlist[temp] = 0
                #开盘波动太大，不交易
                temp = 1 - hsma0.loc[(i + 1):(i + testlen), 'open'] / hsma0.loc[(
                    i + 1):(i + testlen), 'preclose']
                rlist[temp > lr] = 0
                short_r = rlist.sum()

                #做多
                rlist = hsma0.loc[(i + 1):(i + testlen), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + testlen), 'highd']) - 1 - fee
                #开盘价入场
                temp = hsma0.loc[(i + 1):(i + testlen), 'open'] > hsma0.loc[(
                    i + 1):(i + testlen), 'highd']
                rlist[temp] = hsma0.loc[(i + 1):(i + testlen), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + testlen), 'open']) - 1 - fee
                #止盈，收益上限
                temp = hsma0.loc[(i + 1):(i + testlen), 'high'] < hsma0.loc[(
                    i + 1):(i + testlen), 'highd']
                rlist[temp] = 0
                #开盘波动太大，不交易
                temp = hsma0.loc[(i + 1):(i + testlen), 'open'] / hsma0.loc[(
                    i + 1):(i + testlen), 'preclose'] - 1
                rlist[temp > lr] = 0
                long_r = rlist.sum()
                
                hsma0.loc[i, 'r'] = short_r + long_r
            hsma0 = hsma0[['code', 'date', 'r']]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)
            
        return hsmadata
        
    def hsmadata_bestp(self, day, lr):  #双向交易，多空相同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['bestp'] = -1
            hsma0['bestp_r'] = 0
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            for i in range(0, hsma0.shape[0] - day):
                bestp_r = list()
                for j in range(self.pnumber):
                    #p的取值范围：list(np.arange(0.002, 0.022, 0.002))
                    p = (j + 1) * self.minp
                    #做空
                    rlist = 1 - hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] * (1 - p)) - fee
                    #波动不足，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'low'] / hsma0.loc[(i + 1):(i + day), 'open']
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'open'] / hsma0.loc[(i + 1):
                                                      (i + day), 'preclose']
                    rlist[temp > lr] = 0
                    r = rlist.sum()
                    #做多
                    rlist = hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] *
                        (1 + p)) - 1 - fee
                    #止盈，收益上限
                    temp = hsma0.loc[(i + 1):(i + day), 'high'] / hsma0.loc[(
                        i + 1):(i + day), 'open'] - 1
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                        i + 1):(i + day), 'preclose'] - 1
                    rlist[temp > lr] = 0
                    r = r + rlist.sum()
                    bestp_r.append(r)
                hsma0.loc[i, 'bestp'] = bestp_r.index(max(bestp_r))
                hsma0.loc[i, 'bestp_r'] = max(bestp_r)

            hsma0 = hsma0.loc[hsma0.bestp != -1,
                              ['code', 'date', 'bestp', 'bestp_r']]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmadata_bestp2(self, day, lr):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['bestp_long'] = -1
            hsma0['bestp_long_r'] = 0
            hsma0['bestp_short'] = -1
            hsma0['bestp_short_r'] = 0
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]

            #做空
            for i in range(0, hsma0.shape[0] - day):
                bestp_r = list()
                for j in range(self.pnumber):
                    #p的取值范围：list(np.arange(0.002, 0.022, 0.002))
                    p = (j + 1) * self.minp
                    #做空
                    rlist = 1 - hsma0.loc[(i + 1):(i + day), 'close'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] * (1 - p)) - fee
                    #波动不足，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'low'] / hsma0.loc[(i + 1):(i + day), 'open']
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'open'] / hsma0.loc[(i + 1):
                                                      (i + day), 'preclose']
                    rlist[temp > lr] = 0
                    r = rlist.sum()
                    bestp_r.append(r)
                hsma0.loc[i, 'bestp_short'] = bestp_r.index(max(bestp_r))
                hsma0.loc[i, 'bestp_short_r'] = max(bestp_r)
            #做多
            for i in range(0, hsma0.shape[0] - day):
                bestp_r = list()
                for j in range(self.pnumber):
                    #p的取值范围：list(np.arange(0.002, 0.022, 0.002))
                    p = (j + 1) * self.minp
                    #做多
                    rlist = hsma0.loc[(i + 1):(i + day), 'close'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] *
                        (1 + p)) - 1 - fee
                    #止盈，收益上限
                    temp = hsma0.loc[(i + 1):(i + day), 'high'] / hsma0.loc[(
                        i + 1):(i + day), 'open'] - 1
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                        i + 1):(i + day), 'preclose'] - 1
                    rlist[temp > lr] = 0
                    r = rlist.sum()
                    bestp_r.append(r)
                hsma0.loc[i, 'bestp_long'] = bestp_r.index(max(bestp_r))
                hsma0.loc[i, 'bestp_long_r'] = max(bestp_r)

            hsma0 = hsma0.loc[
                (hsma0.bestp_short != -1) & (hsma0.bestp_long != -1), [
                    'code', 'date', 'bestp_short', 'bestp_short_r',
                    'bestp_long', 'bestp_long_r'
                ]]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmadata_roc(self, day):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])

            hsma0['ratio'] = hsma0.close.shift(-day) / hsma0.close - 1

            hsma0 = hsma0.dropna()

            hsmadata = pd.concat(
                [hsmadata, hsma0[['code', 'date', 'ratio']]], ignore_index=True)

        return hsmadata

    def hsmadata_roo(self, day):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])

            hsma0['ratio'] = hsma0.open.shift(-day-1) / hsma0.open.shift(-1)- 1

            hsma0 = hsma0.dropna()

            hsmadata = pd.concat(
                [hsmadata, hsma0[['code', 'date', 'ratio']]], ignore_index=True)

        return hsmadata

    def hsmadata_std(self, day):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['std'] = talib.STDDEV(hsma0.close.values, timeperiod=day, nbdev=1) / hsma0.close
            hsma0['std_day'] = hsma0['std'].shift(-day)

            hsma0.dropna(inplace=True)
            hsmadata = pd.concat([hsmadata, hsma0[['code', 'date', 'std_day']]], ignore_index=True)

        return hsmadata

    def hsmadata_fixvar(self, day, v, lr):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0['std'] = talib.STDDEV(hsma0.close.values, timeperiod=day, nbdev=1)
            hsma0['std'] = hsma0['std'].shift(1)
            hsma0.dropna(inplace=True)
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['long_r'] = -1
            hsma0['short_r'] = -1
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]

            #做空
            for i in range(0, hsma0.shape[0] - day):
                #做空
                rlist = 1 - hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + day), 'open'] -  v * hsma0.loc[(i + 1):(i + day), 'std']) - fee
                #波动不足，不交易
                temp = hsma0.loc[(i + 1):(i + day), 'low'] > (hsma0.loc[(i + 1):(i + day), 'open'] -  v * hsma0.loc[(i + 1):(i + day), 'std'])
                rlist[temp] = 0
                #开盘波动太大，不交易
                temp = 1 - hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                    i + 1):(i + day), 'preclose']
                rlist[temp > lr] = 0
                hsma0.loc[i, 'short_r'] = rlist.sum()
            #做多
            for i in range(0, hsma0.shape[0] - day):
                #做多
                rlist = hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + day), 'open'] +  v * hsma0.loc[(i + 1):(i + day), 'std']) - 1 - fee
                #止盈，收益上限
                temp = hsma0.loc[(i + 1):(i + day), 'high'] < (hsma0.loc[(i + 1):(i + day), 'open'] +  v * hsma0.loc[(i + 1):(i + day), 'std'])
                rlist[temp] = 0
                #开盘波动太大，不交易
                temp = hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                    i + 1):(i + day), 'preclose'] - 1
                rlist[temp > lr] = 0
                hsma0.loc[i, 'long_r'] = rlist.sum()

            hsma0 = hsma0.loc[(hsma0.short_r != -1) & (hsma0.long_r != -1),
                              ['code', 'date', 'short_r', 'long_r']]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata
        
    def hsmadata_fixp(self, day, p, lr):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['long_r'] = -1
            hsma0['short_r'] = -1
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]

            #做空
            for i in range(0, hsma0.shape[0] - day):
                #做空
                rlist = 1 - hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + day), 'open'] * (1 - p)) - fee
                #波动不足，不交易
                temp = 1 - hsma0.loc[(i + 1):(i + day), 'low'] / hsma0.loc[(
                    i + 1):(i + day), 'open']
                rlist[temp < p] = 0
                #开盘波动太大，不交易
                temp = 1 - hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                    i + 1):(i + day), 'preclose']
                rlist[temp > lr] = 0
                hsma0.loc[i, 'short_r'] = rlist.sum()
            #做多
            for i in range(0, hsma0.shape[0] - day):
                #做多
                rlist = hsma0.loc[(i + 1):(i + day), 'nextopen'] / (
                    hsma0.loc[(i + 1):(i + day), 'open'] * (1 + p)) - 1 - fee
                #止盈，收益上限
                temp = hsma0.loc[(i + 1):(i + day), 'high'] / hsma0.loc[(
                    i + 1):(i + day), 'open'] - 1
                rlist[temp < p] = 0
                #开盘波动太大，不交易
                temp = hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                    i + 1):(i + day), 'preclose'] - 1
                rlist[temp > lr] = 0
                hsma0.loc[i, 'long_r'] = rlist.sum()

            hsma0 = hsma0.loc[(hsma0.short_r != -1) & (hsma0.long_r != -1),
                              ['code', 'date', 'short_r', 'long_r']]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmadata_fixma(self, day, m, lr):  #双向交易，多空不同的p

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0['preSMA'] = talib.SMA(hsma0.preclose.values, timeperiod=m)
            hsma0['r'] = -1
            hsma0.dropna(inplace=True)
            hsma0.index = range(0, hsma0.shape[0])
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]

            for i in range(0, hsma0.shape[0] - day):
                if hsma0.preclose[i] < hsma0.preSMA[i]:  #做空
                    temp = hsma0.preclose[(i + 1):(i + day)] > hsma0.preSMA[(
                        i + 1):(i + day)]
                    if any(temp):
                        closeprice = hsma0.open[i + np.min(np.where(temp)) + 1]
                    else:
                        closeprice = hsma0.close[i + day - 1]
                    hsma0.loc[
                        i, 'r'] = 1 - hsma0.loc[i, 'open'] / closeprice - fee
                elif hsma0.preclose[i] > hsma0.preSMA[i]:  #做多
                    temp = hsma0.preclose[(i + 1):(i + day)] < hsma0.preSMA[(
                        i + 1):(i + day)]
                    if any(temp):
                        closeprice = hsma0.open[i + np.min(np.where(temp)) + 1]
                    else:
                        closeprice = hsma0.close[i + day - 1]
                    hsma0.loc[
                        i, 'r'] = closeprice / hsma0.loc[i, 'open'] - 1 - fee
                else:
                    hsma0.loc[i, 'r'] = 0
            hsma0 = hsma0.loc[(hsma0.r != -1), ['code', 'date', 'r']]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmadata_bestp2ma(self, day, lr):  #双向交易，多空不同的p  不能是ma

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['bestma'] = 0
            hsma0['bestp_long'] = -1
            hsma0['bestp_long_r'] = 0
            hsma0['bestp_short'] = -1
            hsma0['bestp_short_r'] = 0
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            plist = list(np.arange(0.002, 0.022, 0.002))
            #做空
            for i in range(0, hsma0.shape[0] - day):
                bestp_r = list()
                for p in plist:
                    #做空
                    rlist = 1 - hsma0.loc[(i + 1):(i + day), 'close'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] * (1 - p)) - fee
                    #波动不足，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'low'] / hsma0.loc[(i + 1):(i + day), 'open']
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = 1 - hsma0.loc[(i + 1):(
                        i + day), 'open'] / hsma0.loc[(i + 1):
                                                      (i + day), 'preclose']
                    rlist[temp > lr] = 0
                    r = rlist.sum()
                    bestp_r.append(r)
                hsma0.loc[i, 'bestp_short'] = plist[bestp_r.index(
                    max(bestp_r))]
                hsma0.loc[i, 'bestp_short_r'] = max(bestp_r)
            #做多
            for i in range(0, hsma0.shape[0] - day):
                bestp_r = list()
                for p in plist:
                    #做多
                    rlist = hsma0.loc[(i + 1):(i + day), 'close'] / (
                        hsma0.loc[(i + 1):(i + day), 'open'] *
                        (1 + p)) - 1 - fee
                    #止盈，收益上限
                    temp = hsma0.loc[(i + 1):(i + day), 'high'] / hsma0.loc[(
                        i + 1):(i + day), 'open'] - 1
                    rlist[temp < p] = 0
                    #开盘波动太大，不交易
                    temp = hsma0.loc[(i + 1):(i + day), 'open'] / hsma0.loc[(
                        i + 1):(i + day), 'preclose'] - 1
                    rlist[temp > lr] = 0
                    r = rlist.sum()
                    bestp_r.append(r)
                hsma0.loc[i, 'bestp_long'] = plist[bestp_r.index(max(bestp_r))]
                hsma0.loc[i, 'bestp_long_r'] = max(bestp_r)

            hsma0 = hsma0.loc[
                (hsma0.bestp_short != -1) & (hsma0.bestp_long != -1), [
                    'code', 'date', 'bestp_short', 'bestp_short_r',
                    'bestp_long', 'bestp_long_r'
                ]]
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        return hsmadata

    def hsmadata_y(self, day):  #code 只能是商品

        hsma0 = pd.read_csv("raw data\\stock\\" + self.code + ".csv")
        hsma0.columns = [
            'date', 'open', 'high', 'low', 'close', 'vol', 'openint'
        ]
        hsma0['code'] = self.code
        for i in range(0, hsma0.shape[0]):
            ymd = hsma0.loc[i, 'date'].split('/')
            hsma0.loc[i, 'date'] = int(ymd[0]) * 10000 + int(
                ymd[1]) * 100 + int(ymd[2])
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        hsma0 = hsma0[hsma0['date'] > 20080000]

        hsma0['open'] = hsma0['open'] + 0.0
        hsma0['high'] = hsma0['high'] + 0.0
        hsma0['low'] = hsma0['low'] + 0.0
        hsma0['close'] = hsma0['close'] + 0.0
        hsma0['vol'] = hsma0['vol'] + 0.0
        hsma0['openint'] = hsma0['openint'] + 0.0

        hsmadata = hsma0[[
            'date',
            'code',
        ]].copy()
        hsmadata['ratio'] = hsma0['close'].shift(-day) / hsma0['close'] - 1

        hsmadata = hsmadata.dropna()

        return hsmadata

    def hsmadata_y_var(self, vn):  #code 只能是商品

        hsma0 = pd.read_csv("raw data\\stock\\" + self.code + ".csv")
        hsma0.columns = [
            'date', 'open', 'high', 'low', 'close', 'vol', 'openint'
        ]
        hsma0['code'] = self.code
        for i in range(0, hsma0.shape[0]):
            ymd = hsma0.loc[i, 'date'].split('/')
            hsma0.loc[i, 'date'] = int(ymd[0]) * 10000 + int(
                ymd[1]) * 100 + int(ymd[2])
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        hsma0 = hsma0[hsma0['date'] > 20080000]

        hsma0['open'] = hsma0['open'] + 0.0
        hsma0['high'] = hsma0['high'] + 0.0
        hsma0['low'] = hsma0['low'] + 0.0
        hsma0['close'] = hsma0['close'] + 0.0
        hsma0['vol'] = hsma0['vol'] + 0.0
        hsma0['openint'] = hsma0['openint'] + 0.0

        hsmadata = hsma0[[
            'date',
            'code',
        ]].copy()
        hsmadata['ROC_1'] = talib.ROC(hsma0.close.values, timeperiod=1)
        hsmadata['ROC_1'] = hsmadata['ROC_1'].abs()
        hsmadata['var'] = talib.MA(hsmadata.ROC_1.values, timeperiod=vn)
        hsmadata['var'] = hsmadata['var'].shift(-vn)

        hsmadata = hsmadata.dropna()

        return hsmadata

    def hsmadata_daycode_lsr(self, hsma, day):
        hsma['r'] = 0
        hsma.loc[(hsma.prob_long > 0.5) & (hsma.prob_short < 0.5), 
                 'r'] = hsma.loc[(hsma.prob_long > 0.5) & (hsma.prob_short < 0.5), 
                    'ratio'] - 0.0004
        hsma.loc[(hsma.prob_long < 0.5) & (hsma.prob_short > 0.5), 
                 'r'] = -hsma.loc[(hsma.prob_long < 0.5) & (hsma.prob_short > 0.5), 
                    'ratio'] - 0.0004
        hsmaratio = pd.DataFrame(hsma.groupby(['date'])['r'].mean()) / day
        hsmaratio.columns = ['dayratio']
        hsmaratio['date'] = hsmaratio.index
        hsmaratio['ratio'] = hsmaratio['dayratio'].cumsum()
        
        return hsmaratio

    def hsmadata_predp_r(self, hsma, lr):  #

        hsmaall = pd.merge(self.hsmaall, hsma)
        hsmaratio = pd.DataFrame()
        portfolio = pd.DataFrame()
        for code in hsmaall.code.unique():
            hsma0 = hsmaall[hsmaall.code == code].copy()
            hsma0.sort_values(by='date', inplace=True)
            hsma0.index = range(0, hsma0.shape[0])
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsma0.predp = (hsma0.predp + 1) * self. minp
            
            #做空
            dayratio0 = 1 - hsma0['close'] / (hsma0['open'] *
                                              (1 - hsma0.predp)) - fee
            temp = 1 - hsma0.low / hsma0.open < hsma0.predp
            dayratio0[temp] = 0
            temp = 1 - hsma0.open / hsma0.preclose > lr
            dayratio0[temp] = 0
            #做多
            dayratio1 = hsma0['close'] / (hsma0['open'] *
                                          (1 + hsma0.predp)) - 1 - fee
            temp = hsma0.high / hsma0.open - 1 < hsma0.predp
            dayratio1[temp] = 0
            temp = hsma0.open / hsma0.preclose - 1 > lr
            dayratio1[temp] = 0

            hsma0['dayratio'] = dayratio0 + dayratio1

            hsmaratio = pd.concat([hsmaratio, hsma0], ignore_index=True)

            temp = hsma0[['date', 'dayratio']].copy()
            temp.columns = ['date', 'dayratio_' + code]
            if portfolio.shape[0] == 0:
                portfolio = temp
            else:
                portfolio = pd.merge(portfolio, temp, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return hsmaratio, portfolio
        
    def hsmadata_predp2ma_r(self, hsma, l, lr):  #

        hsmaall = pd.merge(self.hsmaall, hsma)
        hsmaratio = pd.DataFrame()
        portfolio = pd.DataFrame()
        for code in hsmaall.code.unique():
            hsma0 = hsmaall[hsmaall.code == code].copy()
            hsma0.sort_values(by='date', inplace=True)
            hsma0.index = range(0, hsma0.shape[0])
            hsma0['ma'] = talib.MA(hsma0.preclose.values, timeperiod=l)
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            #做空
            dayratio0 = 1 - hsma0['close'] / (hsma0['open'] *
                                              (1 - hsma0.predp_short)) - fee
            temp = 1 - hsma0.low / hsma0.open < hsma0.predp_short
            dayratio0[temp] = 0
            temp = 1 - hsma0.open / hsma0.preclose > lr
            dayratio0[temp] = 0
            temp = hsma0.preclose > hsma0.ma
            dayratio0[temp] = 0
            #做多
            dayratio1 = hsma0['close'] / (hsma0['open'] *
                                          (1 + hsma0.predp_long)) - 1 - fee
            temp = hsma0.high / hsma0.open - 1 < hsma0.predp_long
            dayratio1[temp] = 0
            temp = hsma0.open / hsma0.preclose - 1 > lr
            dayratio1[temp] = 0
            temp = hsma0.preclose < hsma0.ma
            dayratio0[temp] = 0

            hsma0['dayratio'] = dayratio0 + dayratio1

            hsmaratio = pd.concat([hsmaratio, hsma0], ignore_index=True)

            temp = hsma0[['date', 'dayratio']].copy()
            temp.columns = ['date', 'dayratio_' + code]
            if portfolio.shape[0] == 0:
                portfolio = temp
            else:
                portfolio = pd.merge(portfolio, temp, how='outer', sort=True)

        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return hsmaratio, portfolio

    def hsmadata_predp2_r(self, hsma, lr):  #

        hsmaall = pd.merge(self.hsmaall, hsma)
        hsmaratio = pd.DataFrame()
        portfolio = pd.DataFrame()
        for code in hsmaall.code.unique():
            hsma0 = hsmaall[hsmaall.code == code].copy()
            hsma0.sort_values(by='date', inplace=True)
            hsma0.index = range(0, hsma0.shape[0])
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma0.loc[
                    hsma0.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            #做空
            dayratio0 = 1 - hsma0['close'] / (hsma0['open'] *
                                              (1 - hsma0.predp_short)) - fee
            temp = 1 - hsma0.low / hsma0.open < hsma0.predp_short
            dayratio0[temp] = 0
            temp = 1 - hsma0.open / hsma0.preclose > lr
            dayratio0[temp] = 0
            #做多
            dayratio1 = hsma0['close'] / (hsma0['open'] *
                                          (1 + hsma0.predp_long)) - 1 - fee
            temp = hsma0.high / hsma0.open - 1 < hsma0.predp_long
            dayratio1[temp] = 0
            temp = hsma0.open / hsma0.preclose - 1 > lr
            dayratio1[temp] = 0

            hsma0['dayratio'] = dayratio0 + dayratio1

            hsmaratio = pd.concat([hsmaratio, hsma0], ignore_index=True)

            temp = hsma0[['date', 'dayratio']].copy()
            temp.columns = ['date', 'dayratio_' + code]
            if portfolio.shape[0] == 0:
                portfolio = temp
            else:
                portfolio = pd.merge(portfolio, temp, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return hsmaratio, portfolio

    def hsmadata_sma_r(self, hsma, ncode, length):  #

        hsmadata = pd.DataFrame()
        for code in self.feelist.keys():
            hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
            hsma0['SMA'] = talib.SMA(hsma0.preclose.values, timeperiod=length)
            hsma0 = hsma0.dropna()
            hsmadata = pd.concat([hsmadata, hsma0], ignore_index=True)

        portfolio_all = pd.DataFrame()
        portfolio = pd.DataFrame()
        for d in hsma['startdate'].unique():
            code_long = hsma[(hsma['startdate'] == d)
                             & (hsma['LS'] == 'L')].copy()
            code_short = hsma[(hsma['startdate'] == d)
                              & (hsma['LS'] == 'S')].copy()
            ###选取盈利潜力最大的ncode个品种训练并预测
            code_long.sort_values(by='prob', ascending=False, inplace=True)
            code_short.sort_values(by='prob', ascending=False, inplace=True)
            hsmad = pd.concat(
                [code_long.iloc[0:ncode, ], code_short.iloc[0:ncode, ]],
                ignore_index=True)
            portfolio_all = pd.concat(
                [portfolio_all, hsmad], ignore_index=True)

            portfoliod = pd.DataFrame()
            for i in hsmad.index:
                code = hsmad.loc[i, 'code']
                startdate = hsmad.loc[i, 'startdate']
                enddate = hsmad.loc[i, 'enddate']
                LS = hsmad.loc[i, 'LS']
                hsma0 = hsmadata[(hsmadata.code == code)
                                 & (hsmadata.date >= startdate) &
                                 (hsmadata.date <= enddate)].copy()
                if hsma0.shape[0] == 0:
                    continue
                if self.feelist[code] > 1:
                    fee = self.feelist[code] / self.handlist[code] / hsma0[
                        'close'].iloc[hsma0.shape[0] - 1]
                else:
                    fee = self.feelist[code]
                if LS == 'L':
                    if hsma0.preclose.iloc[0] > hsma0.SMA.iloc[0]:  #开仓
                        dayratio = hsma0.close / hsma0.preclose - 1
                        dayratio.iloc[
                            0] = hsma0.close.iloc[0] / hsma0.open.iloc[0] - 1
                        if any(hsma0.preclose < hsma0.SMA):  #平仓
                            idx = np.where(hsma0.preclose < hsma0.SMA)[0][0]
                            dayratio[idx:len(dayratio)] = 0
                        dayratio.iloc[0] -= fee
                        dayratio.iloc[len(dayratio) - 1] -= fee
                    else:
                        dayratio = np.repeat(0, hsma0.shape[0])
                if LS == 'S':
                    if hsma0.preclose.iloc[0] < hsma0.SMA.iloc[0]:
                        dayratio = 1 - hsma0.close / hsma0.preclose
                        dayratio.iloc[
                            0] = 1 - hsma0.close.iloc[0] / hsma0.open.iloc[0]
                        if any(hsma0.preclose > hsma0.SMA):  #平仓
                            idx = np.where(hsma0.preclose > hsma0.SMA)[0][0]
                            dayratio[idx:len(dayratio)] = 0
                        dayratio.iloc[0] -= fee
                        dayratio.iloc[len(dayratio) - 1] -= fee
                    else:
                        dayratio = np.repeat(0, hsma0.shape[0])
                temp = pd.DataFrame({
                    'date': hsma0.date,
                    'dayratio_' + code + '_' + LS: dayratio
                })
                if portfoliod.shape[0] == 0:
                    portfoliod = temp
                else:
                    portfoliod = pd.merge(
                        portfoliod, temp, how='outer', sort=True)

            if portfolio.shape[0] == 0:
                portfolio = portfoliod
            else:
                portfolio = pd.merge(
                    portfolio, portfoliod, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return portfolio

    def hsmadata_breakhl_fixlength_r(self, hsma, ncode, length, lr):  #

        portfolio = pd.DataFrame()
        for d in hsma['startdate'].unique():
            hsmad = hsma[hsma['startdate'] == d].copy()
            hsmad.sort_values(by='prob', ascending=False, inplace=True)
            hsmad = hsmad.iloc[0:ncode, ]

            portfoliod = pd.DataFrame()
            for i in hsmad.index:
                code = hsmad.loc[i, 'code']
                startdate = hsmad.loc[i, 'startdate']
                enddate = hsmad.loc[i, 'enddate']
                hsma0 = self.hsmaall[self.hsmaall.code == code].copy()
                hsma0['highd'] = talib.MAX(
                    hsma0.high.shift(1).values, timeperiod=length)
                hsma0['lowd'] = talib.MAX(
                    hsma0.low.shift(1).values, timeperiod=length)
                hsma0 = hsma0.dropna()
                hsma0 = hsma0[(hsma0.date >= startdate) & (hsma0.date <= enddate)]
                if hsma0.shape[0] == 0:
                    continue
                if self.feelist[code] > 1:
                    fee = self.feelist[code] / self.handlist[code] / hsma0[
                        'close'].iloc[hsma0.shape[0] - 1]
                else:
                    fee = self.feelist[code]
                
                #做空
                dayratio0 = 1 - hsma0['nextopen'] / (hsma0['lowd']) - fee
                #开盘价入场
                temp = hsma0['open'] < hsma0['lowd']
                dayratio0[temp] = 0#1 - hsma0.loc[temp, 'nextopen'] / hsma0.loc[temp, 'open'] - fee
                temp = hsma0.low > hsma0.lowd
                dayratio0[temp] = 0
                temp = 1 - hsma0.open / hsma0.preclose > lr
                dayratio0[temp] = 0
                #做多
                dayratio1 = hsma0['nextopen'] / (hsma0['highd']) - 1 - fee
                #开盘价入场
                temp = hsma0['open'] > hsma0['highd']
                dayratio1[temp] = 0#hsma0.loc[temp, 'nextopen'] / hsma0.loc[temp, 'open'] - 1 - fee
                temp = hsma0.high < hsma0.highd
                dayratio1[temp] = 0
                temp = hsma0.open / hsma0.preclose - 1 > lr
                dayratio1[temp] = 0

                dayratio = dayratio0 + dayratio1

                temp = pd.DataFrame({
                    'date': hsma0.date,
                    'dayratio_' + code: dayratio
                })
                if portfoliod.shape[0] == 0:
                    portfoliod = temp
                else:
                    portfoliod = pd.merge(
                        portfoliod, temp, how='outer', sort=True)

            if portfolio.shape[0] == 0:
                portfolio = portfoliod
            else:
                portfolio = pd.merge(
                    portfolio, portfoliod, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()
            
        return portfolio

    def hsmadata_fixminpoint_r(self, hsma, ncode, mp, lr):  #

        portfolio = pd.DataFrame()
        for d in hsma['startdate'].unique():
            hsmad = hsma[hsma['startdate'] == d].copy()
            hsmad.sort_values(by='prob', ascending=False, inplace=True)
            hsmad = hsmad.iloc[0:ncode, ]

            portfoliod = pd.DataFrame()
            for i in hsmad.index:
                code = hsmad.loc[i, 'code']
                startdate = hsmad.loc[i, 'startdate']
                enddate = hsmad.loc[i, 'enddate']
                hsma0 = self.hsmaall[(self.hsmaall.code == code)].copy()
                hsma0 = hsma0[(hsma0.date >= startdate) & (hsma0.date <= enddate)].copy()
                if hsma0.shape[0] == 0:
                    continue
                if self.feelist[code] > 1:
                    fee = self.feelist[code] / self.handlist[code] / hsma0[
                        'close'].iloc[hsma0.shape[0] - 1]
                else:
                    fee = self.feelist[code]
                #做空
                dayratio0 = 1 - hsma0['nextopen'] / (hsma0['open'] -  mp * self.minpoint[code]) - fee
                temp = hsma0['low'] > (hsma0['open'] -  mp * self.minpoint[code])
                dayratio0[temp] = 0
                temp = 1 - hsma0.open / hsma0.preclose > lr
                dayratio0[temp] = 0
                #做多
                dayratio1 = hsma0['nextopen'] / (hsma0['open'] +  mp * self.minpoint[code]) - 1 - fee
                temp = hsma0['high'] < (hsma0['open'] +  mp * self.minpoint[code])
                dayratio1[temp] = 0
                temp = hsma0.open / hsma0.preclose - 1 > lr
                dayratio1[temp] = 0

                dayratio = dayratio0 + dayratio1

                temp = pd.DataFrame({
                    'date': hsma0.date,
                    'dayratio_' + code: dayratio
                })
                if portfoliod.shape[0] == 0:
                    portfoliod = temp
                else:
                    portfoliod = pd.merge(
                        portfoliod, temp, how='outer', sort=True)

            if portfolio.shape[0] == 0:
                portfolio = portfoliod
            else:
                portfolio = pd.merge(
                    portfolio, portfoliod, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return portfolio
        
    def hsmadata_fixvar_r(self, hsma, day, ncode, v, lr):  #

        portfolio = pd.DataFrame()
        for d in hsma['startdate'].unique():
            hsmad = hsma[hsma['startdate'] == d].copy()
            hsmad.sort_values(by='prob', ascending=False, inplace=True)
            hsmad = hsmad.iloc[0:ncode, ]

            portfoliod = pd.DataFrame()
            for i in hsmad.index:
                code = hsmad.loc[i, 'code']
                startdate = hsmad.loc[i, 'startdate']
                enddate = hsmad.loc[i, 'enddate']
                hsma0 = self.hsmaall[(self.hsmaall.code == code)].copy()
                hsma0['std'] = talib.STDDEV(hsma0.close.values, timeperiod=day, nbdev=1)
                hsma0['std'] = hsma0['std'].shift(1)
                hsma0.dropna(inplace=True)
                hsma0 = hsma0[(hsma0.date >= startdate) & (hsma0.date <= enddate)].copy()
                if hsma0.shape[0] == 0:
                    continue
                if self.feelist[code] > 1:
                    fee = self.feelist[code] / self.handlist[code] / hsma0[
                        'close'].iloc[hsma0.shape[0] - 1]
                else:
                    fee = self.feelist[code]
                #做空
                dayratio0 = 1 - hsma0['nextopen'] / (hsma0['open'] -  v * hsma0['std']) - fee
                temp = hsma0['low'] > (hsma0['open'] -  v * hsma0['std'])
                dayratio0[temp] = 0
                temp = 1 - hsma0.open / hsma0.preclose > lr
                dayratio0[temp] = 0
                #做多
                dayratio1 = hsma0['nextopen'] / (hsma0['open'] +  v * hsma0['std']) - 1 - fee
                temp = hsma0['high'] < (hsma0['open'] +  v * hsma0['std'])
                dayratio1[temp] = 0
                temp = hsma0.open / hsma0.preclose - 1 > lr
                dayratio1[temp] = 0

                dayratio = dayratio0 + dayratio1

                temp = pd.DataFrame({
                    'date': hsma0.date,
                    'dayratio_' + code: dayratio
                })
                if portfoliod.shape[0] == 0:
                    portfoliod = temp
                else:
                    portfoliod = pd.merge(
                        portfoliod, temp, how='outer', sort=True)

            if portfolio.shape[0] == 0:
                portfolio = portfoliod
            else:
                portfolio = pd.merge(
                    portfolio, portfoliod, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return portfolio
        
    def hsmadata_fixp_r(self, hsma, ncode, p, lr):  #

        portfolio = pd.DataFrame()
        for d in hsma['startdate'].unique():
            hsmad = hsma[hsma['startdate'] == d].copy()
            hsmad.sort_values(by='prob', ascending=False, inplace=True)
            hsmad = hsmad.iloc[0:ncode, ]

            portfoliod = pd.DataFrame()
            for i in hsmad.index:
                code = hsmad.loc[i, 'code']
                startdate = hsmad.loc[i, 'startdate']
                enddate = hsmad.loc[i, 'enddate']
                hsma0 = self.hsmaall[(self.hsmaall.code == code)
                                     & (self.hsmaall.date >= startdate) &
                                     (self.hsmaall.date <= enddate)].copy()
                if hsma0.shape[0] == 0:
                    continue
                if self.feelist[code] > 1:
                    fee = self.feelist[code] / self.handlist[code] / hsma0[
                        'close'].iloc[hsma0.shape[0] - 1]
                else:
                    fee = self.feelist[code]
                #做空
                dayratio0 = 1 - hsma0['nextopen'] / (hsma0['open'] *
                                                     (1 - p)) - fee
                temp = 1 - hsma0.low / hsma0.open < p
                dayratio0[temp] = 0
                temp = 1 - hsma0.open / hsma0.preclose > lr
                dayratio0[temp] = 0
                #做多
                dayratio1 = hsma0['nextopen'] / (hsma0['open'] *
                                                 (1 + p)) - 1 - fee
                temp = hsma0.high / hsma0.open - 1 < p
                dayratio1[temp] = 0
                temp = hsma0.open / hsma0.preclose - 1 > lr
                dayratio1[temp] = 0

                dayratio = dayratio0 + dayratio1

                temp = pd.DataFrame({
                    'date': hsma0.date,
                    'dayratio_' + code: dayratio
                })
                if portfoliod.shape[0] == 0:
                    portfoliod = temp
                else:
                    portfoliod = pd.merge(
                        portfoliod, temp, how='outer', sort=True)

            if portfolio.shape[0] == 0:
                portfolio = portfoliod
            else:
                portfolio = pd.merge(
                    portfolio, portfoliod, how='outer', sort=True)

        portfolio.sort_values(by='date', inplace=True)
        portfolio['dayratio'] = portfolio.iloc[:, 1:portfolio.shape[1]].mean(
            axis=1)
        portfolio['ratio'] = portfolio['dayratio'].cumsum()

        return portfolio

    def tradestat_portfolio(self, portfolio):
        ###对收益曲线的统计
        tradestat = pd.DataFrame({
            'startdate': [min(portfolio['date'])],
            'enddate': [max(portfolio['date'])]
        })
        tradestat['ratio'] = portfolio.loc[portfolio.shape[0] - 1, 'ratio']

        tradestat['meandayratio'] = portfolio['dayratio'].mean()

        mdd = 0
        mdddate = 0
        portfolio['year'] = 0
        for i in portfolio.index:
            portfolio.loc[i, 'year'] = int(str(portfolio.loc[i, 'date'])[0:4])
            mdd1 = portfolio.loc[i, 'ratio'] - min(portfolio.loc[i:, 'ratio'])
            if mdd1 > mdd:
                mdd = mdd1
                mdddate = portfolio.loc[i, 'date']

        for year in range(2008, 2018):
            temp = portfolio[portfolio['year'] == year]
            if temp.shape[0] == 0:
                continue
            temp.index = range(0, temp.shape[0])
            tradestat[str(year) + 'ratio'] = sum(temp['dayratio'])

        tradestat['yearratio'] = tradestat['ratio'] / portfolio.shape[0] * 252
        tradestat['mdd'] = mdd
        tradestat['mdddate'] = mdddate
        tradestat['RRR'] = tradestat['yearratio'] / tradestat['mdd']

        tradestat['sharpratio'] = portfolio['dayratio'].mean() / portfolio[
            'dayratio'].std() * 252**0.5

        print(tradestat)

        return tradestat

    def conditionrank(self, traindata, col, tn, ascending):
        dates = pd.Series(traindata['date'].unique()).sort_values()

        traindatac = pd.DataFrame()
        for d in dates:
            temp = traindata[traindata['date'] == d].copy()
            temp['rank'] = temp[col].rank(ascending=ascending)
            traindatac = pd.concat(
                [traindatac, temp[temp['rank'] <= tn]], ignore_index=True)

        return (traindatac)

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
            hsmatrade = pd.concat(
                [hsmatrade, hsmad[hsmad['rank'] <= n]], ignore_index=True)

        if not self.absratio:
            hsmatrade = pd.merge(hsmatrade, self.ratio_mean)
            hsmatrade.ratio = hsmatrade.ratio + hsmatrade.ratio_mean

        hsmatrade['ratio'] = hsmatrade['ratio'] - self.fee

        hsmatrade.ix[hsmatrade['predratio'] < cr, 'ratio'] = 0

        print(hsmatrade['ratio'].describe())

        return (hsmatrade)

    def hsmatraderegressor(self, hsma, day):

        hsmatrade = hsma[['date', 'code', 'ratio', 'predratio']].copy()

        hsmatrade['dayratio'] = hsma['ratio']

        hsmatrade.ix[
            hsmatrade['predratio'] < 0,
            'dayratio'] = -hsmatrade.ix[hsmatrade['predratio'] < 0, 'ratio']

        hsmatrade['dayratio'] = hsmatrade['dayratio'] - self.fee

        hsmatrade['dayratio'] = hsmatrade['dayratio'] / day

        hsmatrade['cumratio'] = hsmatrade['dayratio'].cumsum()

        print(hsmatrade['dayratio'].describe())

        return (hsmatrade)

    def hsmatraderegressor_r(self, hsma, day, r):

        hsmatrade = hsma[['date', 'code', 'ratio', 'predratio']].copy()

        hsmatrade['dayratio'] = hsma['ratio']

        hsmatrade.ix[
            hsmatrade['predratio'] < 0,
            'dayratio'] = -hsmatrade.ix[hsmatrade['predratio'] < 0, 'ratio']

        hsmatrade['dayratio'] = hsmatrade['dayratio'] - self.fee

        hsmatrade['dayratio'] = hsmatrade['dayratio'] / day

        hsmatrade.ix[(hsmatrade['predratio'] < r) & (hsmatrade['predratio'] >
                                                     -r), 'dayratio'] = 0

        hsmatrade['cumratio'] = hsmatrade['dayratio'].cumsum()

        print(hsmatrade['dayratio'].describe())

        return (hsmatrade)

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

        return (hsmatradec)

    def hsmatradedayclassifier(self, condition, hsma, minn):

        hsmatradec = self.hsmatradeclassifier(condition, hsma)

        hsmatradeday = hsmatradec.groupby(
            ['date'], as_index=False)[['ratio']].mean()
        number = hsmatradec.groupby(['date'], as_index=False)[['ratio']].size()
        number.index = range(len(number))
        hsmatradeday['number'] = number
        hsmatradeday['tradenumber'] = hsmatradeday['number']
        hsmatradeday.ix[hsmatradeday.number > minn, 'tradenumber'] = minn
        hsmatradeday['dayratio'] = hsmatradeday['ratio'] / self.day * (
            hsmatradeday['tradenumber'] / minn)
        hsmatradeday['cumratio'] = hsmatradeday['dayratio'].cumsum()

        print(hsmatradeday['dayratio'].describe())
        plt.plot(hsmatradeday['cumratio'])

        return (hsmatradeday)

    def hsmaregressor(self, hsma0):

        hsmai = hsma0.copy()

        hsmai['dayratio'] = hsmai['ratio'] / self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()

        return (hsmai)

    def hsmaindexregressor_short(self, hsmaindex):

        hsmai = hsmaindex.copy()

        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio < 0,
                 'dayratio'] = -hsmai.ix[hsmai.predratio < 0, 'dayratio']
        hsmai.ix[hsmai.predratio >= 0, 'dayratio'] = 0
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee) / self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()

        return (hsmai)

    def hsmaindexclassifier(self, hsmaindex):

        hsmai = hsmaindex.copy()
        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio == 0,
                 'dayratio'] = -hsmai.ix[hsmai.predratio == 0, 'dayratio']
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee) / self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()

        return (hsmai)

    def hsmaindexclassifier_short(self, hsmaindex):

        hsmai = hsmaindex.copy()
        hsmai['dayratio'] = hsmai['ratio']
        hsmai.ix[hsmai.predratio == 0,
                 'dayratio'] = -hsmai.ix[hsmai.predratio == 0, 'dayratio']
        hsmai.ix[hsmai.predratio == 1, 'dayratio'] = 0
        hsmai['dayratio'] = (hsmai['dayratio'] - self.indexfee) / self.day
        hsmai['cumratio'] = hsmai['dayratio'].cumsum()

        return (hsmai)

    def traintestanalysisclassifier(self, hsma, testlen, ntrain):

        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        tttable = pd.DataFrame()
        for i in range(ntrain, ntest):
            temp = hsma[(hsma['date'] >= dates[i * testlen])
                        & (hsma['date'] < dates[(i + 1) * testlen])].copy()
            temp1 = temp[temp.predratio == 1]

            table = pd.DataFrame(
                {
                    'i': i,
                    'startdate': [dates[i * testlen]],
                    'enddate': [dates[(i + 1) * testlen]]
                },
                columns=['i', 'startdate', 'enddate'])
            table['meanratio_all'] = temp.ratio.mean()
            table['meanratio_pred'] = temp1.ratio.mean()
            table['number_pred'] = temp1.shape[0]

            tttable = pd.concat([tttable, table], ignore_index=True)

        return (tttable)

    def tradestatlist(self, hsmatradeday):  #n=200,100,50

        tradestatlist = pd.DataFrame()

        tradestat = self.tradestat(hsmatradeday)
        tradestat['Hedge'] = 'NoHedge'
        tradestatlist = pd.concat(
            [tradestatlist, tradestat], ignore_index=True)

        temp = hsmatradeday[['date', 'hedge300dayratio', 'hedge300ratio']]
        temp = temp.rename(columns={
            'hedge300dayratio': 'dayratio',
            'hedge300ratio': 'cumratio'
        })
        tradestat = self.tradestat(temp)
        tradestat['Hedge'] = 'Hedge300'
        tradestatlist = pd.concat(
            [tradestatlist, tradestat], ignore_index=True)

        temp = hsmatradeday[['date', 'hedge500dayratio', 'hedge500ratio']]
        temp = temp.rename(columns={
            'hedge500dayratio': 'dayratio',
            'hedge500ratio': 'cumratio'
        })
        tradestat = self.tradestat(temp)
        tradestat['Hedge'] = 'Hedge500'
        tradestatlist = pd.concat(
            [tradestatlist, tradestat], ignore_index=True)

        if any(hsmatradeday.columns == 'hedgecta1ratio'):
            temp = hsmatradeday[[
                'date', 'hedgecta1dayratio', 'hedgecta1ratio'
            ]]
            temp = temp.rename(columns={
                'hedgecta1dayratio': 'dayratio',
                'hedgecta1ratio': 'cumratio'
            })
            tradestat = self.tradestat(temp)
            tradestat['Hedge'] = 'HedgeCTA1'
            tradestatlist = pd.concat(
                [tradestatlist, tradestat], ignore_index=True)

        print(tradestatlist)
        tradestatlist.to_csv(
            "Test\\testresult\\" + self.label + ".csv", index=False)
