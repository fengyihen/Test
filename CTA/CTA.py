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
import talib



class CTA(FutureMinute.FutureMinute):

    def tpr_dayin(self, hsma, tpr, wtime, fee):
        
        hsmatradecode = pd.DataFrame()
        for d in hsma['date'].unique():
            hsmad = hsma[hsma['date'] == d]  
            
            #开仓
            idx = np.where(hsmad['LS'] != 'N')[0]
            idx = idx[(idx >= wtime) & (idx < hsmad.shape[0] - 1)]
            if len(idx) == 0:
                continue
            inpos = idx[0]                 
            
            ls = hsmad['LS'].iloc[inpos]
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
                traderatio = closeprice / openprice - 1 - fee
            else:
                traderatio = 1 - closeprice / openprice - fee
                
            temp = pd.DataFrame({'date' : d, 'LS': ls,
                                 'opentime' : opentime,
                                 'closetime' : closetime,
                                 'openprice' : openprice,
                                 'closeprice' : closeprice,
                                 'traderatio' : traderatio
                                 },index=[0])
            
            hsmatradecode = pd.concat([hsmatradecode, temp])
            
        return hsmatradecode

    def LS_turn(self, hsma, fee):
        
        hsmatradecode = pd.DataFrame()
        LSarray = np.where(hsma.LS != 'N')[0]
        if len(LSarray) == 0:
            return hsmatradecode
            
        i = LSarray[0]
        while True:            
            
            ls = hsma['LS'].iloc[i]
            opendate = hsma['date'].iloc[i]
            opentime = hsma['time'].iloc[i]
            openprice = hsma['open'].iloc[i]
            
            jarray = np.where((hsma['LS'].iloc[i:] != ls) & (hsma['LS'].iloc[i:] != 'N'))[0]
            if len(jarray) == 0:
                j = hsma.shape[0] - 1
            else:
                j = i + jarray[0]

            closedate = hsma['date'].iloc[j]
            closetime = hsma['time'].iloc[j]
            closeprice = hsma['open'].iloc[j]
            
            if ls == 'L':
                traderatio = closeprice / openprice - 1 - fee
            else:
                traderatio = 1 - closeprice / openprice - fee
                
            temp = pd.DataFrame({'LS': ls,
                                 'opendate' : opendate,
                                 'opentime' : opentime,
                                 'openprice' : openprice,
                                 'closedate' : closedate,
                                 'closetime' : closetime,
                                 'closeprice' : closeprice,
                                 'traderatio' : traderatio,
                                 'tradetime' : j - i,
                                 },index=[0])
            
            hsmatradecode = pd.concat([hsmatradecode, temp])
                       
            if j == hsma.shape[0] - 1:
                break
            
            i = j
            
        return hsmatradecode

    def LS_N(self, hsma, fee):
        
        hsmatradecode = pd.DataFrame()
        LSarray = np.where((hsma.LS == 'L') | (hsma.LS == 'S'))[0]
        if len(LSarray) == 0:
            return hsmatradecode
            
        i = LSarray[0]
        while True:            
            
            ls = hsma['LS'].iloc[i]
            opendate = hsma['date'].iloc[i]
            opentime = hsma['time'].iloc[i]
            openprice = hsma['open'].iloc[i]
            
            if ls == 'L':
                jarray = np.where((hsma['LSexit'].iloc[i:] == 'LN'))[0]
                if len(jarray) == 0:
                    j = hsma.shape[0] - 1
                else:
                    j = i + jarray[0]
            if ls == 'S':
                jarray = np.where((hsma['LSexit'].iloc[i:] == 'SN'))[0]
                if len(jarray) == 0:
                    j = hsma.shape[0] - 1
                else:
                    j = i + jarray[0]

            closedate = hsma['date'].iloc[j]
            closetime = hsma['time'].iloc[j]
            closeprice = hsma['open'].iloc[j]
            
            if ls == 'L':
                traderatio = closeprice / openprice - 1 - fee
            else:
                traderatio = 1 - closeprice / openprice - fee
                
            temp = pd.DataFrame({'LS': ls,
                                 'opendate' : opendate,
                                 'opentime' : opentime,
                                 'openprice' : openprice,
                                 'closedate' : closedate,
                                 'closetime' : closetime,
                                 'closeprice' : closeprice,
                                 'traderatio' : traderatio,
                                 'tradetime' : j - i,
                                 },index=[0])
            
            hsmatradecode = pd.concat([hsmatradecode, temp])
            
            if any(LSarray > j):
                i = LSarray[LSarray > j][0]
            else:
                break
            
        return hsmatradecode
        
    def SMA_tpr_dayin(self, length, lr, tpr, wtime):
            
        hsmatrade = pd.DataFrame()
        for code in self.hsmaall['code'].unique():   
            hsma = self.hsmaall[self.hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['MA1'] = talib.SMA(hsma.close.values, timeperiod=length)
            hsma['MA2'] = talib.SMA(hsma.close.values, timeperiod=length*lr)
            hsma['LS'] = 'N'
            temp = (hsma.MA1.shift(1) > hsma.MA2.shift(1)) & (hsma.close.shift(1) > hsma.open.shift(1))
            temp = temp & (hsma.MA1.shift(2) < hsma.MA2.shift(2))
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.MA1.shift(1) < hsma.MA2.shift(1)) & (hsma.close.shift(1) < hsma.open.shift(1))
            temp = temp & (hsma.MA1.shift(2) > hsma.MA2.shift(2))
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.tpr_dayin(hsma, tpr, wtime, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        print(hsmatrade.traderatio.mean())
        return hsmatrade

    def SMA_ls(self, hsmaall, length, lr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['MA1'] = talib.SMA(hsma.close.values, timeperiod=length)
            hsma['MA2'] = talib.SMA(hsma.close.values, timeperiod=length*lr)

            #入场
            hsma['LS'] = 'N'
            temp = (hsma.MA1.shift(1) > hsma.MA2.shift(1)) & (hsma.MA1.shift(2) < hsma.MA2.shift(2)) & (
                    hsma.MA2.shift(1) > hsma.MA2.shift(2))
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.MA1.shift(1) < hsma.MA2.shift(1)) & (hsma.MA1.shift(2) > hsma.MA2.shift(2)) & (
                    hsma.MA2.shift(1) < hsma.MA2.shift(2))
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def RSI_tpr_dayin(self, hsmaall, length, rr, tpr, wtime):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['RSI'] = talib.RSI(hsma.close.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = (hsma.RSI.shift(1) > rr) & (hsma.RSI.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.RSI.shift(1) < 100-rr) & (hsma.RSI.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.tpr_dayin(hsma, tpr, wtime, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade

    def BBANDS_ls(self, hsmaall, length, nstd):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            temp = talib.BBANDS(hsma.close.values, timeperiod=length, nbdevup=nstd, nbdevdn=nstd)
            hsma['upper'] = temp[0]
            hsma['MA'] = temp[1]
            hsma['lower'] = temp[2]

            hsma['LS'] = 'N'
            temp = (hsma.close.shift(1) > hsma.upper.shift(1)) & (hsma.MA.shift(1) > hsma.MA.shift(2))
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.close.shift(1) < hsma.lower.shift(1)) & (hsma.MA.shift(1) < hsma.MA.shift(2))
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def hlbreak_ls(self, hsmaall, length, nstd):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['upper'] = talib.MAX(hsma.high.values, timeperiod=length)
            hsma['lower'] = talib.MIN(hsma.low.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = hsma.close.shift(1) > hsma.upper.shift(2)
            hsma.loc[temp, 'LS'] = 'L'
            temp = hsma.close.shift(1) < hsma.lower.shift(2)
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def ADX_ls(self, hsmaall, length, rr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['ADX'] = talib.ADX(hsma.high.values, 
                                    hsma.low.values,
                                    hsma.close.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = (hsma.ADX.shift(1) > rr) & (hsma.ADX.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.ADX.shift(1) < 100-rr) & (hsma.ADX.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def ADX_N(self, hsmaall, length, rr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['ADX'] = talib.ADX(hsma.high.values, 
                                    hsma.low.values,
                                    hsma.close.values, timeperiod=length)

            #入场
            hsma['LS'] = 'N'
            temp = (hsma.ADX.shift(1) > rr) & (hsma.ADX.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.ADX.shift(1) < 100-rr) & (hsma.ADX.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'S'
            
            #出场
            hsma['LSexit'] = 'N'
            temp = hsma.ADX.shift(1) < 50
            hsma.loc[temp, 'LSexit'] = 'LN'
            temp = hsma.ADX.shift(1) > 50
            hsma.loc[temp, 'LSexit'] = 'SN'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_N(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def RSI_ls(self, hsmaall, length, rr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['RSI'] = talib.RSI(hsma.close.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = (hsma.RSI.shift(1) > rr) & (hsma.RSI.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.RSI.shift(1) < 100-rr) & (hsma.RSI.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def RSI_reverse_ls(self, hsmaall, length, rr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['RSI'] = talib.RSI(hsma.close.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = (hsma.RSI.shift(1) > rr) & (hsma.RSI.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'S'
            temp = (hsma.RSI.shift(1) < 100-rr) & (hsma.RSI.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'L'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_turn(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade

    def RSI_N(self, hsmaall, length, rr):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            print(code)
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['RSI'] = talib.RSI(hsma.close.values, timeperiod=length)

            #入场
            hsma['LS'] = 'N'
            temp = (hsma.RSI.shift(1) > rr) & (hsma.RSI.shift(2) < rr)
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.RSI.shift(1) < 100-rr) & (hsma.RSI.shift(2) > 100-rr)
            hsma.loc[temp, 'LS'] = 'S'
            
            #出场
            hsma['LSexit'] = 'N'
            temp = (hsma.RSI.shift(1) < 50) & (hsma.RSI.shift(2) > 50)
            hsma.loc[temp, 'LSexit'] = 'LN'
            temp = (hsma.RSI.shift(1) > 50) & (hsma.RSI.shift(2) < 50)
            hsma.loc[temp, 'LSexit'] = 'SN'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.LS_N(hsma, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def RSI_MA_tpr_dayin(self, hsmaall, length, rr, tpr, wtime):
            
        hsmatrade = pd.DataFrame()
        for code in hsmaall['code'].unique():   
            hsma = hsmaall[hsmaall['code'] == code].copy()
            hsma.index = range(hsma.shape[0])
            hsma['RSI'] = talib.RSI(hsma.close.values, timeperiod=length)
            hsma['MA'] = talib.SMA(hsma.close.values, timeperiod=length)

            hsma['LS'] = 'N'
            temp = (hsma.RSI.shift(1) > rr) & (hsma.RSI.shift(2) < rr)
            temp = temp & (hsma.close.shift(1) > hsma.MA.shift(1))
            hsma.loc[temp, 'LS'] = 'L'
            temp = (hsma.RSI.shift(1) < 100-rr) & (hsma.RSI.shift(2) > 100-rr)
            temp = temp & (hsma.close.shift(1) < hsma.MA.shift(1))
            hsma.loc[temp, 'LS'] = 'S'
            
            if self.feelist[code] > 1:
                fee = self.feelist[code] / self.handlist[code] / hsma.loc[
                    hsma.shape[0] - 1, 'close']
            else:
                fee = self.feelist[code]
            hsmatradecode = self.tpr_dayin(hsma, tpr, wtime, fee)
            hsmatradecode['code'] = code

            hsmatrade = pd.concat([hsmatrade, hsmatradecode])
        
        return hsmatrade
        
    def insample_test(self, strategy, *tupleArg):
        #样本内测试    
        
        hsmaall = self.hsmaall
        if strategy == 'RSI_tpr_dayin':
            hsmatrade = self.RSI_tpr_dayin(hsmaall, *tupleArg)
        elif strategy == 'RSI_MA_tpr_dayin':
            hsmatrade = self.RSI_MA_tpr_dayin(hsmaall, *tupleArg)
        elif strategy == 'RSI_ls':
            hsmatrade = self.RSI_ls(hsmaall, *tupleArg)
        elif strategy == 'RSI_reverse_ls':
            hsmatrade = self.RSI_reverse_ls(hsmaall, *tupleArg)
        elif strategy == 'RSI_N':
            hsmatrade = self.RSI_N(hsmaall, *tupleArg)            
        elif strategy == 'BBANDS_ls':
            hsmatrade = self.BBANDS_ls(hsmaall, *tupleArg)
        else:
            pass
        
        print('Mean traderatio : {}'.format(hsmatrade.traderatio.mean()))
        print('Number of Trades : {}'.format(hsmatrade.shape[0]))
          
        return hsmatrade    
        
    def insample_bestparam(self, strategy, *tupleArgs):
        
        pass
        
    def out_of_sample_month(self, strategy, ncode, ntrain, *tupleArg):

        dates = pd.Series(self.hsmaall['date'].unique()).sort_values()
        months = (dates // 100).unique()

        hsmatrade = pd.DataFrame()
        for i in range(ntrain, len(months) - 2):
            traindata = self.hsmaall[
                (self.hsmaall['date'] >= months[i - ntrain] * 100)
                & (self.hsmaall['date'] < months[i] * 100)].copy()
            testdata = self.hsmaall[(self.hsmaall['date'] >= months[i] * 100)
                        & (self.hsmaall['date'] < months[i + 1] * 100)].copy()

            print(testdata.date.max())

            #对近期历史数据的统计
            if strategy == 'RSI_tpr_dayin':
                hsmatradet = self.RSI_tpr_dayin(traindata, *tupleArg)
            elif strategy == 'RSI_MA_tpr_dayin':
                hsmatradet = self.RSI_MA_tpr_dayin(traindata, *tupleArg)
            elif strategy == 'RSI_ls':
                hsmatradet = self.RSI_ls(traindata, *tupleArg)
            elif strategy == 'RSI_reverse_ls':
                hsmatradet = self.RSI_reverse_ls(traindata, *tupleArg)
            elif strategy == 'RSI_N':
                hsmatradet = self.RSI_N(traindata, *tupleArg)
            else:
                pass

            if hsmatradet.shape[0] == 0:
                continue            
            portfoliot = hsmatradet.groupby('code')[['traderatio']].sum()
            portfoliot = portfoliot.sort_values(by='traderatio', ascending=False)
            portfoliot = portfoliot[portfoliot.traderatio > 0]
            if portfoliot.shape[0] == 0:
                continue
            codelist = pd.DataFrame({'code':portfoliot.index[0:ncode]})
            
            #对新一期数据的测试
            hsmaall = pd.merge(testdata, codelist)
            if strategy == 'RSI_tpr_dayin':
                hsmatradem = self.RSI_tpr_dayin(hsmaall, *tupleArg)
            elif strategy == 'RSI_MA_tpr_dayin':
                hsmatradem = self.RSI_MA_tpr_dayin(hsmaall, *tupleArg)
            elif strategy == 'RSI_ls':
                hsmatradem = self.RSI_ls(hsmaall, *tupleArg)
            elif strategy == 'RSI_reverse_ls':
                hsmatradem = self.RSI_reverse_ls(hsmaall, *tupleArg)
            elif strategy == 'RSI_N':
                hsmatradem = self.RSI_N(hsmaall, *tupleArg)
            else:
                pass
            
            hsmatrade = pd.concat([hsmatrade, hsmatradem])  
            if hsmatradem.shape[0] > 0:
                print(hsmatrade.traderatio.mean())
                print(hsmatrade.shape[0])
            
        return hsmatrade    
