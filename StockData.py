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
from WindPy import *
from datetime import *

           
class WindData():
    
    #从wind中导入数据
    def __init__(self, startdate, enddate):
        self.startdate = startdate  
        self.enddate = enddate

    def marketdata_history(self):
        
        w.start()
        marketdata = pd.read_csv("raw data\\stock\\marketstk.csv")
        codelist = pd.Series(marketdata['code'].unique()).sort_values()
 
        hsmadata = pd.DataFrame()
        for code in codelist:     
            print(code)
            hsma0 = marketdata[marketdata['code'] == code].copy()  
            if hsma0.date.max() >= self.enddate:
                continue
            date0 = str(hsma0.date.max() + 1)
            date1 = str(self.enddate)

            temp = w.wsd(code, "open,high,low,close,volume,amt,free_turn,last_trade_day", date0, date1, "PriceAdj=B")
            data = temp.Data
            datelist = []
            for i in range(len(data[7])):
                datelist.append(data[7][i].year*10000 + data[7][i].month*100 + data[7][i].day)
            
            hsma1 = pd.DataFrame({'date':datelist, 'open':data[0], 'high':data[1], 'low':data[2], 'close':data[3], 'vol':data[4], 'amt':data[5], 'free_turn':data[6]})
            hsma1['code'] = code
            
            hsma2 = pd.concat([hsma0, hsma1], ignore_index = True)
            hsma2 = hsma2[hsma0.columns]
            
            hsmadata = pd.concat([hsmadata, hsma2], ignore_index = True)
            
        hsmadata.to_csv("raw data\\stock\\marketstk.csv",index=False)
        
    def indexday(self):
        
        w.start()

        for code in ['000300.SH', '000905.SH']:
            hsma0 = pd.read_csv("raw data\\stock\\indexday_" + code + ".csv")
            if hsma0.date.max() >= self.enddate:
                continue
            date0 = str(hsma0.date.max() + 1)
            date1 = str(self.enddate)        
            temp = w.wsd(code, "open,high,low,close,volume,amt,last_trade_day", date0, date1, "")
            data = temp.Data

            datelist = []
            for i in range(len(data[6])):
                datelist.append(data[6][i].year*10000 + data[6][i].month*100 + data[6][i].day)
            
            hsma1 = pd.DataFrame({'date':datelist, 'open':data[0], 'high':data[1], 'low':data[2], 'close':data[3], 'vol':data[4], 'amt':data[5]})
            hsma1['code'] = code
            
            hsma2 = pd.concat([hsma0, hsma1], ignore_index = True)
            hsma2 = hsma2[hsma0.columns]
            hsma2.to_csv("raw data\\stock\\indexday_" + code + ".csv",index=False)
    
    def indexminutes(self, minutes):
        
        w.start()

        for code in ['000300.SH', '000905.SH']:
            hsma0 = pd.read_csv("raw data\\stock\\indexday_" + code + ".csv")
            if hsma0.date.max() >= self.enddate:
                continue
            date0 = str(hsma0.date.max() + 1)
            date1 = str(self.enddate)        
            temp = w.wsd(code, "open,high,low,close,volume,amt,last_trade_day", date0, date1, "")
            data = temp.Data

            datelist = []
            for i in range(len(data[6])):
                datelist.append(data[6][i].year*10000 + data[6][i].month*100 + data[6][i].day)
            
            hsma1 = pd.DataFrame({'date':datelist, 'open':data[0], 'high':data[1], 'low':data[2], 'close':data[3], 'vol':data[4], 'amt':data[5]})
            hsma1['code'] = code
            
            hsma2 = pd.concat([hsma0, hsma1], ignore_index = True)
            hsma2 = hsma2[hsma0.columns]
            hsma2.to_csv("raw data\\stock\\indexday_" + code + ".csv",index=False)
            