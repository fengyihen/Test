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
from WindPy import w
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


class DynamicHedge(object):
    
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
                LTTAQday = pd.read_csv("strategy data\\LTTAQdayMarketMAshort.csv")
                LTdayall = pd.read_csv("strategy data\\LTdayallMarket.csv")
            else:
                LTTAQday = pd.read_csv("strategy data\\LTTAQdayMAshort.csv")
                LTdayall = pd.read_csv("strategy data\\LTdayall.csv")
            LTdayall = LTdayall[LTdayall['coratio'] < 0.09]
            LTdayall = LTdayall[["code","date","ooratio" + str(self.day)]]
            LTdayall = LTdayall.rename(columns = {"ooratio" + str(self.day):'closeratio'})
            hsma = pd.merge(LTTAQday,LTdayall)
        else:
            if self.market == 'market':
                LTdayall = pd.read_csv("strategy data\\LTdayallMarket.csv")
            else:
                LTdayall = pd.read_csv("strategy data\\LTdayall.csv")
            LTdayall = LTdayall[LTdayall['coratio'] < 0.09]
            LTdayall = LTdayall[["code","date","ooratio" + str(self.day)]]
            LTdayall = LTdayall.rename(columns = {"ooratio" + str(self.day):'closeratio'})
            hsma = LTdayall.copy()
        
        if not self.absratio:
            LTindexoo = pd.read_csv("strategy data\\LTindexoo_000905.SH.csv")
            LTindexoo = LTindexoo[['date', "ooratio" + str(self.day)]]
            hsma = pd.merge(hsma,LTindexoo)
            hsma['closeratio'] = hsma['closeratio'] - hsma["ooratio" + str(self.day)]
            hsma = hsma.drop(["ooratio" + str(self.day)],1)

        if self.market == 'market':
            LTdayTTR = pd.read_csv("strategy data\\LTdayTTR0Market_length7_sd2.csv")      
        else:
            LTdayTTR = pd.read_csv("strategy data\\LTdayTTR0_length7_sd2.csv") 
        LTdayTTR = LTdayTTR[["code","date","adx","mfi","rsi","volatility","roc","cci"]]
        LTdayTTR.columns = ["code","date","adx7","mfi7","rsi7","volatility7","roc7","cci7"]
        hsma = pd.merge(hsma,LTdayTTR)
        
        if self.market == 'market':
            LTdayTTR = pd.read_csv("strategy data\\LTdayTTR0Market_length14_sd2.csv")      
        else:
            LTdayTTR = pd.read_csv("strategy data\\LTdayTTR0_length14_sd2.csv") 
        LTdayTTR = LTdayTTR[["code","date","adx","mfi","rsi","volatility","roc","cci"]]
        LTdayTTR.columns = ["code","date","adx14","mfi14","rsi14","volatility14","roc14","cci14"]
        hsma = pd.merge(hsma,LTdayTTR)
        
        if self.market == 'market':
            LTdayvolr = pd.read_csv("strategy data\\LTdayvolrMarket.csv")      
        else:
            LTdayvolr = pd.read_csv("strategy data\\LTdayvolr.csv") 
        hsma = pd.merge(hsma,LTdayvolr)
        
        if self.market == 'market':
            LTdayroc = pd.read_csv("strategy data\\LTdayroc_shortMarket.csv")     
        else:
            LTdayroc = pd.read_csv("strategy data\\LTdayroc_short.csv") 
        hsma = pd.merge(hsma,LTdayroc)

        LTindex = pd.read_csv("strategy data\\LTindexroc_000905.SH.csv")   
        hsma = pd.merge(hsma,LTindex)
        
        if self.market == 'SH':
            
            LTTRANSACTIONday = pd.read_csv("strategy data\\LTTRANSACTIONday.csv")        
            hsma = pd.merge(hsma,LTTRANSACTIONday)

        if self.market == 'SZ':
            
            LTTRADEday = pd.read_csv("strategy data\\LTTRADEday.csv")        
            hsma = pd.merge(hsma,LTTRADEday)
        
        hsma = hsma.dropna()
 
        return(hsma) 
        
    def conditionrank(self, traindata, col, tn):
        dates = pd.Series(traindata['date'].unique()).sort_values()
        
        traindatac = pd.DataFrame()
        for d in dates:
            temp = traindata[traindata['date'] == d].copy()
            temp['rank'] = temp[col].rank(ascending=False)
            traindatac = pd.concat([traindatac,temp[temp['rank'] <= tn]], ignore_index = True)
        
        return(traindatac)
    
    def conditionfilter0(self, hsma0, col, cr):
        
        hsma = hsma0.copy()
        hsma.ix[hsma[col] < cr, 'closeratio'] = self.fee
        
        return(hsma)

    def randomforestregressor(self, testlen, ntrain, ntrees, nodes):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   

            treemodel = RandomForestRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def extratreesregressor(self, testlen, ntrain, ntrees, nodes):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   

            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        
    def extratreesregressor_conditionrank(self, testlen, ntrain, ntrees, nodes, col, tn):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = self.conditionrank(traindata, col, tn)
            testdata = self.conditionrank(testdata, col, tn)
            
            if traindata.shape[0] < 100 | testdata.shape[0] == 0:
                continue

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   

            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def extratreesregressor_cluster(self, testlen, ntrain, ntrees, nodes, columns, cmodel, ncluster):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            testdata['predratio'] = 0

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   
            
            trainlabel, testlabel = self.ClusterModel(traindatax, testdatax, columns, cmodel, ncluster)
            
            for cl in testlabel.unique():
                traindataxcl = traindatax[trainlabel == cl]
                traindataycl = traindatay[trainlabel == cl]
                testdataxcl = testdatax[testlabel == cl]
                treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
                treemodel.fit(traindataxcl, traindataycl)
                testdata.ix[testlabel == cl, 'predratio'] = treemodel.predict(testdataxcl)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
  
    def svrmodel(self, testlen, ntrain, kernel='linear', batch=10000):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            traindata.index = range(0, traindata.shape[0])
            testdata['predratio'] = 0

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)    

            n1 = traindatas.shape[0]
            nbatch = n1 // batch
            for j in range(0, nbatch):
                traindataxb = pd.DataFrame(traindatas).ix[range(j, n1, nbatch), ]
                traindatayb = traindata.ix[range(j, n1, nbatch), 'closeratio']
                svrmodel = svm.NuSVR(kernel = kernel)
                svrmodel.fit(traindataxb, traindatayb)
                testdata['predratio'] = testdata['predratio'] + svrmodel.predict(testdatas)
                
            testdata['predratio'] = testdata['predratio'] / nbatch            
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def linearsvrmodel(self, testlen, ntrain, batch=10000):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            traindata.index = range(0, traindata.shape[0])
            testdata['predratio'] = 0

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            testdatax = testdata[traindatax.columns]   

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)    

            n1 = traindatas.shape[0]
            nbatch = n1 // batch
            for j in range(0, nbatch):
                traindataxb = pd.DataFrame(traindatas).ix[range(j, n1, nbatch), ]
                traindatayb = traindata.ix[range(j, n1, nbatch), 'closeratio']
                svrmodel = svm.LinearSVR()
                svrmodel.fit(traindataxb, traindatayb)
                testdata['predratio'] = testdata['predratio'] + svrmodel.predict(testdatas)
                
            testdata['predratio'] = testdata['predratio'] / nbatch            
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def svcmodel(self, testlen, ntrain, kernel, dayr):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            traindatay[traindata['closeratio'] >= dayr * self.day] = 1
            traindatay[traindata['closeratio'] < dayr * self.day] = 0                
            testdatax = testdata[traindatax.columns]   

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)    

            svrmodel = svm.SVC(kernel = kernel)
            svrmodel.fit(traindatas, traindatay)
            testdata['predratio'] = svrmodel.predict(testdatas)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def knrmodel(self, testlen, ntrain, n_neighbors):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)    

            knrmodel = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors)
            knrmodel.fit(traindatas, traindatay)
            testdata['predratio'] = knrmodel.predict(testdatas)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)


    def GBRTmodel(self, testlen, ntrain, ntrees, nodes):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]  
            
            treemodel = GradientBoostingRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def extratreesclassifier(self, testlen, ntrain, ntrees, nodes, dayr):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            traindatay[traindata['closeratio'] >= dayr * self.day] = 1
            traindatay[traindata['closeratio'] < dayr * self.day] = 0            
            testdatax = testdata[traindatax.columns]   

            treemodel = ExtraTreesClassifier(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def MLPRegressor(self, testlen, ntrain, solver, *hidden_layer_sizes):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindata = traindata.iloc[:, 2:]
            traindatax = traindata.drop(['closeratio'], 1)
            traindatay = traindata['closeratio']            
            testdatax = testdata[traindatax.columns]   
            
            MLPR = MLPRegressor(solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
            MLPR.fit(traindatax, traindatay)
            testdata['predratio'] = MLPR.predict(testdatax)

            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        
    def extratreesregressor_ind(self, testlen, ntrain, ntrees):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       

        stocksector = pd.read_csv("strategy data\\stocksector.csv")
        hsmadata = pd.merge(hsmadata,stocksector)
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            for ind in stocksector['SWN'].unique():
                
                traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day]) & (hsmadata['SWN'] == ind)].copy()
                testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen]) & (hsmadata['SWN'] == ind)].copy()        
                if((traindata.shape[0] < 100) | (testdata.shape[0] < 100)):
                    continue
                
                traindata = traindata.iloc[:, 2:(traindata.shape[1]-1)]
                traindatax = traindata.drop(['closeratio'], 1)
                traindatay = traindata['closeratio']            
                testdatax = testdata[traindatax.columns]   
    
                treemodel = ExtraTreesRegressor(n_estimators=ntrees)
                treemodel.fit(traindatax, traindatay)
                testdata['predratio'] = treemodel.predict(testdatax)
                
                hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        
    def svrmodel_ind(self, testlen, ntrain, kernel='linear'):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       

        stocksector = pd.read_csv("strategy data\\stocksector.csv")
        hsmadata = pd.merge(hsmadata,stocksector)
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            for ind in stocksector['SWN'].unique():
                
                traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day]) & (hsmadata['SWN'] == ind)].copy()
                testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen]) & (hsmadata['SWN'] == ind)].copy()        
                if((traindata.shape[0] < 100) | (testdata.shape[0] < 100)):
                    continue
                
                traindata = traindata.iloc[:, 2:(traindata.shape[1]-1)]
                traindatax = traindata.drop(['closeratio'], 1)
                traindatay = traindata['closeratio']            
                testdatax = testdata[traindatax.columns]   
    
                svrmodel = svm.NuSVR(kernel = kernel)
                svrmodel.fit(traindatax, traindatay)
                testdata['predratio'] = svrmodel.predict(testdatax)
                
                hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def ClusterModel(self, traindata, testdata, columns, cmodel, ncluster):
        traindata = traindata[columns]      
        testdata = testdata[columns].copy()
        
        if cmodel == 'KMeans':
            kmeans = KMeans(n_clusters=ncluster).fit(traindata)
            testdata['cluster'] = kmeans.predict(testdata)
        
        return(kmeans.labels_, testdata['cluster'])
        
    def hsmatrade(self, hsma, n):
        #根据指标选股          
        dates = pd.Series(hsma['date'].unique()).sort_values()

        if not self.absratio:
            LTindexoo = pd.read_csv("strategy data\\LTindexoo_000905.SH.csv")
            LTindexoo = LTindexoo[['date', "ooratio" + str(self.day)]]
            hsma = pd.merge(hsma,LTindexoo)
            hsma['closeratio'] = hsma['closeratio'] + hsma["ooratio" + str(self.day)]
            hsma = hsma.drop(["ooratio" + str(self.day)],1)
        
        hsmatrade = pd.DataFrame()
        for d in dates:
            hsmad = hsma[hsma['date'] == d].copy()
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
        
        
        return(hsmatrade) 
        
    def hsmatradedayclassifier(self, hsma, minn, idxma):
        
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
        
    def hsmapredmono(self, hsma, qn = 10):
        
        dates = pd.Series(hsma['date'].unique()).sort_values()
        
        hsmaall = pd.DataFrame()
        for d in dates:
            temp = hsma[hsma['date'] == d]
            for i in range(0, qn):
                tempi = temp[(temp['predratio'] >= temp['predratio'].quantile(i/qn)) & (temp['predratio'] <= temp['predratio'].quantile((i+1)/qn))].copy()
                tempi['quantile'] = i + 1
                hsmaall = pd.concat([hsmaall, tempi[['date', 'closeratio', 'quantile']]], ignore_index = True)
                
        hsmapredmono = hsmaall.groupby(['quantile'], as_index=False)[['closeratio']].mean()
        return(hsmapredmono)

    def hsmapredIC(self, hsma):
        
        dates = pd.Series(hsma['date'].unique()).sort_values()
        
        ICSeries = pd.Series([])
        for d in dates:
            temp = hsma[hsma['date'] == d]
            icd = temp.predratio.corr(temp.closeratio)
            ICSeries = ICSeries.append(pd.Series([icd])) 
                
        print(ICSeries.mean())
        
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

            
    def extratreestrade_op(self, testlen, ntrain, ntrees, nodes, n, testdate, label):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        codelist = pd.DataFrame({'code':hsmadata['code'].unique()})
        
        i = ntest
        traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()

        traindata = traindata.iloc[:, 2:]
        traindatax = traindata.drop(['closeratio'], 1)
        traindatay = traindata['closeratio']            

        treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
        treemodel.fit(traindatax, traindatay)
        
        if self.datataq:
            testdata = pd.read_csv("Test\\dynamic hedge\\testresult\\DynamicHedge_testdata_taq.csv")#market='market'
        else:
            testdata = pd.read_csv("Test\\dynamic hedge\\testresult\\DynamicHedge_testdata_notaq.csv")#market='market'
        
        testdata = pd.merge(codelist,testdata)
        testdatax = testdata[traindatax.columns]
        testdata['predratio'] = treemodel.predict(testdatax)
        
        hsmad = testdata[testdata['date'] == testdate].copy()
        hsmad['rank'] = hsmad['predratio'].rank(ascending=False)
        hsmad = hsmad[hsmad['rank'] <= n]
        hsmad.to_csv("Test\\dynamic hedge\\testresult\\DynamicHedge_op_" + str(testdate) + "_" + label +".csv",index=False)


        
        
        
        