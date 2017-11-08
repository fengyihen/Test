# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import sys;
sys.path.append("Test")
from imp import reload
import Future
reload(Future)
import math
import talib
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


class FutureSklearn(Future.Future):

    def linearregressor(self, testlen, ntrain, raw, timesteps, day, feature_sel):
        
        if raw == True:
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        else:
            hsmadata_x = self.hsmadata_x()
        hsmadata_y = self.hsmadata_y_var(day)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
               
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] <= dates[(i+1)*testlen-1])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]   

            #在train中做变量筛选, sklearn.feature_selection中的方法
            if feature_sel == "SelectKBest":
                selector = SelectKBest()
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            else:
                traindatax1, testdatax1 = traindatax, testdatax 
            
            linearmodel = linear_model.LinearRegression()
            linearmodel.fit(traindatax1, traindatay)
            testdata['predratio'] = linearmodel.predict(testdatax1)
            
            if i == ntrain:
                hsma = testdata.copy()
            else:
                hsma = pd.concat([hsma, testdata], ignore_index = True)
          
        return(hsma)

    def logisticmodel(self, testlen, ntrain, raw, timesteps, day, feature_sel):
        
        if raw == True:
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        else:
            hsmadata_x = self.hsmadata_x()
        hsmadata_y = self.hsmadata_y(day)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
               
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]   

            #在train中做变量筛选, sklearn.feature_selection中的方法
            if feature_sel == "RFECV":
                estimator = LogisticRegression()
                selector = RFECV(estimator, step=1, cv=5)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            elif feature_sel == "SelectFromModel":
                estimator = LogisticRegression()
                selector = SelectFromModel(estimator)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            elif feature_sel == "SelectKBest":
                selector = SelectKBest()
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            else:
                traindatax1, testdatax1 = traindatax, testdatax 
            
            linearmodel = linear_model.LogisticRegression()
            linearmodel.fit(traindatax1, traindatay)
            testdata['predratio'] = linearmodel.predict(testdatax1)
            
            if i == ntrain:
                hsma = testdata.copy()
            else:
                hsma = pd.concat([hsma, testdata], ignore_index = True)
          
        return(hsma)

    def linearvar(self, testlen, ntrain, raw, timesteps, vn, feature_sel):
        
        if raw == True:
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        else:
            hsmadata_x = self.hsmadata_x()
        hsmadata_y = self.hsmadata_y_var(vn)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
               
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] <= dates[(i+1)*testlen-1])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'var'], 1)
            traindatay = traindata['var']            
            testdatax = testdata[traindatax.columns]   

            #在train中做变量筛选, sklearn.feature_selection中的方法
            if feature_sel == "SelectKBest":
                selector = SelectKBest()
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            else:
                traindatax1, testdatax1 = traindatax, testdatax 
            
            linearmodel = linear_model.LinearRegression()
            linearmodel.fit(traindatax1, traindatay)
            testdata['predvar'] = linearmodel.predict(testdatax1)
            
            if i == ntrain:
                hsma = testdata.copy()
            else:
                hsma = pd.concat([hsma, testdata], ignore_index = True)
          
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

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]    

            treemodel = RandomForestRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
        hsma.to_hdf('Test\\stocksklearn\\hsma_extratreesregressor_' + self.label + '.h5', 'hsma')
        
        return(hsma)

    def extratreesregressor(self, testlen, timesteps, day, lr, ntrain, ntrees, nodes):
        
        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)
        
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()        
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] <= dates[i*testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] <= dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            ###选取盈利潜力最大的10个品种训练并预测
            selectcode = traindata.groupby(['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code' : selectcode.code.iloc[0:10]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata,selectcode)
            
            traindatax = traindata.drop(['date', 'code', 'bestp', 'bestp_r'], 1)
            traindatay = traindata['bestp']            
            testdatax = testdata[traindatax.columns]   

            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predp'] = treemodel.predict(testdatax)
            
            for code in testdata.code.unique():
                testdata.loc[testdata.code == code, 'predp'] = testdata.loc[testdata.code == code, 'predp'].iloc[0]
    
            testdata = testdata[testdata.date > dates[i*testlen]]
            
            if i == ntrain:
                hsma = testdata[['code', 'date', 'bestp', 'bestp_r', 'predp']].copy()
            else:
                hsma = pd.concat([hsma, testdata[['code', 'date', 'bestp', 'bestp_r', 'predp']]], ignore_index = True)

          
        return(hsma)

    def extratreesregressor_doublerank(self, testlen, ntrain, ntrees, nodes, col1, tn1, ascending1, col2, tn2, ascending2):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatac = self.conditionrank(traindata, col1, tn1, ascending1)
            traindatac = self.conditionrank(traindatac, col2, tn2, ascending2)
            testdatac = self.conditionrank(testdata, col1, tn1, ascending1)
            testdatac = self.conditionrank(testdatac, col2, tn2, ascending2)
            
            traindatay = traindatac['ratio']
            traindatax = traindatac.drop(['date', 'code', 'ratio', 'rank'], 1)          
            testdatax = testdatac[traindatax.columns]             

            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdatac['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdatac], ignore_index = True)
            hsmatradeday = self.hsmatradedayregressor(condition, hsma, n, cr, cta1)
            stocksklearn.tradestatlist(hsmatradeday)

        hsma.to_hdf('Test\\stocksklearn\\hsma_extratreesregressor_doublerank_' + self.label + '.h5', 'hsma')

        return(hsma)
        
    def scaler_extratreesregressor(self, testlen, ntrain, ntrees, nodes):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]   
            
            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)  
                
            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatas, traindatay)
            testdata['predratio'] = treemodel.predict(testdatas)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
        hsma.to_hdf('Test\\stocksklearn\\hsma_extratreesregressor_' + self.label + '.h5', 'hsma')

        return(hsma)
        
    def GBRTregressor(self, testlen, ntrain, ntrees, nodes):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]   

            treemodel = GradientBoostingRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
        hsma.to_hdf('Test\\stocksklearn\\hsma_GBRTregressor_' + self.label + '.h5', 'hsma')            
            
        return(hsma)
        
    def MLPRegressor(self, testlen, ntrain, *hidden_layer_sizes):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        store = pd.HDFStore('Test\\stocksklearn\\hsma_MLPRegressor_' + self.label + '.h5')
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]    
            
            MLPR = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', random_state=1)
            MLPR.fit(traindatax, traindatay)
            testdata['predratio'] = MLPR.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            store['hsma'] = hsma
            
        store.close()
        return(hsma)

    def logistic_binandwoe(self, testlen, ntrain, feature_sel, varthreshold, cv, binn, bq, r0):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio'] >= r0           
            testdatax = testdata[traindatax.columns]  
            traindatax, testdatax = self.binandwoe_traintest(traindatax, traindatay, testdatax, binn, bq, r0)

            #在train中做变量筛选, sklearn.feature_selection中的方法
            if feature_sel == "VarianceThreshold":
                selector = VarianceThreshold(threshold = varthreshold)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            elif feature_sel == "RFECV":
                estimator = LogisticRegression()
                selector = RFECV(estimator, step=1, cv=cv)
                traindatax1 = pd.DataFrame(selector.fit_transform(traindatax, traindatay))
                traindatax1.columns = traindatax.columns[selector.get_support(True)]
                testdatax1 = testdatax[traindatax1.columns]
            else:
                traindatax1, testdatax1 = traindatax, testdatax  
            
            #训练并预测模型
            classifier = LogisticRegression()  # 使用类，参数全是默认的
            classifier.fit(traindatax1, traindatay)  
            testdata['predratio'] = classifier.predict_proba(testdatax1)[:, 1]
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            hsma.to_hdf('Test\\stocksklearn\\hsma_logisticbinandwoe_' + self.label + '.h5', 'hsma')

        return(hsma)
        
    def extratrees_linearsvr_regressor(self, testlen, ntrain, ntrees, nodes):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        store = pd.HDFStore('Test\\stocksklearn\\hsma_' + self.label + '.h5')
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]   

            treemodel = ExtraTreesRegressor(n_estimators=ntrees,min_samples_split=nodes*2, min_samples_leaf=nodes)
            treemodel.fit(traindatax, traindatay)
            testdata['predratio'] = treemodel.predict(testdatax)

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax) 
            svrmodel = svm.LinearSVR()
            svrmodel.fit(traindatas, traindatay)
            testdata['predratio1'] = svrmodel.predict(testdatas)
            
            testdata = testdata[testdata.predratio1 > 0]
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            store['hsma'] = hsma
            
        store.close()
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
            traindatax = traindata.drop(['ratio'], 1)
            traindatay = traindata['ratio']            
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

    def svcstkmodel(self, testlen, ntrain, kernel, dayr):
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        codelist = pd.Series(hsmadata['code'].unique()).sort_values()
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            for code in codelist:
                hsmadata1 = hsmadata[hsmadata['code'] == code]
                traindata = hsmadata1[(hsmadata1['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata1['date'] < dates[i*testlen - self.day])].copy()
                testdata = hsmadata1[(hsmadata1['date'] >= dates[i*testlen]) & (hsmadata1['date'] < dates[(i+1)*testlen])].copy()        
    
                if (traindata.shape[0]) == 0 or (testdata.shape[0]) == 0:
                    continue
                
                traindata = traindata.iloc[:, 2:]
                traindatax = traindata.drop(['ratio'], 1)
                traindatay = traindata['ratio']            
                traindatay[traindata['ratio'] >= dayr * self.day] = 1
                traindatay[traindata['ratio'] < dayr * self.day] = 0                
                testdatax = testdata[traindatax.columns]   
 
                if len(traindatay.unique()) == 1:
                    continue
                
                scaler = preprocessing.StandardScaler().fit(traindatax)
                traindatas = scaler.transform(traindatax)
                testdatas = scaler.transform(testdatax)    
    
                svrmodel = svm.SVC(kernel = kernel, probability=True)
                svrmodel.fit(traindatas, traindatay)
                testdata['predratio'] = svrmodel.predict_proba(testdatas)[:, 1]
                
                hsma = pd.concat([hsma, testdata], ignore_index = True)
                
        hsma.to_hdf('Test\\stocksklearn\\hsma_svcstkmodel_' + self.label + '.h5', 'hsma')
                
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
            traindatax = traindata.drop(['ratio'], 1)
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
        


    def svc_index(self, testlen, ntrain, index, kernel, dayr):
        
        hsmadata_index = self.hsmadata_index[self.hsmadata_index.code == index]
        if self.mtype != 'None':
            hsmadata_index = hsmadata_index[hsmadata_index['date'] >= self.hsmadata['date'].min()]
        dates = pd.Series(hsmadata_index['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsmaindex = pd.DataFrame()        
        for i in range(ntrain, ntest):
            traindata = hsmadata_index[(hsmadata_index['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata_index['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata_index[(hsmadata_index['date'] >= dates[i*testlen]) & (hsmadata_index['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio'].copy()               
            traindatay[traindata['ratio'] >= dayr * self.day] = 1
            traindatay[traindata['ratio'] < dayr * self.day] = 0   
            testdatax = testdata[traindatax.columns]
            
            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatas = scaler.transform(traindatax)
            testdatas = scaler.transform(testdatax)    

            svrmodel = svm.SVC(kernel = kernel)
            svrmodel.fit(traindatas, traindatay)
            testdata['predratio'] = svrmodel.predict(testdatas)
            
            hsmaindex = pd.concat([hsmaindex, testdata], ignore_index = True)

        return(hsmaindex)

    def MLPRegressor_index(self, testlen, ntrain, index, *hidden_layer_sizes):
        
        hsmadata_index = self.hsmadata_index[self.hsmadata_index.code == index]
        if self.mtype != 'None':
            hsmadata_index = hsmadata_index[hsmadata_index['date'] >= self.hsmadata['date'].min()]
        dates = pd.Series(hsmadata_index['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen        
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata_index[(hsmadata_index['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata_index['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata_index[(hsmadata_index['date'] >= dates[i*testlen]) & (hsmadata_index['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            traindatay = traindata['ratio']            
            testdatax = testdata[traindatax.columns]    
            
            MLPR = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', random_state=1)
            MLPR.fit(traindatax, traindatay)
            testdata['predratio'] = MLPR.predict(testdatax)
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        
    def indexhighlow1day(self, hsmaindex, pr, mr):
        
        hsma0 = pd.read_csv("raw data\\stock\\indexday_" + hsmaindex.code[0] + ".csv")        
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        
        hsma0['high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['high'] - 1
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['high'], 'high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < hsma0['high'], 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0
        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['low']
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['low'], 'low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > hsma0['low'], 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
        
        hsma0 = hsma0[['code', 'date', 'high1ratio', 'low1ratio']].copy()
        
        hsmaindex = pd.merge(hsma0, hsmaindex)
        hsmaindex = hsmaindex.sort_values(by='date')
        
        hsmaindex['ratio'] = 0
        hsmaindex.ix[hsmaindex.predratio > pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio > pr, 'high1ratio']
        hsmaindex.ix[hsmaindex.predratio < -pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio < -pr, 'low1ratio']
  
        return hsmaindex

    def indexhighlow1day_roc(self, hsmaindex, pr, mr, rocn):
        
        hsma0 = pd.read_csv("raw data\\stock\\indexday_" + hsmaindex.code[0] + ".csv")        
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        
        hsma0['high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['high'] - 1
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['high'], 'high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < hsma0['high'], 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0
        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['low']
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['low'], 'low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > hsma0['low'], 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
        
        hsma0 = hsma0[['code', 'date', 'high1ratio', 'low1ratio']].copy()
        
        hsmaindex = pd.merge(hsma0, hsmaindex)
        hsmaindex = hsmaindex.sort_values(by='date')
        
        hsmaindex.ix[hsmaindex['ROC_' + str(rocn)] < 0, 'high1ratio'] = 0
        hsmaindex.ix[hsmaindex['ROC_' + str(rocn)] > 0, 'low1ratio'] = 0
        
        hsmaindex['ratio'] = 0
        hsmaindex.ix[hsmaindex.predratio > pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio > pr, 'high1ratio']
        hsmaindex.ix[hsmaindex.predratio < -pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio < -pr, 'low1ratio']
  
        return hsmaindex

    def closer1day(self, hsma, pr, mr, cr):#cr < mr
        
        hsma0 = self.hsma0.copy()

        buypoints = (hsma0['close'] * (1 + cr))
        for i in buypoints.index:
            buypoints[i]  = math.ceil(buypoints[i] / self.minpoint) * self.minpoint        
        hsma0['high1ratio'] = hsma0['open'].shift(-(self.day+1)) / buypoints - 1
        hsma0.ix[hsma0['open'].shift(-1) > (hsma0['close'] * (1 + cr)), 'high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < (hsma0['close'] * (1 + cr)), 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0

        sellpoints = (hsma0['close'] * (1 - cr))
        for i in sellpoints.index:
            sellpoints[i]  = math.floor(sellpoints[i] / self.minpoint) * self.minpoint          
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / sellpoints
        hsma0.ix[hsma0['open'].shift(-1) < (hsma0['close'] * (1 - cr)), 'low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > (hsma0['close'] * (1 - cr)), 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
       
        hsma0 = hsma0[['code', 'date', 'high1ratio', 'low1ratio']].copy()
        
        hsma = pd.merge(hsma, hsma0)
        hsma = hsma.sort_values(by='date')
        
        hsma['ratio'] = 0
        hsma.ix[(hsma.predratio > pr) & (hsma0['high1ratio'] != 0), 'ratio'] = hsma.ix[(hsma.predratio > pr) & (hsma0['high1ratio'] != 0), 'high1ratio'] - self.fee
        hsma.ix[(hsma.predratio < -pr) & (hsma0['low1ratio'] != 0), 'ratio'] = hsma.ix[(hsma.predratio < -pr) & (hsma0['low1ratio'] != 0), 'low1ratio'] - self.fee

        hsma['dayratio'] = hsma['ratio']/self.day
        hsma['cumratio'] = hsma['dayratio'].cumsum() 
        
        return hsma

    def closer1day_ROC(self, hsma, d, pr, mr, cr):#cr < mr
        
        hsma0 = self.hsma0.copy()
        
        hsma0['high1ratio'] = hsma0['open'].shift(-(self.day+1)) / (hsma0['close'] * (1 + cr)) - 1
        hsma0.ix[hsma0['open'].shift(-1) > (hsma0['close'] * (1 + cr)), 'high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < (hsma0['close'] * (1 + cr)), 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0
        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / (hsma0['close'] * (1 - cr))
        hsma0.ix[hsma0['open'].shift(-1) < (hsma0['close'] * (1 - cr)), 'low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > (hsma0['close'] * (1 - cr)), 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
       
        hsma0 = hsma0[['code', 'date', 'high1ratio', 'low1ratio']].copy()
        
        hsma = pd.merge(self.hsmadata_x, hsma0)
        hsma = hsma.sort_values(by='date')
        
        col = 'ROC_' + str(d)
        
        hsma['ratio'] = 0
        hsma.ix[(hsma[col] > pr) & (hsma0['high1ratio'] != 0), 'ratio'] = hsma.ix[(hsma[col] > pr) & (hsma0['high1ratio'] != 0), 'high1ratio'] - self.fee
        hsma.ix[(hsma[col] < -pr) & (hsma0['low1ratio'] != 0), 'ratio'] = hsma.ix[(hsma[col] < -pr) & (hsma0['low1ratio'] != 0), 'low1ratio'] - self.fee

        hsma['dayratio'] = hsma['ratio']/self.day
        hsma['cumratio'] = hsma['dayratio'].cumsum() 
        
        hsma = hsma.dropna()
        
        return hsma
        
    def closer1dayin(self, hsmaindex, pr, mr, cr):#cr < mr
        
        hsma0 = pd.read_csv("raw data\\stock\\indexday_" + hsmaindex.code[0] + ".csv")        
        hsma0 = hsma0[hsma0['vol'] > 0].sort_values(by='date').copy()
        
        hsma0['high1ratio'] = hsma0['close'].shift(-self.day) / (hsma0['close'] * (1 + cr)) - 1
        hsma0.ix[hsma0['open'].shift(-1) > (hsma0['close'] * (1 + cr)), 'high1ratio'] = hsma0['open'].shift(-self.day) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < (hsma0['close'] * (1 + cr)), 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0
        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-self.day) / (hsma0['close'] * (1 - cr))
        hsma0.ix[hsma0['open'].shift(-1) < (hsma0['close'] * (1 - cr)), 'low1ratio'] = 1 - hsma0['open'].shift(-self.day) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > (hsma0['close'] * (1 - cr)), 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
        
        hsma0 = hsma0[['code', 'date', 'high1ratio', 'low1ratio']].copy()
        
        hsmaindex = pd.merge(hsma0, hsmaindex)
        hsmaindex = hsmaindex.sort_values(by='date')
        
        hsmaindex['ratio'] = 0
        hsmaindex.ix[hsmaindex.predratio > pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio > pr, 'high1ratio']
        hsmaindex.ix[hsmaindex.predratio < -pr, 'ratio'] = hsmaindex.ix[hsmaindex.predratio < -pr, 'low1ratio']
  
        return hsmaindex
        
    def closer1day_base(self, mr, cr):#cr < mr
        
        hsma0 = self.hsma0.copy()       
        
        buypoints = (hsma0['close'] * (1 + cr))
        for i in buypoints.index:
            buypoints[i]  = math.ceil(buypoints[i] / self.minpoint) * self.minpoint
        hsma0['high1ratio'] = hsma0['open'].shift(-(self.day+1)) / buypoints - 1
        hsma0.ix[hsma0['open'].shift(-1) > (hsma0['close'] * (1 + cr)), 'high1ratio'] = hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < (hsma0['close'] * (1 + cr)), 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0

        sellpoints = (hsma0['close'] * (1 - cr))
        for i in sellpoints.index:
            sellpoints[i]  = math.floor(sellpoints[i] / self.minpoint) * self.minpoint        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / sellpoints
        hsma0.ix[hsma0['open'].shift(-1) < (hsma0['close'] * (1 - cr)), 'low1ratio'] = 1 - hsma0['open'].shift(-(self.day+1)) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > (hsma0['close'] * (1 - cr)), 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
               
        hsma0['ratio'] = hsma0['high1ratio'] + hsma0['low1ratio']
        hsma0.ix[hsma0['high1ratio']!=0, 'ratio'] = hsma0.ix[hsma0['high1ratio']!=0, 'ratio'] - self.fee
        hsma0.ix[hsma0['low1ratio']!=0, 'ratio'] = hsma0.ix[hsma0['low1ratio']!=0, 'ratio'] - self.fee
        hsma0['ratio'] = hsma0['ratio'] / 2

        hsma0['dayratio'] = hsma0['ratio']/self.day
        hsma0['cumratio'] = hsma0['dayratio'].cumsum()        
        
        hsma0 = hsma0.dropna()
        hsma0.index = range(hsma0.shape[0])
        return hsma0

    def closer1day_bollinger(self, malen, mr, cr):#cr < mr
        
        hsma0 = self.hsma0.copy()       
        hsma0['MA'] = talib.MA(hsma0.close.values, timeperiod = malen)
        
        buypoints = (hsma0['MA'] * (1 + cr))
        buypoints.iloc[0:(malen-1)] = 999999999
        for i in buypoints.index:
            buypoints[i]  = math.ceil(buypoints[i] / self.minpoint) * self.minpoint
        hsma0['high1ratio'] = hsma0['open'].shift(-2) / buypoints - 1
        hsma0.ix[hsma0['open'].shift(-1) > buypoints, 'high1ratio'] = hsma0['open'].shift(-2) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < buypoints, 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0

        sellpoints = (hsma0['MA'] * (1 - cr))
        sellpoints.iloc[0:(malen-1)] = -999999999
        for i in sellpoints.index:
            sellpoints[i]  = math.floor(sellpoints[i] / self.minpoint) * self.minpoint        
        hsma0['low1ratio'] = 1 - hsma0['open'].shift(-2) / sellpoints
        hsma0.ix[hsma0['open'].shift(-1) < sellpoints, 'low1ratio'] = 1 - hsma0['open'].shift(-2) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > sellpoints, 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
               
        hsma0['ratio'] = hsma0['high1ratio'] + hsma0['low1ratio']
        hsma0.ix[hsma0['high1ratio']!=0, 'ratio'] = hsma0.ix[hsma0['high1ratio']!=0, 'ratio'] - self.fee
        hsma0.ix[hsma0['low1ratio']!=0, 'ratio'] = hsma0.ix[hsma0['low1ratio']!=0, 'ratio'] - self.fee
        hsma0['ratio'] = hsma0['ratio'] / 2

        hsma0['dayratio'] = hsma0['ratio']
        hsma0['cumratio'] = hsma0['dayratio'].cumsum()        
        
        hsma0 = hsma0.dropna()
        hsma0.index = range(hsma0.shape[0])
        return hsma0
        
    def closer1dayin_base(self, mr, cr):#cr < mr
        
        hsma0 = pd.read_csv("raw data\\stock\\" + self.code + ".csv")        
        
        hsma0['high1ratio'] = hsma0['close'].shift(-self.day) / (hsma0['close'] * (1 + cr)) - 1
        hsma0.ix[hsma0['open'].shift(-1) > (hsma0['close'] * (1 + cr)), 'high1ratio'] = hsma0['close'].shift(-self.day) / hsma0['open'].shift(-1) - 1
        hsma0.ix[hsma0['high'].shift(-1) < (hsma0['close'] * (1 + cr)), 'high1ratio'] = 0
        hsma0.ix[hsma0['open'].shift(-1) > hsma0['close'] * (1 + mr), 'high1ratio'] = 0
        
        hsma0['low1ratio'] = 1 - hsma0['close'].shift(-self.day) / (hsma0['close'] * (1 - cr))
        hsma0.ix[hsma0['open'].shift(-1) < (hsma0['close'] * (1 - cr)), 'low1ratio'] = 1 - hsma0['close'].shift(-self.day) / hsma0['open'].shift(-1) 
        hsma0.ix[hsma0['low'].shift(-1) > (hsma0['close'] * (1 - cr)), 'low1ratio'] = 0   
        hsma0.ix[hsma0['open'].shift(-1) < hsma0['close'] * (1 - mr), 'low1ratio'] = 0 
               
        hsma0['ratio'] = (hsma0['high1ratio'] + hsma0['low1ratio']) / 2
        
        hsma0 = hsma0.dropna()

        return hsma0

    def MA(self, malen):
        
        hsma = self.hsma0.copy()
        hsma['MA'] = talib.MA(hsma.close.values, timeperiod = malen)
        hsma = hsma.dropna()
        hsma.index = range(hsma.shape[0])
        hsma['buy'] = 0
        hsma['sell'] = 0
        hsma['fee'] = 0
        hsma['net'] = hsma.ix[0, 'open']
        hsma['LS'] = 'N'
        
        for i in range(1, hsma.shape[0]):
            #初始化当前净值
            hsma.ix[i, 'net'] = hsma.ix[i-1, 'net']
                
            #开仓
            if hsma.ix[i-1, 'LS'] == 'N':
                if (hsma.ix[i-1, 'close'] > hsma.ix[i-1, 'MA']) & (hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA']):
                    hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'close'] - hsma.ix[i, 'buy']
                    hsma.ix[i, 'LS'] = 'L'
                if (hsma.ix[i-1, 'close'] < hsma.ix[i-1, 'MA']) & (hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA']):
                    hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'       
                 
            #平仓，反向开仓
            if hsma.ix[i-1, 'LS'] == 'L':
                if (hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA']):
                    hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i-1, 'close'] + hsma.ix[i, 'sell'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'   
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i, 'close'] - hsma.ix[i-1, 'close']
                    hsma.ix[i, 'LS'] = 'L'                     
            if hsma.ix[i-1, 'LS'] == 'S':
                if (hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA']):
                    hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'buy'] + hsma.ix[i, 'close'] - hsma.ix[i, 'buy']
                    hsma.ix[i, 'LS'] = 'L'
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'                    
        
        hsma['simpleratio'] = hsma['net'] / hsma.ix[0, 'net'] - 1
        hsma['dayratio'] = (hsma['net'] - hsma.net.shift(1)) / hsma.close.shift(1)
        hsma.ix[0, 'dayratio'] = 0
        hsma['cumratio'] = hsma['dayratio'].cumsum()
        
        return hsma

    def MAcross(self, malen1, malen2):
        
        hsma = self.hsma0.copy()
        hsma['MA1'] = talib.MA(hsma.close.values, timeperiod = malen1)
        hsma['MA2'] = talib.MA(hsma.close.values, timeperiod = malen2)
        hsma = hsma.dropna()
        hsma.index = range(hsma.shape[0])
        hsma['buy'] = 0
        hsma['sell'] = 0
        hsma['fee'] = 0
        hsma['net'] = hsma.ix[0, 'open']
        hsma['LS'] = 'N'
        
        for i in range(1, hsma.shape[0]):
            #初始化当前净值
            hsma.ix[i, 'net'] = hsma.ix[i-1, 'net']
                
            #开仓
            if hsma.ix[i-1, 'LS'] == 'N':
                if (hsma.ix[i-1, 'MA1'] > hsma.ix[i-1, 'MA2']) & (hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA1']):
                    hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA1'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'close'] - hsma.ix[i, 'buy']
                    hsma.ix[i, 'LS'] = 'L'
                if (hsma.ix[i-1, 'MA1'] < hsma.ix[i-1, 'MA2']) & (hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA1']):
                    hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA1'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'       
                 
            #平仓，反向开仓
            if hsma.ix[i-1, 'LS'] == 'L':
                if (hsma.ix[i-1, 'MA1'] < hsma.ix[i-1, 'MA2']) & (hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA1']):
                    hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA1'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i-1, 'close']
                    hsma.ix[i, 'LS'] = 'N'   
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i, 'close'] - hsma.ix[i-1, 'close']
                    hsma.ix[i, 'LS'] = 'L'                     
            if hsma.ix[i-1, 'LS'] == 'S':
                if (hsma.ix[i-1, 'MA1'] > hsma.ix[i-1, 'MA2']) & (hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA1']):
                    hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA1'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'buy']
                    hsma.ix[i, 'LS'] = 'N'
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'                    
        
        hsma['simpleratio'] = hsma['net'] / hsma.ix[0, 'net'] - 1
        hsma['dayratio'] = (hsma['net'] - hsma.net.shift(1)) / hsma.close.shift(1)
        hsma.ix[0, 'dayratio'] = 0
        hsma['cumratio'] = hsma['dayratio'].cumsum()
        
        return hsma
               
    def MA_ADX(self, malen, s):
        
        hsma = self.hsma0.copy()
        hsma['MA'] = talib.MA(hsma.close.values, timeperiod = malen)
        hsma['ADX'] = talib.ADX(hsma.high.values,hsma.low.values,hsma.close.values, timeperiod = malen)
        hsma = hsma.dropna()
        hsma.index = range(hsma.shape[0])
        hsma['buy'] = 0
        hsma['sell'] = 0
        hsma['fee'] = 0
        hsma['net'] = hsma.ix[0, 'open']
        hsma['LS'] = 'N'
        
        for i in range(1, hsma.shape[0]):
            #初始化当前净值
            hsma.ix[i, 'net'] = hsma.ix[i-1, 'net']
                
            #开仓
            if hsma.ix[i-1, 'LS'] == 'N':
                if (hsma.ix[i-1, 'close'] > hsma.ix[i-1, 'MA']) & (hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA']) & (hsma.ix[i-1, 'ADX'] >= s):
                    hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'close'] - hsma.ix[i, 'buy']
                    hsma.ix[i, 'LS'] = 'L'
                if (hsma.ix[i-1, 'close'] < hsma.ix[i-1, 'MA']) & (hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA']) & (hsma.ix[i-1, 'ADX'] >= s):
                    hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                    hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee / 2
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'       
                 
            #平仓，反向开仓
            if hsma.ix[i-1, 'LS'] == 'L':
                if hsma.ix[i, 'low'] <= hsma.ix[i-1, 'MA']:
                    if (hsma.ix[i-1, 'close'] < hsma.ix[i-1, 'MA']) & (hsma.ix[i-1, 'ADX'] >= s):
                        hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                        hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee
                        hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i-1, 'close'] + hsma.ix[i, 'sell'] - hsma.ix[i, 'close']
                        hsma.ix[i, 'LS'] = 'S'                         
                    else:
                        hsma.ix[i, 'sell'] = min(hsma.ix[i, 'open'], math.floor(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                        hsma.ix[i, 'fee'] = hsma.ix[i, 'sell'] * self.fee / 2
                        hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i, 'sell'] - hsma.ix[i-1, 'close']
                        hsma.ix[i, 'LS'] = 'N'   
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i, 'close'] - hsma.ix[i-1, 'close']
                    hsma.ix[i, 'LS'] = 'L'                     
            if hsma.ix[i-1, 'LS'] == 'S':
                if hsma.ix[i, 'high'] >= hsma.ix[i-1, 'MA']:
                    if (hsma.ix[i-1, 'close'] > hsma.ix[i-1, 'MA']) & (hsma.ix[i-1, 'ADX'] >= s):
                        hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                        hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee
                        hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'buy'] + hsma.ix[i, 'close'] - hsma.ix[i, 'buy']
                        hsma.ix[i, 'LS'] = 'L'
                    else:
                        hsma.ix[i, 'buy'] = max(hsma.ix[i, 'open'], math.ceil(hsma.ix[i-1, 'MA'] / self.minpoint) * self.minpoint)
                        hsma.ix[i, 'fee'] = hsma.ix[i, 'buy'] * self.fee / 2
                        hsma.ix[i, 'net'] = hsma.ix[i, 'net'] - hsma.ix[i, 'fee'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'buy']
                        hsma.ix[i, 'LS'] = 'N'
                else:
                    hsma.ix[i, 'net'] = hsma.ix[i, 'net'] + hsma.ix[i-1, 'close'] - hsma.ix[i, 'close']
                    hsma.ix[i, 'LS'] = 'S'                    
        
        hsma['simpleratio'] = hsma['net'] / hsma.ix[0, 'net'] - 1
        hsma['dayratio'] = (hsma['net'] - hsma.net.shift(1)) / hsma.close.shift(1)
        hsma.ix[0, 'dayratio'] = 0
        hsma['cumratio'] = hsma['dayratio'].cumsum()
        
        return hsma
     
    def MA_optimizer(self, malens):
        
        tradestatlist = pd.DataFrame()
        for malen in malens:
            hsma = self.MA(malen)
            tradestat = self.tradestat(hsma)
            tradestat['malen'] = malen
            tradestatlist = pd.concat([tradestatlist, tradestat], ignore_index = True)
        
        tradestatlist = tradestatlist.sort_values(by='RRR', ascending=False).copy()
        print(tradestatlist)
        return tradestatlist
"""        
def MA_optimizer_allcode(day, length, timesteps, malens, label):
    
    allcodestat = pd.DataFrame()
    for code in self.feelist.keys():
        print(code)
        day = 1
        length = [1, 3, 5, 10, 20, 40]
        timesteps = 10
        label = 'sklearntest'+str(day)
        futuremodel = futuresklearn.futuresklearn.FutureSklearn(code, day, length, timesteps, label)
        tradestatlist = futuremodel.MA_optimizer(malens)
        allcodestat = pd.concat([allcodestat, tradestatlist], ignore_index = True)    
            
        allcodestat.to_csv("Test\\testresult\\MA_optimizer_allcode_" + self.label + ".csv",index=False)
        
        return allcodestat
"""