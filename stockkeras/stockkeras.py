# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import sys;
sys.path.append("Test")
import time
from Stock import Stock
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.regularizers import l2, activity_l2
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, rmsprop




class StockKerasSequential(Stock):

    def hsmaseq_raw(self, timesteps):
        
        traindata = pd.read_csv("raw data\\stock\\" + self.dataset + ".csv")
        codelist = pd.Series(traindata['code'].unique()).sort_values()
        
        if not os.path.exists('Test\\stockkeras\\hsmaseq_raw_' + str(timesteps) + '_' + self.label):
            os.makedirs('Test\\stockkeras\\hsmaseq_raw_' + str(timesteps) + '_' + self.label)
            
        for code in codelist:
            traindatac = traindata[traindata['code'] == code].copy()
            traindatac = traindatac.sort_values(by='date')
            traindatac['ratio'] = traindatac['open'].shift(-(self.day+1)) / traindatac['open'].shift(-1) - 1   
            traindatac['open'] = traindatac['open'] / traindatac['open'].shift(1) - 1
            traindatac['high'] = traindatac['high'] / traindatac['high'].shift(1) - 1
            traindatac['low'] = traindatac['low'] / traindatac['low'].shift(1) - 1
            traindatac['close'] = traindatac['close'] / traindatac['close'].shift(1) - 1
            traindatac = traindatac.dropna()
            
            if traindatac.shape[0] < timesteps:
                continue               

            traindatac['code'] = int(code[0:6])
            traindatac = traindatac[['date', 'code', 'ratio', 'open', 'high', 'low', 'close', 'free_turn']]
            print(code)
            flag = 0
            for i in range(timesteps):
                traindataci = traindatac.iloc[range(i, traindatac.shape[0]), ]
                traindataci.index = range(traindataci.shape[0])
                nrow = traindataci.shape[0] - traindataci.shape[0] % timesteps  
                traindataci = traindataci.iloc[range(nrow), ]
                
                period = int(traindataci.shape[0] / timesteps)
                xy_temp = traindataci.values.reshape((period, timesteps, traindataci.shape[1]))
                
                if flag == 0:
                    xy_train = xy_temp
                    flag = 1
                else:
                    xy_train = np.concatenate((xy_train,xy_temp))                    
            np.save('Test\\stockkeras\\hsmaseq_raw_' + str(timesteps) + '_' + self.label + '\\' + code,xy_train)

    def hsmaseq(self, timesteps):
        
        traindata = self.hsmadata
        codelist = pd.Series(traindata['code'].unique()).sort_values()
        
        if not os.path.exists('Test\\stockkeras\\hsmaseq_' + str(timesteps) + '_' + self.label):
            os.makedirs('Test\\stockkeras\\hsmaseq_' + str(timesteps) + '_' + self.label)
            
        for code in codelist:
            traindatac = traindata[traindata['code'] == code].copy()
            traindatac = traindatac.sort_values(by='date')
            traindatac = traindatac.dropna()

            if traindatac.shape[0] < timesteps:
                continue   
            
            traindatac['code'] = int(code[0:6])
            print(code)
            flag = 0
            for i in range(timesteps):
                traindataci = traindatac.iloc[range(i, traindatac.shape[0]), ]
                traindataci.index = range(traindataci.shape[0])
                nrow = traindataci.shape[0] - traindataci.shape[0] % timesteps  
                traindataci = traindataci.iloc[range(nrow), ]
                
                period = int(traindataci.shape[0] / timesteps)
                xy_temp = traindataci.values.reshape((period, timesteps, traindataci.shape[1]))
                
                if flag == 0:
                    xy_train = xy_temp
                    flag = 1
                else:
                    xy_train = np.concatenate((xy_train,xy_temp))                    
            np.save('Test\\stockkeras\\hsmaseq_' + str(timesteps) + '_' + self.label + '\\' + code,xy_train)

    def hsmaseqslice(self, timesteps, traindate0, traindate1, raw):
        
        codelist = pd.Series(self.hsmadata['code'].unique()).sort_values()
        
        flag = 0
        for code in codelist:
            if raw == True:
                path = 'Test\\stockkeras\\hsmaseq_raw_'
            else:
                path = 'Test\\stockkeras\\hsmaseq_'
            if not os.path.exists(path + str(timesteps) + '_' + self.label + '\\' + code + '.npy'):
                continue
            print(code)
            xy_train = np.load(path + str(timesteps) + '_' + self.label + '\\' + code + '.npy')
            x_temp = xy_train[(xy_train[:, timesteps-1, 0] >= traindate0) & (xy_train[:, timesteps-1, 0] <= traindate1),:,3:]
            y_temp = xy_train[(xy_train[:, timesteps-1, 0] >= traindate0) & (xy_train[:, timesteps-1, 0] <= traindate1),timesteps-1,0:3]
        
            if flag == 0:
                x_train = x_temp
                y_train = y_temp
                flag = 1
            else:
                x_train = np.concatenate((x_train,x_temp))
                y_train = np.concatenate((y_train,y_temp))
        
        return x_train, y_train

    def hsmacnnslice(self, traindata):
        
        flag = 0
        for colname in ['openr', 'highr', 'lowr', 'closer', 'amtr', 'freeturn']:
            cols = []
            for col in traindata.columns:
                if col[0:4] == colname[0:4]:
                    cols.append(col)
            if flag == 0:
                x_train = traindata[cols]
                x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
                flag = 1
            else:
                temp = traindata[cols]
                temp = temp.values.reshape((temp.shape[0], 1, temp.shape[1]))
                x_train = np.concatenate((x_train, temp), axis=1)  
        
        y_train = traindata['ratio'].values
                
        return x_train, y_train            
                    
    def dnn_regression(self, testlen, ntrain, ntrees, nodes):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        #store = pd.HDFStore('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5')
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatay = traindata['ratio'].values            

            scaler = preprocessing.StandardScaler().fit(traindatax)
            traindatax = scaler.transform(traindatax)
            testdatax = scaler.transform(testdatax)
            
            starttime = time.clock()

            model = Sequential()
            model.add(Dense(64, input_dim=traindatax.shape[1], init='uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, init='uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(16, init='uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, init='uniform'))
            model.add(Activation('linear'))
            
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)
            
            model.fit(traindatax, traindatay, nb_epoch=10, batch_size=2000)
            testdata['predratio'] = model.predict(testdatax)
            
            endtime = time.clock()
            print("The function run time is : %.03f seconds" %(endtime-starttime))
    
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            #store['hsma'] = hsma
            
        #store.close()
        hsma.to_hdf('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5', 'hsma')
        #hsma = pd.read_hdf('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5', 'hsma')
        return(hsma)

    def lstm_regression(self, testlen, ntrain, timesteps, raw):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindate0 = dates[(i-ntrain)*testlen]
            traindate1 = dates[i*testlen - self.day - 1]
            testdate0 = dates[i*testlen]
            testdate1 = dates[(i+1)*testlen - 1]
            
            traindatax, traindatay = self.hsmaseqslice(timesteps, traindate0, traindate1, raw)
            traindatay = traindatay[:, 2]
            traindatax1 = traindatax#[:,:,[0]]
            testdatax, testdatay = self.hsmaseqslice(timesteps, testdate0, testdate1, raw)
            testdatax1 = testdatax#[:,:,[0]]
            testdata = pd.DataFrame(testdatay)
            testdata.columns = ['date', 'code', 'ratio']
            print(testdate1)
            
            starttime = time.clock()

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, traindatax1.shape[2]), W_regularizer=l2(0.01)))#, dropout_W=0.2, dropout_U=0.2
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(LSTM(32, return_sequences=True))#, dropout_W=0.2, dropout_U=0.2
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(LSTM(16))#, dropout_W=0.2, dropout_U=0.2
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, init='uniform'))
            model.add(Activation('linear'))
            
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer='rmsprop')#sgd
            
            model.fit(traindatax1, traindatay, nb_epoch=3, batch_size=2000)
            testdata['predratio'] = model.predict(testdatax1)
            
            endtime = time.clock()
            print("The function run time is : %.03f seconds" %(endtime-starttime))
    
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            
            hsma.to_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return(hsma)

    def lstm_classification(self, testlen, ntrain, timesteps, raw, r):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindate0 = dates[(i-ntrain)*testlen]
            traindate1 = dates[i*testlen - self.day - 1]
            testdate0 = dates[i*testlen]
            testdate1 = dates[(i+1)*testlen - 1]
            
            traindatax, traindatay = self.hsmaseqslice(timesteps, traindate0, traindate1, raw)
            traindatay = traindatay[:, 2] > r
            traindatax1 = traindatax#[:,:,[0]]
            testdatax, testdatay = self.hsmaseqslice(timesteps, testdate0, testdate1, raw)
            testdatax1 = testdatax#[:,:,[0]]
            testdata = pd.DataFrame(testdatay)
            testdata.columns = ['date', 'code', 'ratio']
            print(testdate1)
            
            starttime = time.clock()

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, traindatax1.shape[2]), dropout_W=0.2, dropout_U=0.2))#, W_regularizer=l2(0.01)
            model.add(Activation('sigmoid'))
            #model.add(Dropout(0.5))
            #model.add(LSTM(32, return_sequences=True, dropout_W=0.2, dropout_U=0.2))
            #model.add(Activation('sigmoid'))
            #model.add(Dropout(0.5))
            model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
            model.add(Activation('sigmoid'))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer=sgd)#rmsprop
            
            model.fit(traindatax1, traindatay, nb_epoch=3, batch_size=2000)
            testdata['predratio'] = model.predict_proba(testdatax1)
            
            endtime = time.clock()
            print("The function run time is : %.03f seconds" %(endtime-starttime))
    
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            
            hsma.to_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return(hsma)
        
    def cnn_regression(self, testlen, ntrain, timesteps, raw):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            x_train, y_train = self.hsmacnnslice(traindata)  
            x_test, y_test = self.hsmacnnslice(testdata)
            
            starttime = time.clock()

            model = Sequential()
            model.add(Convolution1D(64, 3, border_mode='same', input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(Activation('relu'))
            # now model.output_shape == (None, 10, 64)            
            # add a new conv1d on top
            model.add(Convolution1D(32, 3, border_mode='same'))
            model.add(Activation('relu'))
            # now model.output_shape == (None, 10, 32)            
            model.add(Flatten())
            model.add(Dense(1))
            model.add(Activation('tanh'))
            
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)
            
            model.fit(x_train, y_train, batch_size=1000, nb_epoch=1)
            testdata['predratio'] = model.predict(x_test)
            
            endtime = time.clock()
            print("The function run time is : %.03f seconds" %(endtime-starttime))
    
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            
            hsma.to_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return(hsma)





        
        
        