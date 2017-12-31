# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""

import os
import sys
sys.path.append("Test")
import time
from imp import reload
import FutureDay
reload(FutureDay)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras import backend

#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1


class FutureKeras(FutureDay.Future):
    def hsmaseq(self, traindatax, timesteps, data_dim):

        for i in range(traindatax.shape[0]):
            temp = traindatax[i, ]
            temp = temp.reshape(1, timesteps, data_dim)

            if i == 0:
                traindata_lstm = temp
            else:
                traindata_lstm = np.concatenate((traindata_lstm, temp))

        return traindata_lstm

    def hsmaseqslice(self, timesteps, traindate0, traindate1, raw):

        codelist = pd.Series(self.hsmadata['code'].unique()).sort_values()

        flag = 0
        for code in codelist:
            if raw == True:
                path = 'Test\\stockkeras\\hsmaseq_raw_'
            else:
                path = 'Test\\stockkeras\\hsmaseq_'
            if not os.path.exists(path + str(timesteps) + '_' + self.label +
                                  '\\' + code + '.npy'):
                continue
            print(code)
            xy_train = np.load(path + str(timesteps) + '_' + self.label + '\\'
                               + code + '.npy')
            x_temp = xy_train[(xy_train[:, timesteps - 1, 0] >= traindate0) & (
                xy_train[:, timesteps - 1, 0] <= traindate1), :, 3:]
            y_temp = xy_train[(xy_train[:, timesteps - 1, 0] >= traindate0) &
                              (xy_train[:, timesteps - 1, 0] <=
                               traindate1), timesteps - 1, 0:3]

            if flag == 0:
                x_train = x_temp
                y_train = y_temp
                flag = 1
            else:
                x_train = np.concatenate((x_train, x_temp))
                y_train = np.concatenate((y_train, y_temp))

        return x_train, y_train

    def hsmacnnslice(self, traindata):

        flag = 0
        for colname in [
                'openr', 'highr', 'lowr', 'closer', 'volr', 'openintr'
        ]:
            cols = []
            for col in traindata.columns:
                if col[0:len(colname)] == colname[0:len(colname)]:
                    cols.append(col)
            if flag == 0:
                x_train = traindata[cols]
                x_train = x_train.values.reshape((x_train.shape[0], 1,
                                                  x_train.shape[1]))
                flag = 1
            else:
                temp = traindata[cols]
                temp = temp.values.reshape((temp.shape[0], 1, temp.shape[1]))
                x_train = np.concatenate((x_train, temp), axis=1)

        y_train = traindata['ratio'].values

        return x_train, y_train

    def LSTM_1_cls(self, X_train, y_train, X_test, timesteps, epochs,
                   batchsize, activation):
        #单层LSTM模型
        model = Sequential()
        model.add(
            LSTM(
                64,
                activation=activation,
                return_sequences=False,
                input_shape=(
                    timesteps,
                    X_train.shape[2])))  #, dropout_W=0.2, dropout_U=0.2
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize)
        prob = model.predict(X_test)
        predp = model.predict(X_test).argmax(axis=1)

        backend.clear_session()

        return prob, predp

    def LSTM_2_cls(self, X_train, y_train, X_test, timesteps, epochs,
                   batchsize, activation):
        #单层LSTM模型
        model = Sequential()
        model.add(
            LSTM(
                128,
                activation=activation,
                return_sequences=True,
                input_shape=(
                    timesteps,
                    X_train.shape[2])))  #, dropout_W=0.2, dropout_U=0.2
        model.add(LSTM(64, activation=activation, return_sequences=True))
        model.add(LSTM(32, activation=activation, return_sequences=False))
        model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize)
        prob = model.predict(X_test)
        predp = model.predict(X_test).argmax(axis=1)

        backend.clear_session()

        return prob, predp

    def base_regression(self, testlen, ntrain, epochs, timesteps, ncode, day,
                        lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            testdata['predp_short'] = 0
            testdata['predp_long'] = 0

            for code in testdata.code.unique():
                train1 = traindata[traindata.code == code].copy()
                testdata.loc[
                    testdata.code == code, 'predp_short'] = train1.loc[
                        train1.date == train1.date.max(), 'bestp_short'].values
                testdata.loc[testdata.code == code, 'predp_long'] = train1.loc[
                    train1.date == train1.date.max(), 'bestp_long'].values

            testdata = testdata[testdata.date > dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'predp_short',
                                 'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        testdata[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def dnn_regression(self, testlen, ntrain, epochs, timesteps, day, lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:10]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            traindatax = traindata.drop(['date', 'code', 'bestp', 'bestp_r'],
                                        1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatay = traindata['bestp'].values

            #scaler = preprocessing.StandardScaler().fit(traindatax)
            #traindatax = scaler.transform(traindatax)
            #testdatax = scaler.transform(testdatax)

            starttime = time.clock()

            model = Sequential()
            model.add(
                Dense(
                    64,
                    input_dim=traindatax.shape[1],
                    kernel_initializer='uniform'))
            model.add(Activation('sigmoid'))
            #model.add(Dropout(0.5))
            model.add(
                Dense(
                    32,
                    input_dim=traindatax.shape[1],
                    kernel_initializer='uniform'))
            model.add(Activation('sigmoid'))
            model.add(Dense(1, kernel_initializer='uniform'))
            model.add(Activation('linear'))

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)

            model.fit(traindatax, traindatay, epochs=epochs, batch_size=2000)
            testdata['predp'] = model.predict(testdatax)

            endtime = time.clock()
            print("The function run time is : %.03f seconds" %
                  (endtime - starttime))

            for code in testdata.code.unique():
                testdata.loc[testdata.code == code, 'predp'] = testdata.loc[
                    testdata.code == code, 'predp'].iloc[0]

            testdata = testdata[testdata.date > dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'bestp', 'bestp_r',
                                 'predp']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        testdata[['code', 'date', 'bestp', 'bestp_r', 'predp']]
                    ],
                    ignore_index=True)

        #hsma.to_hdf('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def dnn_regression_p2(self, testlen, ntrain, epochs, timesteps, day, lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:10]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            traindatax = traindata.drop([
                'date', 'code', 'bestp_short', 'bestp_short_r', 'bestp_long',
                'bestp_long_r', 'bestp_r'
            ], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatay_short = traindata['bestp_short'].values
            traindatay_long = traindata['bestp_long'].values

            starttime = time.clock()

            model = Sequential()
            model.add(
                Dense(
                    128,
                    input_dim=traindatax.shape[1],
                    kernel_initializer='uniform'))
            model.add(Activation('linear'))
            #model.add(Dropout(0.5))
            model.add(Dense(64, kernel_initializer='uniform'))
            model.add(Activation('linear'))
            #model.add(Dropout(0.5))
            model.add(Dense(32, kernel_initializer='uniform'))
            model.add(Activation('linear'))
            #model.add(Dropout(0.5))
            model.add(Dense(16, kernel_initializer='uniform'))
            model.add(Activation('linear'))
            #model.add(Dropout(0.5))
            model.add(Dense(1, kernel_initializer='uniform'))
            model.add(Activation('linear'))

            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)  #
            model.compile(loss='mean_squared_error', optimizer=sgd)

            model.fit(
                traindatax, traindatay_short, epochs=epochs, batch_size=2000)
            testdata['predp_short'] = model.predict(testdatax)

            model.fit(
                traindatax, traindatay_long, epochs=epochs, batch_size=2000)
            testdata['predp_long'] = model.predict(testdatax)

            endtime = time.clock()
            print("The function run time is : %.03f seconds" %
                  (endtime - starttime))

            for code in testdata.code.unique():
                testdata.loc[testdata.code == code,
                             'predp_short'] = testdata.loc[
                                 testdata.code == code, 'predp_short'].iloc[0]
                testdata.loc[testdata.code == code,
                             'predp_long'] = testdata.loc[
                                 testdata.code == code, 'predp_long'].iloc[0]

            testdata = testdata[testdata.date > dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'predp_short',
                                 'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        testdata[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)

        #hsma.to_hdf('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def cnn1D_regression_p2(self, testlen, ntrain, epochs, timesteps, day, lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:10]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            traindatax = traindata.drop([
                'date', 'code', 'bestp_short', 'bestp_short_r', 'bestp_long',
                'bestp_long_r', 'bestp_r'
            ], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatay_short = traindata['bestp_short'].values
            traindatay_long = traindata['bestp_long'].values

            starttime = time.clock()

            model = Sequential()
            model.add(
                Conv1D(
                    64, 3, activation='relu', input_dim=traindatax.shape[1]))
            model.add(Conv1D(64, 3, activation='relu'))
            model.add(MaxPooling1D(3))
            model.add(Conv1D(128, 3, activation='relu'))
            model.add(Conv1D(128, 3, activation='relu'))
            #model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)  #
            model.compile(loss='mean_squared_error', optimizer=sgd)

            model.fit(
                traindatax, traindatay_short, epochs=epochs, batch_size=2000)
            testdata['predp_short'] = model.predict(testdatax)

            model.fit(
                traindatax, traindatay_long, epochs=epochs, batch_size=2000)
            testdata['predp_long'] = model.predict(testdatax)

            endtime = time.clock()
            print("The function run time is : %.03f seconds" %
                  (endtime - starttime))

            for code in testdata.code.unique():
                testdata.loc[testdata.code == code,
                             'predp_short'] = testdata.loc[
                                 testdata.code == code, 'predp_short'].iloc[0]
                testdata.loc[testdata.code == code,
                             'predp_long'] = testdata.loc[
                                 testdata.code == code, 'predp_long'].iloc[0]

            testdata = testdata[testdata.date > dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'predp_short',
                                 'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        testdata[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)

        #hsma.to_hdf('Test\\stockkeras\\hsma_dnn_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def lstm_regression(self, testlen, ntrain, epochs, timesteps, ncode, day,
                        lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            hsma0 = pd.DataFrame()
            for code in traindata.code.unique():

                traindatac = traindata[traindata.code == code].copy()
                testdatac = testdata[testdata.code == code].copy()

                traindatax = traindatac.drop([
                    'date', 'code', 'bestp_short', 'bestp_short_r',
                    'bestp_long', 'bestp_long_r', 'bestp_r'
                ], 1)
                testdatax = testdatac[traindatax.columns].values
                traindatax = traindatax.values
                traindatax_lstm = self.hsmaseq(
                    traindatax, timesteps, data_dim=6)
                testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
                traindatay_short = traindatac['bestp_short'].values
                traindatay_long = traindatac['bestp_long'].values

                model = Sequential()
                model.add(
                    LSTM(
                        64,
                        activation='tanh',
                        return_sequences=False,
                        input_shape=(timesteps, traindatax_lstm.shape[2]
                                     )))  #, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                #model.add(LSTM(32, return_sequences=True))#, dropout_W=0.2, dropout_U=0.2
                #model.add(Activation('relu'))
                #model.add(Dropout(0.5))
                #model.add(LSTM(16, activation='linear'))#, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                model.add(Dense(1))
                model.add(Activation('linear'))

                sgd = SGD(lr=0.01, nesterov=True)
                model.compile(loss='mean_squared_error', optimizer=sgd)  #sgd

                model.fit(
                    traindatax_lstm,
                    traindatay_short,
                    epochs=epochs,
                    batch_size=2000)
                testdatac['predp_short'] = model.predict(testdatax_lstm)

                model.fit(
                    traindatax_lstm,
                    traindatay_long,
                    epochs=epochs,
                    batch_size=2000)
                testdatac['predp_long'] = model.predict(testdatax_lstm)

                testdatac.loc[
                    testdatac.code == code, 'predp_short'] = testdatac.loc[
                        testdatac.code == code, 'predp_short'].iloc[0]
                testdatac.loc[testdatac.code == code,
                              'predp_long'] = testdatac.loc[
                                  testdatac.code == code, 'predp_long'].iloc[0]

                testdatac = testdatac[testdatac.date > dates[i * testlen]]

                hsma0 = pd.concat([hsma0, testdatac], ignore_index=True)

            if i == ntrain:
                hsma = hsma0[['code', 'date', 'predp_short',
                              'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        hsma0[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def lstm_classification_fixma(self, testlen, ntrain, epochs, batchsize,
                                  timesteps, ncode, day, m, lr, tr,
                                  modellabel):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_fixma(day, m, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'r'], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
            traindatay = to_categorical(traindata['r'].values > tr)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize)
                testdata['prob'] = prob[:, 1]
            elif modellabel == 'LSTM2cls':
                prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize)
                testdata['prob'] = prob[:, 1]
            else:
                pass

            codeprob = pd.DataFrame()
            for code in testdata.code.unique():
                temp = pd.DataFrame(
                    {
                        'code': code,
                        'prob':
                        testdata.loc[testdata.code == code, 'prob'].iloc[0]
                    },
                    index=[0])
                codeprob = pd.concat([codeprob, temp], ignore_index=True)

            ###选取盈利潜力最大的10个品种训练并预测
            codeprob.sort_values(by='prob', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': codeprob.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            testdata = testdata[testdata.date == dates[i * testlen]]

            if i == ntrain:
                hsma = testdata[['code', 'date', 'r']].copy()
            else:
                hsma = pd.concat(
                    [hsma, testdata[['code', 'date', 'r']]], ignore_index=True)
            hsma.to_hdf('Test\\futurekeras\\testresult\\hsma_lstm_cls_fixma_' +
                        self.label + '_' + modellabel + '.h5', 'hsma')

        return (hsma)

    def lstm_classification_std(self, testlen, ntrain, length_t, epochs,
                                batchsize, timesteps, day, tr, activation,
                                attr, modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        elif attr == 'rawcci':
            hsmadata_x = self.hsmadata_rawcci_x(timesteps, length_t)
        else:
            print('Wrong Attr!')

        hsmadata_y = self.hsmadata_std(day)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_std_day' + str(
            day
        ) + '_attr' + str(attr) + '_length_t' + str(length_t) + '_tr' + str(
            tr) + '_timesteps' + str(timesteps) + '_' + str(
                activation) + '_' + modellabel + '_' + self.label + '.h5'

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
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            preddate = dates[i * testlen]
            startdate = min(testdata.date[testdata.date > preddate])
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.enddate.max():
                    continue
            print(testdata.date.max())

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'std_day'], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
            traindatay = to_categorical(traindata['std_day'].values > tr)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob'] = prob[:, 1]
            elif modellabel == 'LSTM2cls':
                prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob'] = prob[:, 1]
            else:
                pass

            codeprob = pd.DataFrame()
            for code in testdata.code.unique():
                if any((testdata.code == code) & (testdata.date == preddate)):
                    temp = pd.DataFrame(
                        {
                            'code':
                            code,
                            'preddate':
                            preddate,
                            'startdate':
                            startdate,
                            'enddate':
                            enddate,
                            'prob':
                            testdata.loc[(testdata.code == code) & (
                                testdata.date == preddate), 'prob'].values
                        },
                        index=[0])
                    codeprob = pd.concat([codeprob, temp], ignore_index=True)

            hsma = pd.concat([hsma, codeprob], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def lstm_cls_breakhlfixlength(self, testlen, ntrain, length, epochs,
                                  batchsize, timesteps, lr, tr, activation,
                                  attr, modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        else:
            print('Wrong Attr!')

        hsmadata_y = self.hsmadata_breakhl_fixlength(testlen, length, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_breakhlfixlength_testlen' + str(
            testlen) + '_attr' + str(attr) + '_length' + str(length) + '_tr' + str(
            tr) + '_timesteps' + str(timesteps) + '_' + str(
            activation) + '_' + modellabel + '_' + self.label + '.h5'

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
                & (hsmadata['date'] <= dates[(i + 1) * testlen])].copy()
            preddate = dates[i * testlen]
            startdate = min(testdata.date[testdata.date > preddate])
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.enddate.max():
                    continue
            print(enddate)

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'r'], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
            traindatay = to_categorical(traindata['r'].values > tr)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob'] = prob[:, 1]
            elif modellabel == 'LSTM2cls':
                prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob'] = prob[:, 1]
            else:
                pass

            codeprob = pd.DataFrame()
            for code in testdata.code.unique():
                if any((testdata.code == code) & (testdata.date == preddate)):
                    temp = pd.DataFrame(
                        {
                            'code':
                            code,
                            'preddate':
                            preddate,
                            'startdate':
                            startdate,
                            'enddate':
                            enddate,
                            'prob':
                            testdata.loc[(testdata.code == code) & (
                                testdata.date == preddate), 'prob'].values
                        },
                        index=[0])
                    codeprob = pd.concat([codeprob, temp], ignore_index=True)

            hsma = pd.concat([hsma, codeprob], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def lstm_classification_fixpv(self, testlen, ntrain, length_t, epochs,
                                 batchsize, timesteps, day, p, v, lr, tr,
                                 activation, attr, yvar, modellabel, readfile):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        elif attr == 'rawcci':
            hsmadata_x = self.hsmadata_rawcci_x(timesteps, length_t)
        else:
            print('Wrong Attr!')

        if yvar == 'fixp':
            hsmadata_y = self.hsmadata_fixp(day, p, lr)
            filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_{}_p{}_day{}_attr{}_length_t{}_tr{}_timesteps{}_{}_{}_{}.h5'.format(
            yvar,p,day,attr,length_t,tr,timesteps,activation,modellabel,self.label)
        elif yvar == 'fixvar':
            hsmadata_y = self.hsmadata_fixvar(day, v, lr)
            filename = 'Test\\futurekeras\\testresult\\hsma_lstm_cls_{}_v{}_day{}_attr{}_length_t{}_tr{}_timesteps{}_{}_{}_{}.h5'.format(
            yvar,v,day,attr,length_t,tr,timesteps,activation,modellabel,self.label)
        else:
            print('Wrong Yvar!')
            
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

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
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            preddate = dates[i * testlen]
            startdate = min(testdata.date[testdata.date > preddate])
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.enddate.max():
                    continue
            print(testdata.date.max())

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'short_r', 'long_r'],
                                        1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
            traindatay_short = to_categorical(traindata['short_r'].values > tr)
            traindatay_long = to_categorical(traindata['long_r'].values > tr)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay_short,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob_short'] = prob[:, 1]

                prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay_long,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob_long'] = prob[:, 1]
            elif modellabel == 'LSTM2cls':
                prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay_short,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob_short'] = prob[:, 1]

                prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay_long,
                                          testdatax_lstm, timesteps, epochs,
                                          batchsize, activation)
                testdata['prob_long'] = prob[:, 1]
            else:
                pass

            testdata['prob'] = testdata['prob_long'] + testdata['prob_short']
            codeprob = pd.DataFrame()
            for code in testdata.code.unique():
                if any((testdata.code == code) & (testdata.date == preddate)):
                    temp = pd.DataFrame(
                        {
                            'code':
                            code,
                            'preddate':
                            preddate,
                            'startdate':
                            startdate,
                            'enddate':
                            enddate,
                            'prob_long':
                            testdata.loc[(testdata.code == code) &
                                         (testdata.date == preddate),
                                         'prob_long'].values,
                            'prob_short':
                            testdata.loc[(testdata.code == code) &
                                         (testdata.date == preddate),
                                         'prob_short'].values,
                            'prob':
                            testdata.loc[(testdata.code == code) & (
                                testdata.date == preddate), 'prob'].values
                        },
                        index=[0])
                    codeprob = pd.concat([codeprob, temp], ignore_index=True)

            hsma = pd.concat([hsma, codeprob], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def lstm_classification_fixp_pred(
            self, preddate, testlen, ntrain, length_t, epochs, batchsize,
            timesteps, day, p, lr, tr, activation, attr, modellabel):

        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        elif attr == 'rawcci':
            hsmadata_x = self.hsmadata_rawcci_x(timesteps, length_t)
        else:
            print('Wrong Attr!')

        hsmadata_y = self.hsmadata_fixp(day, p, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata_x['date'].unique()).sort_values()
        dates.index = range(0, len(dates))

        filename = 'Test\\futurekeras\\predresult\\hsma_lstm_cls_fixp_p{}_day{}_attr{}_length_t{}_tr{}_timesteps{}_{}_{}_{}_{}.csv'.format(
                    p,day,attr,length_t,tr,timesteps,activation,modellabel,self.label,preddate)

        predidx = np.where(dates == preddate)[0][0]
        traindata = hsmadata[
            (hsmadata['date'] >= dates[predidx - ntrain * testlen])
            & (hsmadata['date'] <= dates[predidx - day - 1])].copy()
        testdata = hsmadata_x[hsmadata_x['date'] == preddate].copy()

        ###变换数据集成LSTM所需格式
        traindatax = traindata.drop(['date', 'code', 'short_r', 'long_r'], 1)
        testdatax = testdata[traindatax.columns].values
        traindatax = traindatax.values
        traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
        testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
        traindatay_short = to_categorical(traindata['short_r'].values > tr)
        traindatay_long = to_categorical(traindata['long_r'].values > tr)

        ###建模并预测
        if modellabel == 'LSTM1cls':
            prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay_short,
                                      testdatax_lstm, timesteps, epochs,
                                      batchsize, activation)
            testdata['prob_short'] = prob[:, 1]

            prob, _ = self.LSTM_1_cls(traindatax_lstm, traindatay_long,
                                      testdatax_lstm, timesteps, epochs,
                                      batchsize, activation)
            testdata['prob_long'] = prob[:, 1]
        elif modellabel == 'LSTM2cls':
            prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay_short,
                                      testdatax_lstm, timesteps, epochs,
                                      batchsize, activation)
            testdata['prob_short'] = prob[:, 1]

            prob, _ = self.LSTM_2_cls(traindatax_lstm, traindatay_long,
                                      testdatax_lstm, timesteps, epochs,
                                      batchsize, activation)
            testdata['prob_long'] = prob[:, 1]
        else:
            pass

        testdata['prob'] = testdata['prob_long'] + testdata['prob_short']
        codeprob = pd.DataFrame()
        for code in testdata.code.unique():
            if any((testdata.code == code) & (testdata.date == preddate)):
                temp = pd.DataFrame(
                    {
                        'code':
                        code,
                        'preddate':
                        preddate,
                        'prob_long':
                        testdata.loc[(testdata.code == code) & (
                            testdata.date == preddate), 'prob_long'].values,
                        'prob_short':
                        testdata.loc[(testdata.code == code) & (
                            testdata.date == preddate), 'prob_short'].values,
                        'prob':
                        testdata.loc[(testdata.code == code) & (
                            testdata.date == preddate), 'prob'].values
                    },
                    index=[0])
                codeprob = pd.concat([codeprob, temp], ignore_index=True)

        codeprob.sort_values(by='prob', ascending=False, inplace=True)

        codeprob.to_csv(filename)

        return codeprob

    def lstm_classification_r(self, testlen, ntrain, epochs,
                                batchsize, timesteps, day, tr, activation,
                                attr, attry, modellabel, readfile):
        if attr == 'raw':
            hsmadata_x = self.hsmadata_raw_x(timesteps)
        elif attr == 'raw2':
            hsmadata_x = self.hsmadata_raw_x2(timesteps)
        elif attr == 'rawcci':
            hsmadata_x = self.hsmadata_rawcci_x(timesteps)
        else:
            print('Wrong Attr!')
        
        if attry == 'roc':
            hsmadata_y = self.hsmadata_roc(day)
        elif attry == 'roo':
            hsmadata_y = self.hsmadata_roo(day)
        else:
            print('Wrong Attr_y!')
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        filename = 'testresult\\futurekeras\\train\\hsma_lstm_cls_r_day' + str(
            day
        ) + '_attr' + str(attr) + '_attry' + str(attry) + '_tr' + str(
            tr) + '_timesteps' + str(timesteps) + '_' + str(
                activation) + '_' + modellabel + '_' + self.label + '.h5'

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
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            startdate = dates[i * testlen]
            enddate = testdata.date.max()
            if hsma.shape[0] > 0:
                if startdate <= hsma.date.max():
                    continue
            print(enddate)

            ###变换数据集成LSTM所需格式
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1)
            testdatax = testdata[traindatax.columns].values
            traindatax = traindatax.values
            dim = int(traindatax.shape[1] / timesteps)
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=dim)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=dim)
            traindatay_short = to_categorical(traindata['ratio'].values < -tr)
            traindatay_long = to_categorical(traindata['ratio'].values > tr)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                prob, predp = self.LSTM_1_cls(
                    traindatax_lstm, traindatay_short, testdatax_lstm,
                    timesteps, epochs, batchsize, activation)
                testdata['prob_short'] = prob[:, 1]

                prob, predp = self.LSTM_1_cls(traindatax_lstm, traindatay_long,
                                              testdatax_lstm, timesteps,
                                              epochs, batchsize, activation)
                testdata['prob_long'] = prob[:, 1]
            elif modellabel == 'LSTM2cls':
                prob, predp = self.LSTM_2_cls(
                    traindatax_lstm, traindatay_short, testdatax_lstm,
                    timesteps, epochs, batchsize, activation)
                testdata['prob_short'] = prob[:, 1]

                prob, predp = self.LSTM_2_cls(traindatax_lstm, traindatay_long,
                                              testdatax_lstm, timesteps,
                                              epochs, batchsize, activation)
                testdata['prob_long'] = prob[:, 1]
            else:
                break

            hsmatemp = testdata[['date', 'code', 
                                 'ratio', 'prob_long', 'prob_short']]

            hsma = pd.concat([hsma, hsmatemp], ignore_index=True)
            hsma.to_hdf(filename, 'hsma')

        return (hsma)

    def lstm_classification_bestp(self, testlen, ntrain, epochs, batchsize,
                            timesteps, ncode, day, lr, activation, modellabel):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day - 1])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

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
            traindatax_lstm = self.hsmaseq(traindatax, timesteps, data_dim=6)
            testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
            traindatay = to_categorical(traindata['bestp'].values)

            ###建模并预测
            if modellabel == 'LSTM1cls':
                _, testdata['predp'] = self.LSTM_1_cls(
                    traindatax_lstm, traindatay, testdatax_lstm, timesteps,
                    epochs, batchsize, activation)
            elif modellabel == 'LSTM2cls':
                _, testdata['predp'] = self.LSTM_2_cls(
                    traindatax_lstm, traindatay, testdatax_lstm, timesteps,
                    epochs, batchsize, activation)
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
            hsma.to_hdf('Test\\futurekeras\\testresult\\hsma_lstm_cls_' +
                        self.label + '_' + modellabel + '.h5', 'hsma')

        return (hsma)

    def lstm_classification_bestp_code(self, testlen, ntrain, epochs, batchsize,
                                 timesteps, ncode, day, lr, modellabel):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            hsma0 = pd.DataFrame()
            for code in traindata.code.unique():

                traindatac = traindata[traindata.code == code].copy()
                testdatac = testdata[testdata.code == code].copy()

                traindatax = traindatac.drop([
                    'date', 'code', 'bestp_short', 'bestp_short_r',
                    'bestp_long', 'bestp_long_r', 'bestp_r'
                ], 1)
                testdatax = testdatac[traindatax.columns].values
                traindatax = traindatax.values
                traindatax_lstm = self.hsmaseq(
                    traindatax, timesteps, data_dim=6)
                testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
                traindatay_short = to_categorical(
                    traindatac['bestp_short'].values)
                traindatay_long = to_categorical(
                    traindatac['bestp_long'].values)

                if modellabel == 'LSTM1cls':
                    testdatac['predp_short'] = self.LSTM_1_cls(
                        traindatax_lstm, traindatay_short, testdatax_lstm,
                        timesteps, epochs, batchsize)
                    testdatac['predp_long'] = self.LSTM_1_cls(
                        traindatax_lstm, traindatay_long, testdatax_lstm,
                        timesteps, epochs, batchsize)
                elif modellabel == 'LSTM2cls':
                    testdatac['predp_short'] = self.LSTM_2_cls(
                        traindatax_lstm, traindatay_short, testdatax_lstm,
                        timesteps, epochs, batchsize)
                    testdatac['predp_long'] = self.LSTM_2_cls(
                        traindatax_lstm, traindatay_long, testdatax_lstm,
                        timesteps, epochs, batchsize)
                else:
                    pass

                testdatac['predp_short'] = testdatac['predp_short'].iloc[0]
                testdatac['predp_long'] = testdatac['predp_long'].iloc[0]

                testdatac = testdatac[testdatac.date > dates[i * testlen]]

                hsma0 = pd.concat([hsma0, testdatac], ignore_index=True)

            if i == ntrain:
                hsma = hsma0[['code', 'date', 'predp_short',
                              'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        hsma0[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)
            hsma.to_hdf('Test\\futurekeras\\testresult\\hsma_lstm_cls_code_' +
                        modellabel + '.h5', 'hsma')

        return (hsma)

    def lstm_classification_code1(self, testlen, ntrain, epochs, timesteps,
                                  ncode, day, lr):

        hsmadata_x = self.hsmadata_raw_x(timesteps)
        hsmadata_y = self.hsmadata_bestp2(day, lr)
        hsmadata = pd.merge(hsmadata_y, hsmadata_x)

        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen

        for i in range(ntrain, ntest):
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] <= dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <=
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            ###选取盈利潜力最大的10个品种训练并预测
            traindata[
                'bestp_r'] = traindata['bestp_short_r'] + traindata['bestp_long_r']
            selectcode = traindata.groupby(
                ['code'], as_index=False)[['bestp_r']].mean()
            selectcode.sort_values(by='bestp_r', ascending=False, inplace=True)
            selectcode = pd.DataFrame({'code': selectcode.code.iloc[0:ncode]})
            traindata = pd.merge(traindata, selectcode)
            testdata = pd.merge(testdata, selectcode)

            hsma0 = pd.DataFrame()
            for code in traindata.code.unique():

                traindatac = traindata[traindata.code == code].copy()
                testdatac = testdata[testdata.code == code].copy()

                traindatax = traindatac.drop([
                    'date', 'code', 'bestp_short', 'bestp_short_r',
                    'bestp_long', 'bestp_long_r', 'bestp_r'
                ], 1)
                testdatax = testdatac[traindatax.columns].values
                traindatax = traindatax.values
                traindatax_lstm = self.hsmaseq(
                    traindatax, timesteps, data_dim=6)
                testdatax_lstm = self.hsmaseq(testdatax, timesteps, data_dim=6)
                traindatay_short = to_categorical(
                    traindatac['bestp_short'].values)
                traindatay_long = to_categorical(
                    traindatac['bestp_long'].values)

                model = Sequential()
                model.add(
                    LSTM(
                        64,
                        activation='relu',
                        return_sequences=False,
                        input_shape=(timesteps, traindatax_lstm.shape[2]
                                     )))  #, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                #model.add(LSTM(32, activation='relu', return_sequences=False))  #, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                #model.add(LSTM(16, activation='linear'))#, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                model.add(
                    Dense(traindatay_short.shape[1], activation='softmax'))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

                model.fit(
                    traindatax_lstm,
                    traindatay_short,
                    epochs=epochs,
                    batch_size=2000)
                testdatac['predp_short'] = (
                    model.predict(testdatax_lstm).argmax(axis=1) + 1
                ) * self.minp

                model = Sequential()
                model.add(
                    LSTM(
                        64,
                        activation='relu',
                        return_sequences=False,
                        input_shape=(timesteps, traindatax_lstm.shape[2]
                                     )))  #, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                #model.add(LSTM(32, activation='relu', return_sequences=False))  #, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                #model.add(LSTM(16, activation='linear'))#, dropout_W=0.2, dropout_U=0.2
                #model.add(Dropout(0.5))
                model.add(
                    Dense(traindatay_long.shape[1], activation='softmax'))

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

                model.fit(
                    traindatax_lstm,
                    traindatay_long,
                    epochs=epochs,
                    batch_size=2000)
                testdatac['predp_long'] = (
                    model.predict(testdatax_lstm).argmax(axis=1) + 1
                ) * self.minp

                testdatac.loc[
                    testdatac.code == code, 'predp_short'] = testdatac.loc[
                        testdatac.code == code, 'predp_short'].iloc[0]
                testdatac.loc[testdatac.code == code,
                              'predp_long'] = testdatac.loc[
                                  testdatac.code == code, 'predp_long'].iloc[0]

                testdatac = testdatac[testdatac.date > dates[i * testlen]]

                hsma0 = pd.concat([hsma0, testdatac], ignore_index=True)

            if i == ntrain:
                hsma = hsma0[['code', 'date', 'predp_short',
                              'predp_long']].copy()
            else:
                hsma = pd.concat(
                    [
                        hsma,
                        hsma0[['code', 'date', 'predp_short', 'predp_long']]
                    ],
                    ignore_index=True)
            #hsma = pd.read_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return (hsma)

    def cnn_regression(self, testlen, ntrain, epochs, raw, timesteps, day):

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
            traindata = hsmadata[
                (hsmadata['date'] >= dates[(i - ntrain) * testlen])
                & (hsmadata['date'] < dates[i * testlen - day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i * testlen])
                                & (hsmadata['date'] <
                                   dates[(i + 1) * testlen])].copy()
            print(testdata.date.max())

            x_train, y_train = self.hsmacnnslice(traindata)
            x_test, y_test = self.hsmacnnslice(testdata)

            starttime = time.clock()

            model = Sequential()
            model.add(
                Conv1D(
                    512, 3, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(3))
            # now model.output_shape == (None, 10, 64)
            # add a new conv1d on top
            #model.add(Conv1D(32, 3))
            #model.add(Activation('relu'))
            # now model.output_shape == (None, 10, 32)
            model.add(Flatten())
            model.add(Dense(1))
            model.add(Activation('linear'))

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)

            model.fit(x_train, y_train, epochs=epochs, batch_size=2000)
            testdata['predratio'] = model.predict(x_test)

            endtime = time.clock()
            print("The function run time is : %.03f seconds" %
                  (endtime - starttime))

            if i == ntrain:
                hsma = testdata.copy()
            else:
                hsma = pd.concat([hsma, testdata], ignore_index=True)

            #hsma.to_hdf('Test\\stockkeras\\hsma_lstm_regression_' + self.label + '.h5', 'hsma')

        return (hsma)
