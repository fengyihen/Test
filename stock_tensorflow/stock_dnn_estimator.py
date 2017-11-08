# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016

@author: Yizhen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys;
sys.path.append("Test")
from Stock import Stock
import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import talib
import matplotlib.pyplot as plt
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


class StockDnnEstimator(Stock):
    
   
    def dnnclassifier_estimator(self, testlen, ntrain, r, hidden_units, steps):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')
            traindatay = (traindata['ratio'] > r).astype(int).values           
            testdatax = testdata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')    

            # Specify that all features have real-value data
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=traindatax.shape[1])]
            # Build 3 layer DNN with 10, 20, 10 units respectively.
            classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                        hidden_units=hidden_units,
                                                        n_classes=2,
                                                        model_dir=None)#"/tmp/stock_model")
            # Fit model.
            classifier.fit(x=traindatax, y=traindatay, steps=steps)
            testdata['predratio'] = list(classifier.predict(testdatax))
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
 
    def dnnregressor_estimator(self, testlen, ntrain, hidden_units, steps):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        store = pd.HDFStore('Test\\stock_tensorflow\\hsma_' + self.label + '.h5')
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        
            print(testdata.date.max())
            
            traindatax = traindata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')
            traindatay = traindata['ratio'].values.astype('float32')           
            testdatax = testdata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')    

            # Specify that all features have real-value data
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=traindatax.shape[1])]
            # Build 3 layer DNN with 10, 20, 10 units respectively.
            regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                                        hidden_units=hidden_units,
                                                        dropout = 0.5)
            # Fit model.
            regressor.fit(x=traindatax, y=traindatay, steps=steps)
            testdata['predratio'] = list(regressor.predict(testdatax))
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)
            store['hsma'] = hsma
            
        store.close()
        return(hsma)

    def widendeepclassifier_estimator(self, testlen, ntrain, r, hidden_units, steps):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')
            traindatay = (traindata['ratio'] > r).astype(int).values     
            traindatay = traindata['ratio'].values.astype('float32')
            testdatax = testdata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')    

            # Specify that all features have real-value data
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=traindatax.shape[1])]
            # Build 3 layer DNN with 10, 20, 10 units respectively.
            estimator = tf.contrib.learn.DNNLinearCombinedClassifier(linear_feature_columns=feature_columns,
                                                                      dnn_feature_columns=feature_columns,
                                                                      dnn_hidden_units=hidden_units)
                               
            # Fit model.
            estimator.fit(x=traindatax, y=traindatay, steps=steps)
            testdata['predratio'] = list(estimator.predict(testdatax))
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)

    def widendeepregressor_estimator(self, testlen, ntrain, hidden_units, steps):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')
            traindatay = traindata['ratio'].values.astype('float32')
            testdatax = testdata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')    

            # Specify that all features have real-value data
            feature_columns = [tf.contrib.layers.real_valued_column("", dimension=traindatax.shape[1])]
            # Build 3 layer DNN with 10, 20, 10 units respectively.
            estimator = tf.contrib.learn.DNNLinearCombinedRegressor(linear_feature_columns=feature_columns,
                                                                      dnn_feature_columns=feature_columns,
                                                                      dnn_hidden_units=hidden_units)
                            
            # Fit model.
            estimator.fit(x=traindatax, y=traindatay, steps=steps)
            testdata['predratio'] = list(estimator.predict(testdatax))
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        
    def dnnclassifier(self, testlen, ntrain, r, hidden_units, steps):
        
        hsmadata = self.hsmadata
        dates = pd.Series(hsmadata['date'].unique()).sort_values()
        dates.index = range(0, len(dates))
        ntest = len(dates) // testlen       
        
        hsma = pd.DataFrame()
        for i in range(ntrain, ntest):
            traindata = hsmadata[(hsmadata['date'] >= dates[(i-ntrain)*testlen]) & (hsmadata['date'] < dates[i*testlen - self.day])].copy()
            testdata = hsmadata[(hsmadata['date'] >= dates[i*testlen]) & (hsmadata['date'] < dates[(i+1)*testlen])].copy()        

            traindatax = traindata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')
            traindatay = (traindata['ratio'] > r).values.astype('float32').reshape([traindata.shape[0], 1])    
            traindatay = np.c_[traindatay, 1-traindatay]
            testdatax = testdata.drop(['date', 'code', 'ratio'], 1).values.astype('float32')    
            testdatay = (testdata['ratio'] > r).values.astype('float32').reshape([testdata.shape[0], 1])           
            testdatay = np.c_[testdatay, 1-testdatay]
            
            # Create the model
            x = tf.placeholder(tf.float32, [None, traindatax.shape[1]])
            W = tf.Variable(tf.zeros([traindatax.shape[1], 2]))
            b = tf.Variable(tf.zeros([2]))
            y = tf.matmul(x, W) + b
            
            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 2])
            
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            
            # Train
            for _ in range(1000):
                #batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: traindatax, y_: traindatay})
                
            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: testdatax, y_: testdatay}))
    
            testdata['predratio'] = list(classifier.predict(testdatax))
            
            hsma = pd.concat([hsma, testdata], ignore_index = True)

        return(hsma)
        

        
        
        