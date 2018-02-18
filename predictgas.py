# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:20:47 2018

@author: cvrse
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import os
#import pandas as pd
#import random

from pred_func import *
#import functions from estimator.py to this file

def main():
 
## Data format - define headers etc. ##
    header_in = ['Time', 'Power', 'AirTemp', 'RH', 'Pressure', 'Total', 'Diffuse', 
           'Total', 'Diffuse', 'Temp', 'Speed',
          'Dir', 'Rain', 'Voltage', 'Solar Total', 'Solar Diffuse', 
          'Total','Diffuse', 'Rain', 'Dir.Avg2', 'Dir.Std2', 'WindClass2.0', 'WindClass20', 
          'WindClass20', 'WindClass20', 'WindClass20', 'WindClass20', 'WindClass245', 'WindClass245',
          'WindClass245', 'WindClass245', 'WindClass245', 'WindClass245', 'WindClass290', 'WindClass290',
          'WindClass290', 'WindClass290', 'WindClass290', 'WindClass290', 'WindClass2135', 'WindClass2135',
          'WindClass2135', 'WindClass2135', 'WindClass2135', 'WindClass2135', 'WindClass2180', 'WindClass2180',
          'WindClass2180', 'WindClass2180', 'WindClass2180', 'WindClass2180', 'WindClass2225',
          'WindClass2225', 'WindClass2225', 'WindClass2225', 'WindClass2225', 'WindClass2225', 
          'WindClass2270', 'WindClass2270', 'WindClass2270', 'WindClass2270', 'WindClass2270',
          'WindClass2270', 'WindClass2315', 'WindClass2315', 'WindClass2315', 'WindClass2315',
          'WindClass2315', 'WindClass2315']
    
    header_out = ['Time', 'Pulse']
    
    fn_in = 'lboro_2017_11_hourly-gmt+1.csv'
    fn_out = 'EnicaLogger-211_NOV17_Gasdata-311017.csv'
    col_in = 'AirTemp'
    col_out = 'Pulse'
    split = 360
    
    # Parameters
    learning_rate = 0.01
    batch_size = 5
    train_lin = 1
    train_dnn = 1

## Not needed, yet... ##
 #   training_epochs = 1000
 #   display_step = 50

    # Run the defined functions to get the data in tensor format
    ds = read_data(fn_in, fn_out,header_in, header_out, col_in, col_out) # create data-set
    (train_x, train_y), (test_x, test_y) = train_test_data(ds,col_in,col_out,split) #create test and train data
    train_feats, train_labels = train_tf(train_x, train_y, batch_size) # construct the tensors for training
    test_feats, test_labels = evaluate_tf(test_x, test_y, batch_size)
    
    feature_columns = [
            tf.feature_column.numeric_column(key='AirTemp')
            ]
    
    print('---------------Feature columns-----------------')
    print(feature_columns)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # Bucketized column, maybe later...
        #feature_columns = tf.feature_column.bucketized_column(
        #source_column = feature_columns,
        #boundaries = [0, 5, 10, 15, 20, 25, 30])
        
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    predictor_lin = tf.estimator.LinearRegressor(
         config=tf.estimator.RunConfig(model_dir='./linear_model/', save_summary_steps=100),
         feature_columns=feature_columns)
    
    predictor_dnn = tf.estimator.DNNRegressor(
        config=tf.estimator.RunConfig(model_dir='./dnn_model/', save_summary_steps=100),
        feature_columns = feature_columns,
        hidden_units=[3,1],
        optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=learning_rate, 
                l1_regularization_strength=0.01
            ))
    
    # train the models
    if train_lin == 1:
        print('%%% Training linear model %%%%')
        predictor_lin.train(input_fn =lambda:train_tf(train_x,train_y,batch_size))
    else:
        print('%%% Linear Model not trained %%%%')
    
    if train_dnn == 1:
        print('%%% Training DNN %%%%')
        predictor_dnn.train(input_fn =lambda:train_tf(train_x,train_y,batch_size))
    else:
        print('%%% DNN not trained %%%%')
    
    # Evaluate the models - print the outcomes
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------Linear models---------')  
    eval_result_lin = predictor_lin.evaluate(input_fn =lambda:evaluate_tf(test_x, test_y, batch_size)) 
    print(eval_result_lin)
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------DNN models---------')    
    eval_result_dnn = predictor_dnn.evaluate(input_fn =lambda:evaluate_tf(test_x, test_y, batch_size)) 
    print(eval_result_dnn)
    print('--------%%%%%%%%%%%%%%%%%---------')

    # Make some predictions
    #predict_x = {'AirTemp': [10, 0, -5, 20, 25, -20],
     #            }
    predict_x = test_x
    #predict_y = {'Pulse': [0, 0, 0, 0, 0, 0]}
    
    predictions_lin = predictor_lin.predict(input_fn = lambda:evaluate_tf(predict_x, batch_size=batch_size))
    predictions_dnn = predictor_dnn.predict(input_fn = lambda:evaluate_tf(predict_x, batch_size=batch_size))
    


    
    pred_arr_lin = []
    pred_arr_dnn = []
    for pred_dict_lin in zip(predictions_lin):
        pred_arr_lin = np.append(pred_arr_lin,pred_dict_lin[0]['predictions'])
        
        
    for pred_dict_dnn in zip(predictions_dnn):
        pred_arr_dnn = np.append(pred_arr_dnn,pred_dict_dnn[0]['predictions'])
    
    
    print(pred_arr_dnn)
    print(pred_arr_lin)
    print(predict_x['AirTemp'])
    
    print('%%%%%%%%%----Plot Predictions----%%%%%%')
    plt.plot(predict_x['AirTemp'], pred_arr_lin, 'x', predict_x['AirTemp'], pred_arr_dnn, 'x', predict_x['AirTemp'], test_y,'x')
        
    return[]    
    
main()
    