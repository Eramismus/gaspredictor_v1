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
    
    fn_in = 'weatherdata.csv'
    fn_out = 'gasdata.csv'
    col_in = 'AirTemp'
    col_out = 'Pulse'
    split = 360
    
    # Parameters - try changing these and see what happens
    learning_rate = 0.1
    batch_size = 5
    
    # Determine if linear or dnn models are trained 0=no training, 1=training
    # Note: training data is shuffled and model data updated everytime training is conducted
    train_lin = 1
    train_dnn = 1

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
        
    """Build a linear regression model. You can test around with these."""
    predictor_lin = tf.estimator.LinearRegressor(
         config=tf.estimator.RunConfig(model_dir='./linear_model/', save_summary_steps=100),
         feature_columns=feature_columns)
    
    """"
    Build 2 hidden layer DNN with 3, 1 units respectively. Test around with the number of nodes and see what happens.
    Is it possible to find an optimum? Also change the linear regularization strengths and see what happens.
    """
    predictor_dnn = tf.estimator.DNNRegressor(
        config=tf.estimator.RunConfig(model_dir='./dnn_model/', save_summary_steps=100),
        feature_columns = feature_columns,
        hidden_units=[3,3],
        optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=learning_rate, 
                l1_regularization_strength=1.0,
                l2_regularization_strength=0.1
            ))
    
    """"Note: the model data is pushed to folders linear_model and dnn_model."""
    
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
    print('--------Linear model evaluation---------')  
    eval_result_lin = predictor_lin.evaluate(input_fn =lambda:evaluate_tf(test_x, test_y, batch_size)) 
    print(eval_result_lin)
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------DNN model evaluation---------')    
    eval_result_dnn = predictor_dnn.evaluate(input_fn =lambda:evaluate_tf(test_x, test_y, batch_size)) 
    print(eval_result_dnn)
    print('--------%%%%%%%%%%%%%%%%%---------')


    """The predictions"""
    """Make some predictions either by using some random values or using the evaluation data"""
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
    
    
    """ Plot and print predictions"""
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------Prediction input---------')     
    print(predict_x['AirTemp'])
    print('--------%%%%%%%%%%%%%%%%%---------')
    
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------DNN model predictions---------')  
    print(pred_arr_dnn)
    print('--------%%%%%%%%%%%%%%%%%---------')
    
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------Linear model predictions---------')      
    print(pred_arr_lin)
    print('--------%%%%%%%%%%%%%%%%%---------')
    
    print('%%%%%%%%%----Plot Predictions----%%%%%%')
    plt.figure(figsize=(11.69,8.27))
    plot_lin = plt.plot(predict_x['AirTemp'], pred_arr_lin, 'x', label="linear") 
    plot_dnn = plt.plot(predict_x['AirTemp'], pred_arr_dnn, 'x', label="dnn") 
    plot_meas = plt.plot(predict_x['AirTemp'], test_y,'x', label="measured")
    plt.legend()
    plt.xlabel("Temperature [C]")
    plt.ylabel("Gas consumption [MWh]")
    plt.title("Temperature and gas consumption")
    plt.savefig("predictions")
    print('--------%%%%%%%%%%%%%%%%%---------')    
    
    return[]    
    
main()
    