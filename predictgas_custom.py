# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:48:33 2018

@author: cvrse
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import os
#import pandas as pd
#import random

from pred_func_custom import *
#import functions from pred_func.py to this file

def main():
 
## Data format - define headers etc. ##
    header_in = ['Time', 'Power', 'AirTemp', 'RH', 'Pressure', 'Total', 'Diffuse', 
           'Total', 'Diffuse', 'Temp', 'Speed',
          'Dir', 'Rain', 'Voltage', 'SolarTotal', 'Solar Diffuse', 
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
    col_in = ['AirTemp', 'SolarTotal', 'Speed']
    col_out = 'Pulse'
    split = 600
    
    # Parameters
    learn_rate = 0.1
    l1reg_rate = 0
    l2reg_rate = 0
    batch_size = 5

## Not needed, yet... ##
 #   training_epochs = 1000
 #   display_step = 50

    # Run the defined functions to get the data in tensor format
    ds = read_data(fn_in, fn_out,header_in, header_out, col_in, col_out) # create data-set
    (train_x, train_y), (test_x, test_y) = train_test_data(ds,col_in,col_out,split) #create test and train data
    train_feats, train_labels = train_tf(train_x, train_y, batch_size) # construct the tensors for training
    test_feats, test_labels = evaluate_tf(test_x, test_y, batch_size)
    print(train_feats)
    print(train_labels)
    feature_columns = [
            tf.feature_column.numeric_column(key='AirTemp'),
            tf.feature_column.numeric_column(key='SolarTotal'),
            tf.feature_column.numeric_column(key='Speed')
            ]
    
    print('---------------Feature columns-----------------')
    print(feature_columns)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    
    # Define the model
    predictor = tf.estimator.Estimator(
            model_fn=my_dnnmodel,
            config=tf.estimator.RunConfig(model_dir='./dnn_model_mulcols/', save_summary_steps=100),
            params={
                'feature_cols': feature_columns,
                'hidden_units': [80,40,20],
                'learn_rate': learn_rate,
                'l1reg_rate': l1reg_rate,
                'l2reg_rate': l2reg_rate,
                'n_out': 1 #output amounts
                })    
    
    # Train the model
    predictor.train(
            input_fn=lambda:train_tf(train_x,train_y,batch_size)
            )
    
    # Evaluate the model
    eval_result = predictor.evaluate(
            input_fn=lambda:evaluate_tf(test_x,test_y,batch_size)
            )

    
    # Evaluate the models - print the outcomes
    print('--------%%%%%%%%%%%%%%%%%---------')
    print('--------Model evaluation results---------') 
    print(eval_result)
    
    # Make predictions
    print('------%%%%%-------')
    print('---Evaluations as predictions----')
    predictions = predictor.predict(input_fn = lambda:evaluate_tf(test_x, batch_size=batch_size))
    
    print(predictions)
    
    preds_arr = []
    pred_gas_arr = []
    for pred_dict in zip(predictions):
        preds_arr = np.append(preds_arr,pred_dict[0]['Abs'])
        pred_gas_arr = np.append(pred_gas_arr,pred_dict[0]['Act_pred'])
   
    # Print arrays for checks
    #print(preds_arr)
    #print(pred_gas_arr)
    
    print('%%%%%%%%%----Plot Predictions----%%%%%%')
    plt.figure(figsize=(11.69,8.27))
    plot_dnn = plt.plot(test_x['AirTemp'], pred_gas_arr, 'x', label="dnn") 
    plot_meas = plt.plot(test_x['AirTemp'], test_y,'x', label="measured")
    plt.legend()
    plt.xlabel("Temperature [C]")
    plt.ylabel("Gas consumption [kWh]")
    plt.title("Temperature and gas consumption")
    plt.savefig("predictions_temp")
    print('--------%%%%%%%%%%%%%%%%%---------')
    
    print('%%%%%%%%%----Plot Predictions----%%%%%%')
    plt.figure(figsize=(11.69,8.27))
    plot_dnn = plt.plot(test_x['Speed'], pred_gas_arr, 'x', label="dnn") 
    plot_meas = plt.plot(test_x['Speed'], test_y,'x', label="measured")
    plt.legend()
    plt.xlabel("Wind Speed @ 10 m [m/s]")
    plt.ylabel("Gas consumption [kWh]")
    plt.title("Temperature and gas consumption")
    plt.savefig("predictions_windspeed")
    print('--------%%%%%%%%%%%%%%%%%---------')
    
    print('%%%%%%%%%----Plot Predictions----%%%%%%')
    plt.figure(figsize=(11.69,8.27))
    plot_dnn = plt.plot(test_x['SolarTotal'], pred_gas_arr, 'x', label="dnn") 
    plot_meas = plt.plot(test_x['SolarTotal'], test_y,'x', label="measured")
    plt.legend()
    plt.xlabel("Solar Radiation [W/m2]")
    plt.ylabel("Gas consumption [kWh]")
    plt.title("Temperature and gas consumption")
    plt.savefig("predictions_solar")
    print('--------%%%%%%%%%%%%%%%%%---------')

    return []  
    
    
        # Bucketized column, maybe later...
        #feature_columns = tf.feature_column.bucketized_column(
        #source_column = feature_columns,
        #boundaries = [0, 5, 10, 15, 20, 25, 30])   


main()

    