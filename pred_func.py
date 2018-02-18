# -*- coding: utf-8 -*-

""" 
This file contains the necessary functions for the main program predictgas.py

"""

import tensorflow as tf
import pandas as pd

def read_data(fn_input, fn_output,header_in,header_out,col_in,col_out):
    df_x = pd.read_csv(fn_input, skiprows=2, names=header_in, index_col=['Time'], parse_dates = ['Time'], dayfirst=True)
    df_y = pd.read_csv(fn_output, skiprows=1, names=header_out, index_col=['Time'], parse_dates = ['Time'], dayfirst=True)
    
    # Re-sample both to hourly
    df_x_h = df_x.resample('H').mean()
    df_y_h = df_y.resample('H').sum()
    
    x = df_x_h[col_in] 
    y = df_y_h[col_out]*0.01*39*0.277777 # convert to kWh
    
    # Shift to temperatures to correspond with GMT (data in GMT+1)    
    x = x.shift(periods=-1, freq='H', axis=0)

    ds = pd.concat([x,y],axis=1)
    ds = ds.dropna(axis=0, how='any')
    
    return ds

# Return train and test dataframes    
def train_test_data(ds,col_in,col_out,split):
    # Randomize the data
    ds_rand = ds.sample(frac=1)
    ds_rand.index = range(720)

    ds_train = ds_rand.loc[:split,:]
    ds_test = ds_rand.loc[split:,:]
    
    train_x = ds_train
    # If you want to see your training data uncomment the line below 
    # print(train_x)
    train_y = ds_train[col_out]
    
    # If you want to see your training data uncomment the line below
    # print(train_y)
    test_x = ds_test
    test_y = ds_test[col_out]
    
    return (train_x, train_y), (test_x, test_y)

# Input function for training
def train_tf(features, labels, batch_size):    
    # dict converts train_x into dictionary
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    #print(dataset)
    dataset = dataset.shuffle(1000).repeat(count=100).batch(batch_size)
    
    features_result, labels_result = dataset.make_one_shot_iterator().get_next()
    
    return features_result, labels_result


def evaluate_tf(features, labels=None, batch_size=None): 
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
