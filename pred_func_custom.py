# -*- coding: utf-8 -*-

""" 
This file contains the necessary functions for the main program predictgas.py

"""

import tensorflow as tf
import pandas as pd

def my_dnnmodel(features,labels,mode,params):

    """ This function defines a DNN model """
    # Define the feature columns i.e. the input layer
    net = tf.feature_column.input_layer(features,params['feature_cols'])
    """
    Construct the hidden layers - activation function is here Rectified Linear Unit 
    more: 
        https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    """
    for units in params['hidden_units']:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu) 
        # change the activation to see what happens for example: relu6, crelu, selu, softplus or dropout (random)
    
    ''' 
    And the output layer - as logits final form will be calculated with some 
    activation operator i.e. tf.nn...
    '''
    preds = tf.layers.dense(net,params['n_out'],activation=None)

    # Compute predictions
    pred_gas = tf.nn.relu(preds)
    #pred_gas = preds
    
    #print(pred_gas)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'Preds': preds,  # Predictions from the network 
                'Abs': tf.abs(preds,name=None), # Absolute values of those predictions
                'Act_pred': pred_gas # With Relu-activation
                }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print('%%%%%-labels')
    print(labels)
    print('%%%%%-predictions')
    print(preds)
    print(pred_gas)
    
    #print('FFFFF')
    #print(tf.rank(preds))
    #print(tf.shape(preds))
    
    # Reshape for evaluation and make 64 point floats
    preds = tf.reshape(preds,shape=(-1,))
    preds = tf.cast(preds,tf.float64)
    pred_gas = tf.cast(pred_gas,tf.float64)
    #print(preds)
    # Computation of loss - using mean squared error
    loss = tf.losses.mean_squared_error(labels=labels, predictions=preds)
    
    # Compute evaluation metric
    meanrelerror = tf.metrics.mean_relative_error(labels=labels, 
                                           predictions=pred_gas, 
                                           normalizer=labels,
                                           name = 'meanrelerr_op'
                                           )
    # Define metrics for evaluation
    metrics = {'meanrelerror': meanrelerror}
    tf.summary.scalar('meanrelerror', meanrelerror[1])
    
    #Evaluation method
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        
    #Train method
    assert mode == tf.estimator.ModeKeys.TRAIN
        # Optimizer for training
    optimizer = tf.train.AdagradOptimizer(
                learning_rate=params['learn_rate']
                )
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

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
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
   # print('%%%%%%%%%%%%%%%%%%%%%%')
   # print(dataset)
    dataset = dataset.shuffle(1000).repeat(count=100).batch(batch_size)
    
    features_result, labels_result = dataset.make_one_shot_iterator().get_next()
    
    #print('%%%%%%%%%%%%%%%%%%%%%%')
    #print(features_result)
    #print('%%%%%%%%%%%%%%%%%%%%%%')
    #print(labels_result)
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
