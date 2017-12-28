#!/usr/bin/env python3
# coding: utf-8
from scipy import sparse

from sklearn.base import BaseEstimator
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LocallyConnected1D, ZeroPadding1D, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers.noise import AlphaDropout

from keras.callbacks import LearningRateScheduler

import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import math

def create_network(input_size, units, output_size, 
                   activation='selu',
                   dropout=AlphaDropout,
                   dropout_rate=0.1,
                   kernel_initializer='lecun_normal',
                   optimizer='adam',
                   final_activation='softmax',
                   loss_func = 'categorical_crossentropy',
                   metrics=[],
                   verbose=False):
    """Generic function to create a fully-connected neural network. Defaults are fitting for Selu activated network using Adam.
    # Arguments
        units: list of int. Number of units in the layers
        dropout: keras.layers.Layer. A dropout layer to apply.
        dropout_rate: 0 <= float <= 1. The rate of dropout.
        kernel_initializer: str. The initializer for the weights.
        optimizer: str/keras.optimizers.Optimizer. The optimizer to use.
        output_size: int > 0. The size of the output vector
        input_size: int > 0. The size of the input vector
        final_activation: the activation applied to the output
        metrics: list of metrics to use
    # Returns
        A Keras model instance (compiled).
    """
    model = Sequential()
    model.add(Dense(units[0], input_shape=(input_size,),
                    kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    for u_cnt in units[1:]:
        model.add(Dense(u_cnt, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(dropout(dropout_rate))
        
    model.add(Dense(output_size))
    model.add(Activation(final_activation))
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)
                  
    if verbose:
        model.summary()
    
    return model

def create_locally_connected_head_network(input_size, units, output_size, final_activation, verbose, loss_func = 'categorical_crossentropy', metrics=[], optimizer='adam'):
    kinit = "lecun_normal";
    model = Sequential()
    
    model.add(Reshape((input_size, 1), input_shape=(input_size,)))
    
    stride = 500 #this cannot just be changed without fixing padsNeeded to be more general
    padsNeeded = ((math.ceil(input_size / stride) + 1) * stride - input_size - 1) % stride
    if verbose: 
        print("Padding of %i zeros will be used to pad input of size %i to size %i" % (padsNeeded, input_size, input_size + padsNeeded))
    model.add(ZeroPadding1D(padding=(0, padsNeeded)))
    model.add(LocallyConnected1D(units[0], 1000, strides=stride, kernel_initializer=kinit))
    model.add(Activation("selu"))
    
    model.add(Flatten())
    
    for u_cnt in units[1:]:
        model.add(Dense(u_cnt, kernel_initializer=kinit))
        model.add(Activation("selu"))
        model.add(AlphaDropout(0.1))
        
    model.add(Dense(output_size))
    model.add(Activation(final_activation))
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)
                  
    if verbose:
        model.summary()
    
    return model
    

def _batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index].toarray()
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def _batch_generatorp(X, batch_size):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if counter == number_of_batches:
            counter = 0

def _epoch_lr(epoch):
    lr = 0.001;
    if epoch > 50:
        lr = 0.0001;
    return lr;

class SeluNet(BaseEstimator):
    def __init__(self, verbose=0, model=None, final_activation='sigmoid'):
        self.verbose = verbose
        self.model = model
        self.final_activation = final_activation
            
    def fit(self, X, y):
        if not self.model: 
            # sample micro f1 scores:
            # normal network:
            # std thr is 0.142
            # on sample title pretty good: [2048] + [1024] * 2
            # on sample full pretty good:
            # bsize: 256
            # [512] + [1024] * 2: 0.355
            # [512] * 3: 0.340
            # bsize: 64, 100 epochs
            # [700] + [1024] * 3 (lr 0.001): 0.284
            # [700] + [1024] * 3 (lr 0.001 + 0.0001 @ 80): 0.231
            # [700] + [1024] * 3 (lr 0.001 + 0.0001 @ 80, thr 0.12): 0.303
            # [700] + [1024] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.12): 0.355
            # [700] + [1400] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.125): 0.363
            # [700] + [2048] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.125): 0.34
            
            # local head network:
            # [100] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.125): 0.328
            # [100] + [512] * 5 (lr 0.001 + 0.0001 @ 80, thr, 0.125): 
            # [100] + [1024] * 5 (lr 0.001 + 0.0001 @ 80, thr, 0.125):
            # [25] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.125): 0.325
            # [25] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.15): 0.331
            # [25] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.175): 0.343
            # [25] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.215): 0.342
            # [25] + [1024] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.125):
            # [10] + [1024] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.125):
            # [10] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.1): 0.348
            # [10] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.065): 0.332
            # [10] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.085): 0.319
            
            # [5] + [1024] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.175): 0.356/0.343
            # [10] + [1024] * 1 (lr 0.001 + 0.0001 @ 80, thr, 0.175): 0.298
            # [5] + [2048] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.175): 0.355
            # [5] + [4096] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.185): 0.352

            # [10] + [1500] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.35
            
            # [10] + [1500] + [1000] (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.355
            
            # [10] + [1500] + [750] + [500] (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.318
            # [3] + [1000] + [750] + [500] (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.291
            
            # [3] + [1000] + [750] + [500] (lr 0.001 + 0.0001 @ 80, thr, 0.11): 0.327
            
            # 80 epochs
            # [3] + [1000] + [750] (lr 0.001 + 0.0001 @ 60, thr, 0.11): 0.322
            
            # 60 epochs
            # [5] + [1500] + [1000] (lr 0.001 + 0.0001 @ 50, thr, 0.21): 0.321
            
            # 120 epochs
            # [5] + [4096] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.365

            # 200 epochs, 128 bsize
            # [5] + [8192] * 2 (lr 0.001 + 0.0001 @ 80, thr, 0.19): 0.34
            
            # bsize 32, 150 epochs
            # [5] + [2048] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.185): fail
            
            #bsize 128
            # [10] + [512] * 3 (lr 0.001 + 0.0001 @ 80, thr, 0.05): 0.287
            
            self.model = create_locally_connected_head_network(X.shape[1], [5] + [1500] + [1000], y.shape[1],
                final_activation = self.final_activation, verbose = self.verbose)
        
        bsize = 64
        self.model.fit_generator(generator=_batch_generator(X, y, bsize, True),
                                 samples_per_epoch=np.ceil(X.shape[0] / bsize), nb_epoch = 60, verbose = self.verbose, callbacks=[LearningRateScheduler(_epoch_lr)])

    def predict(self, X):
        pred = self.predict_proba(X)
        return sparse.csr_matrix(pred > 0.21)

    def predict_proba(self, X):
        bsize = 128
        pred = self.model.predict_generator(generator=_batch_generatorp(X, bsize), val_samples=np.ceil(X.shape[0] / bsize), verbose=self.verbose)
        return pred


if __name__ == "__main__":
    import doctest
    doctest.testmod()
