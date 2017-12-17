#!/usr/bin/env python3
# coding: utf-8
from scipy import sparse

from sklearn.base import BaseEstimator
from keras.layers import Dense, Activation, Dropout, BatchNormalization
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
    return lr;

class SeluNet(BaseEstimator):
    def __init__(self, verbose=0, model=None, final_activation='sigmoid'):
        self.verbose = verbose
        self.model = model
        self.final_activation = final_activation
            
    def fit(self, X, y):
        if not self.model: 
            self.model = create_network(X.shape[1], [2048] + [1024] * 2, y.shape[1],
                final_activation = self.final_activation, verbose = self.verbose)
        
        bsize = 256
        self.model.fit_generator(generator=_batch_generator(X, y, bsize, True),
                                 samples_per_epoch=np.ceil(X.shape[0] / bsize), nb_epoch = 100, verbose = self.verbose, callbacks=[LearningRateScheduler(_epoch_lr)])

    def predict(self, X):
        pred = self.predict_proba(X)
        return sparse.csr_matrix(pred > 0.142)

    def predict_proba(self, X):
        bsize = 512
        pred = self.model.predict_generator(generator=_batch_generatorp(X, bsize), val_samples=np.ceil(X.shape[0] / bsize), verbose=self.verbose)
        return pred


if __name__ == "__main__":
    import doctest
    doctest.testmod()
