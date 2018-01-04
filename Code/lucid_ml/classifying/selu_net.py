#!/usr/bin/env python3
# coding: utf-8
from scipy import sparse

from sklearn.base import BaseEstimator

import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge

# keras imports are only done inside functions and not globally, so that tensorflow is not initialized until those functions are called.
# This makes it possible to call these functions with @processify.

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session, get_session

import tempfile

import time

import math

import os

import re

from sklearn.metrics import f1_score

import random
import string

def create_locally_connected_network(input_size, units, output_size, final_activation, verbose, loss_func = 'categorical_crossentropy', metrics=[], optimizer='adam', inner_local_units_size=50):
    # be nice to other processes that also use gpu memory by not monopolizing it on process start
    # in case of vram memory limitations on large networks it may be helpful to not set allow_growth and grab it all directly
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session, get_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    
    from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, LocallyConnected1D, ZeroPadding1D, Reshape, Flatten
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers.noise import AlphaDropout
    from keras.models import Model

    kinit = "lecun_normal";
    model = Sequential()
    
    model.add(Reshape((input_size, 1), input_shape=(input_size,)))
    
    dropoutRate = 0.1
    
    stride = 500 #this cannot just be changed without fixing padsNeeded to be more general
    padsNeeded = ((math.ceil(input_size / stride) + 1) * stride - input_size - 1) % stride
    model.add(ZeroPadding1D(padding=(0, padsNeeded)))
    model.add(LocallyConnected1D(units[0], 1000, strides=stride, kernel_initializer=kinit))
    model.add(Activation("selu"))
    
    for u_cnt in units[1:]:
        model.add(LocallyConnected1D(u_cnt, inner_local_units_size, strides=1, kernel_initializer=kinit))
        model.add(Activation("selu"))
        model.add(AlphaDropout(dropoutRate))

    model.add(Flatten())

    model.add(Dense(output_size, activation=final_activation, kernel_initializer=kinit))
   
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)
                  
    if verbose:
        model.summary()
    
    return model


def create_locally_connected_head_network(input_size, units, output_sizes, final_activations, verbose, loss_funcs = 'categorical_crossentropy', metrics=[], optimizer='adam'):
    # be nice to other processes that also use gpu memory by not monopolizing it on process start
    # in case of vram memory limitations on large networks it may be helpful to not set allow_growth and grab it all directly
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session, get_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, LocallyConnected1D, ZeroPadding1D, Reshape, Flatten
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers.noise import AlphaDropout
    
    from keras.models import Model
    
    kinit = "lecun_normal";
    head = Sequential()
    
    head.add(Reshape((input_size, 1), input_shape=(input_size,)))
    
    dropoutRate = 0.1
    
    stride = 500 #this cannot just be changed without fixing padsNeeded to be more general
    padsNeeded = ((math.ceil(input_size / stride) + 1) * stride - input_size - 1) % stride
    if verbose: 
        print("Padding of %i zeros will be used to pad input of size %i to size %i" % (padsNeeded, input_size, input_size + padsNeeded))
    head.add(ZeroPadding1D(padding=(0, padsNeeded)))
    head.add(LocallyConnected1D(units[0], 1000, strides=stride, kernel_initializer=kinit))
    head.add(Activation("selu"))
    head.add(Flatten())
    
    for u_cnt in units[1:]:
        head.add(Dense(u_cnt, kernel_initializer=kinit))
        head.add(Activation("selu"))
        head.add(AlphaDropout(dropoutRate))

    inp = Input(shape=(input_size,))
    
    outputs = []
    
    headi = head(inp)
    
    for o, a in zip(output_sizes, final_activations):
        l = headi;
        for oc, ac in zip(o, a):
            l = Dense(oc, activation=ac, kernel_initializer=kinit)(l) # in the original experiments the kernal initializer was not set correctly here. Likely no big change?
        outputs.append(l)

    model = Model(inputs=[inp], outputs=outputs)
    
    model.compile(loss=loss_funcs,
                  optimizer=optimizer,
                  metrics=metrics)
                  
    if verbose:
        model.summary()
    
    return model

def _batch_generator(X, y, batch_size, shuffle):
    sparseFormat = "csr"
    # csr: 119ms
    # csc: 213ms
    # bsr: fail
    # lil: 128ms, double memory of csr
    # dok: fail, extreme memory usage
    # coo: fail
    
    # using float32 tripples speed compared to float64
    X = X.astype("float32").asformat(sparseFormat)
    y = y.astype("float32").asformat(sparseFormat)

    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        
        # I've tried to use toarray(out=...) to not create new blocks of memory all the time
        # the result was no speed increase and training would diverge. ?!
        # float32 alone is what gives a good speed increase without any convergence troubles...
        X_batch = X[batch_index].toarray()
        y_batch = y[batch_index].toarray()
        
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def _batch_generatorp(X, batch_size):
    X = X.astype("float32").asformat("csr")

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

def find_weights_path_by(ending, workdir):
    files = os.listdir(workdir)
    for f in files:
        if (f.endswith(ending)):
            return os.path.join(workdir, f)
    return None
        
def clear_weights(in_directory):
    files = os.listdir(in_directory)
    weights = [f for f in files if f.endswith(".hdf5")]
    bestEpoch = -1
    bestWeightName = None
    
    for w in weights:
        epoch = int(re.match("weights.([0-9]+)-.*", w).group(1))
        if epoch > bestEpoch:
            bestEpoch = epoch
            bestWeightName = w
            
    for w in weights:
        if w != bestWeightName:
            os.remove(os.path.join(in_directory, w))

def sample_f1(model, X_val, y_val):
    print("\nFinding validation sample f1")
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='samples')
    print("Validation sample f1 is %f" % f1)
    return f1

def put_val(d, n, v):
    d[n] = v

def createF1Callback(model, X_val, y_val):
    from keras.callbacks import LambdaCallback
    return LambdaCallback(on_epoch_end=lambda epoch, logs: put_val(logs, "f1_sample", sample_f1(model, X_val, y_val)))

class SeluNet(BaseEstimator):

    def __init__(self, verbose=0, model=None):
        self.verbose = verbose
        self.model = model
        self.thresholds = None
    
    # stupid search for each output separatly. Does not seem to help significantly, i.e. does not generalize
    def search_thresholds(self, X_val, y_val):
        bsize = 128
        pred = self.model.predict_generator(generator=_batch_generatorp(X_val, bsize), steps=np.ceil(X_val.shape[0] / bsize), verbose=self.verbose)
        best = f1_score(y_val, pred > self.thresholds, average="samples")
        
        search_thrs = [0.15, 0.175, 0.19, 0.2, 0.21, 0.225, 0.25, 0.275, 0.3, 0.35, 0.5, 0.9]
        
        for out_idx in np.arange(y_val.shape[1]):
            bestThr = self.thresholds[out_idx]
    
            for try_thr in search_thrs:
                self.thresholds[out_idx] = try_thr
                newF1 = f1_score(y_val, pred > self.thresholds, average='samples')
                if newF1 > best:
                    best = newF1
                    bestThr = try_thr

            print("Setting threshold for label %i to %f for sample_f1 of %f" % (out_idx, bestThr, best))
            self.thresholds[out_idx] = bestThr
            
        return best
        
    def fit(self, X, y):
        if not self.model: 
 
            # to compare the standard settings sgd gets something like
 #           avg_n_labels_gold: 5.236 (+/- 0.000) <> avg_n_labels_pred: 3.125 (+/- 0.000) <> f1_macro: 0.235 (+/- 0.000) <> f1_micro: 0.508 (+/- 0.000) <> f1_samples: 0.480 (+/- 0.000) <> p_macro: 0.320 (+/- 0.000) <> p_micro: 0.679 (+/- 0.000) <> p_samples: 0.657 (+/- 0.000) <> r_macro: 0.205 (+/- 0.000) <> r_micro: 0.405 (+/- 0.000) <> r_samples: 0.422 (+/- 0.000)
#Duration: 9:23:23.939776
            
            # [10] + [2048] * 2, threshold 0.19, ~36 hours
 #           avg_n_labels_gold: 5.240 (+/- 0.015) <> avg_n_labels_pred: 5.410 (+/- 0.245) <> f1_macro: 0.223 (+/- 0.006) <> f1_micro: 0.505 (+/- 0.009) <> f1_samples: 0.508 (+/- 0.005) <> p_macro: 0.245 (+/- 0.012) <> p_micro: 0.497 (+/- 0.017) <> p_samples: 0.555 (+/- 0.008) <> r_macro: 0.231 (+/- 0.005) <> r_micro: 0.513 (+/- 0.009) <> r_samples: 0.525 (+/- 0.009)

            # [3] + [1024] * 2, threshold 0.19, ~21 hours
            #avg_n_labels_gold: 5.240 (+/- 0.022) <> avg_n_labels_pred: 4.629 (+/- 0.133) <> f1_macro: 0.236 (+/- 0.004) <> f1_micro: 0.527 (+/- 0.003) <> f1_samples: 0.508 (+/- 0.004) <> p_macro: 0.281 (+/- 0.004) <> p_micro: 0.562 (+/- 0.010) <> p_samples: 0.568 (+/- 0.008) <> r_macro: 0.224 (+/- 0.005) <> r_micro: 0.497 (+/- 0.007) <> r_samples: 0.509 (+/- 0.008)
            
            # [2] + [512] * 2, threshold 0.19 ~22 hours
            #avg_n_labels_gold: 5.240 (+/- 0.023) <> avg_n_labels_pred: 5.932 (+/- 0.596) <> f1_macro: 0.230 (+/- 0.006) <> f1_micro: 0.500 (+/- 0.013) <> f1_samples: 0.488 (+/- 0.010) <> p_macro: 0.248 (+/- 0.015) <> p_micro: 0.474 (+/- 0.033) <> p_samples: 0.493 (+/- 0.025) <> r_macro: 0.241 (+/- 0.007) <> r_micro: 0.532 (+/- 0.015) <> r_samples: 0.544 (+/- 0.016)

            
            
            layerSizes = [2] + [512] * 2
            
            # this is basically a guess. A more refined way to find the threshold would be nice. Simply searching it however seems to not work, see search_thresholds()
            self.thresholds = [0.19] * y.shape[1]
            # this combination of categorical crossentropy and sigmoids is kinda weird. However the original mlp code of Quadflor also
            # used it and it seems to work, unlike binary cross entropy, which seems to suffer from the fact that the outputs are very sparse(?)
            self.model = create_locally_connected_head_network(X.shape[1], layerSizes, [[y.shape[1]]], 
                loss_funcs=["categorical_crossentropy"], final_activations = [["sigmoid"]], verbose = self.verbose)

            #self.model = create_locally_connected_network(X.shape[1], [2] * 10 + [1], y.shape[1], loss_func="categorical_crossentropy", final_activation="sigmoid", verbose=self.verbose)

        from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

        cachedir = tempfile.gettempdir()
        
        tmpdir = os.path.join(cachedir, ''.join(random.choice(string.ascii_uppercase) for _ in range(7)))

        print("Will use %s as temporary directory. If execution fails this may need to be cleaned up by hand! For very large networks this process may require dozens of gigabytes of temporary storage. TODO: improve..." % tmpdir)
        trainingComplete = False
        try:
            os.mkdir(tmpdir)
            
            vcode = 0
            if self.verbose:
                vcode = 1
        
            checkpointer = ModelCheckpoint(mode="max", monitor='f1_sample', filepath = os.path.join(tmpdir, "weights.{epoch:02d}-{f1_sample:.4f}.hdf5"), 
                                            verbose = vcode, save_best_only = True)
         
            estopper = EarlyStopping(mode="max", monitor='f1_sample', min_delta=0, verbose= vcode, patience = 9);

            lreducer = ReduceLROnPlateau(mode="max", monitor='f1_sample', factor=0.1, verbose = vcode, patience = 6, min_lr = 0.00001);
            
            
            trainSlice = math.floor(X.shape[0] * 0.91)
            
            idxs = np.arange(X.shape[0])
            np.random.shuffle(idxs)
            
            trainIdxs = idxs[:trainSlice]
            valIdxs = idxs[trainSlice:]
            
            f1setter = createF1Callback(self, X[valIdxs], y[valIdxs])
            
            if self.verbose:
                print("Will use %i examples for training and %i samples for validation" % (trainSlice, X.shape[0] - trainSlice))
            
            bsize = 64
            self.model.fit_generator(generator=_batch_generator(X[trainIdxs], y[trainIdxs], bsize, True),
                                     steps_per_epoch=np.ceil(trainSlice / bsize), 
                                     epochs = 99999, # early stopping will take care of it
                                     verbose = self.verbose,
                                     #validation_data=_batch_generator(X[valIdxs], y[valIdxs], bsize, False),
                                     #validation_steps=np.ceil((X.shape[0] - trainSlice)/bsize),
                                     callbacks=[f1setter, checkpointer, lreducer, estopper])
                                     
            trainingComplete = True
             
        finally:      
            clear_weights(tmpdir);
            bestWeights = find_weights_path_by("hdf5", tmpdir);
            if trainingComplete:
                assert bestWeights != None
                self.model.load_weights(bestWeights);
            os.remove(bestWeights);
            os.rmdir(tmpdir)


    def predict(self, X):
        bsize = 128
        pred = self.model.predict_generator(generator=_batch_generatorp(X, bsize), steps=np.ceil(X.shape[0] / bsize), verbose=self.verbose)
        return sparse.csr_matrix(pred > self.thresholds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
