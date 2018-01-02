#!/usr/bin/env python3
# coding: utf-8
from scipy import sparse

from sklearn.base import BaseEstimator

import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session, get_session

import time

import math

import os

import re

from sklearn.metrics import f1_score

import random
import string

def create_locally_connected_head_network(input_size, units, output_sizes, final_activations, verbose, loss_funcs = 'categorical_crossentropy', metrics=[], optimizer='adam'):
    # be nice to other processes that also use gpu memory by not monopolizing it on process start
    # on the sample data set only ~700mb of vram is needed this way, instead of all of it
    # in case of vram memory limitations on large networks it may be helpful to not set allow_growth and grab it all directly
#    import tensorflow as tf
#    from keras.backend.tensorflow_backend import set_session, get_session
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    set_session(tf.Session(config=config))

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
            l = Dense(oc, activation=ac)(l)
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
        
        #y_counts = np.sum(y_batch, axis=1, keepdims=True)
        
        #y_batch /= y_counts
        
        counter += 1
        yield X_batch, y_batch #[y_batch, y_counts]
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

def _epoch_lr(epoch):
    lr = 0.0005;
    if epoch > 40:
        lr = 0.00005;
    return lr;

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
            
            
            # complete micro f1 scores:
            
            # [5] + [1500] + [1000] ????, 0.456
            
            #bsize 64, 120 epochs
            # [10] + [4096] * 2 (lr 0.0005 + 0.00005 @ 80, thr, 0.19): 0.361
            
            #bsize 64, 65 epochs
            # [5] + [1024] * 4 (lr 0.0005 + 0.00005 @ 40, thr, 0.19): 0.483
            # [5] + [1024] * 2 (lr 0.0005 + 0.00005 @ 40, thr, 0.19): 0.527
            
            #bsize 128, 65 epochs
            # [10] + [2048] * 2 (lr 0.0005 + 0.00005 @ 40, thr, 0.19): 0.54 (cv: ...)
            
            # to compare the standard settings sgd gets something like
 #           avg_n_labels_gold: 5.236 (+/- 0.000) <> avg_n_labels_pred: 3.125 (+/- 0.000) <> f1_macro: 0.235 (+/- 0.000) <> f1_micro: 0.508 (+/- 0.000) <> f1_samples: 0.480 (+/- 0.000) <> #p_macro: 0.320 (+/- 0.000) <> p_micro: 0.679 (+/- 0.000) <> p_samples: 0.657 (+/- 0.000) <> r_macro: 0.205 (+/- 0.000) <> r_micro: 0.405 (+/- 0.000) <> r_samples: 0.422 (+/- 0.000)
#Duration: 9:23:23.939776

            
            # [3] + [2048], [[1024, y.shape[1]], [1024, 1] 0.472
            
            
            # TODO threshold per class search instead of global guess?
            # TODO the combination of sigmoid with cross entropy really should not make any sense
            # figure out either a way to use softmax and kl divergence (how should the threshold work in that case?) or fix binary cross entropy
            # or improve softmax + number of labels approach somehow...
            
            # [10] + [2048] * 2: 
 #           avg_n_labels_gold: 5.240 (+/- 0.015) <> avg_n_labels_pred: 5.410 (+/- 0.245) <> f1_macro: 0.223 (+/- 0.006) <> f1_micro: 0.505 (+/- 0.009) <> f1_samples: 0.508 (+/- 0.005) <> #p_macro: 0.245 (+/- 0.012) <> p_micro: 0.497 (+/- 0.017) <> p_samples: 0.555 (+/- 0.008) <> r_macro: 0.231 (+/- 0.005) <> r_micro: 0.513 (+/- 0.009) <> r_samples: 0.525 (+/- 0.009)

            
            self.thresholds = [0.19] * y.shape[1]
            self.model = create_locally_connected_head_network(X.shape[1], [10] + [2048] * 2, [[y.shape[1]]], 
                loss_funcs=["categorical_crossentropy"], final_activations = [["sigmoid"]], verbose = self.verbose)
            
            #self.model = create_locally_connected_head_network(X.shape[1], [5] + [1024] * 2, [[y.shape[1]], [1]],
            #    final_activations = [["softmax"], ["selu", "linear"]], loss_funcs=["categorical_crossentropy", "mse"], verbose = self.verbose)
        
        from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

        cachedir = "/MegaKeks/Dokumente/StudiGits/Quadflor/cache/"
        
        tmpdir = os.path.join(cachedir, ''.join(random.choice(string.ascii_uppercase) for _ in range(7)))
        
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
        
        bsize = 128
        self.model.fit_generator(generator=_batch_generator(X[trainIdxs], y[trainIdxs], bsize, True),
                                 steps_per_epoch=np.ceil(trainSlice / bsize), 
                                 epochs = 99999, # early stopping will take care of it
                                 verbose = self.verbose,
                                 #validation_data=_batch_generator(X[valIdxs], y[valIdxs], bsize, False),
                                 #validation_steps=np.ceil((X.shape[0] - trainSlice)/bsize),
                                 callbacks=[f1setter, checkpointer, lreducer, estopper])
                                 
        clear_weights(tmpdir);
        bestWeights = find_weights_path_by("hdf5", tmpdir);
        assert bestWeights != None
        self.model.load_weights(bestWeights);
        os.remove(bestWeights);
        os.rmdir(tmpdir)
        
        #if self.verbose:
        #    print("Will commence search for the best thresholds!")
        
        #self.search_thresholds(X[valIdxs], y[valIdxs])
        

    def predict(self, X):
        bsize = 128
        pred = self.model.predict_generator(generator=_batch_generatorp(X, bsize), steps=np.ceil(X.shape[0] / bsize), verbose=self.verbose)
        
        return sparse.csr_matrix(pred > self.thresholds)
        
#        vals = pred[0]
#        counts = pred[1]
        
#        result = np.zeros_like(vals)
        
#        for li in range(vals.shape[0]):
#            rval = [x for x in enumerate(vals[li])]
#            count = int(round(counts[li][0]))
#            if count < 1:
#                count = 1
#            bestIdxs = [a[0] for a in sorted(rval, key=lambda x: x[1], reverse=True)[:count]]
#            result[li,bestIdxs] = 1
        
#        return sparse.csr_matrix(result > 0)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
