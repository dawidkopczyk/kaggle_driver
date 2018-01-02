"""
@author: dawidkopczyk
Script that contains deep neural networks utilities for Kaggle Porto Competition
"""

from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import gini_normalized
from sklearn.model_selection import StratifiedKFold

_EPSILON = K.epsilon()

# Create model   
def create_model(features, learning_rate=0.001, neurons=16, activation='relu', init_mode='glorot_uniform', optimizer='Adam', dropout=0.0):
    
    # Define model 
    model = Sequential([
                Dense(neurons, input_shape=(features, ), kernel_initializer=init_mode, activation=activation),
                Dropout(dropout),
                Dense(neurons, kernel_initializer=init_mode, activation=activation),
                Dropout(dropout),
                Dense(neurons, kernel_initializer=init_mode, activation=activation),
                Dropout(dropout),
                Dense(2, activation='softmax'),
            ])
    #model.summary()
    
    # Compile model
    model.compile(loss=fbeta_bce_loss, optimizer=optimizer, metrics=['acc'])
    return model

# Define default score of Keras estimator
def default_score(estimator, X, y):
    return estimator.score(X, y)

# Overwrite incorrect StratifiedKFold in Keras estimator 
class StratifiedKFoldNN(StratifiedKFold):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedKFoldNN, self).__init__(n_splits, shuffle, random_state)
    def split(self, X, y, groups=None):
        return super(StratifiedKFold, self).split(X, y[:,1], groups)
    
# Create Keras callback to produce gini score at the end of epoch  
class gini_callback(keras.callbacks.Callback):  
    def on_train_begin(self, logs={}):
        self.params['metrics'].append('val_gini')
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):                    
        y_pred_prob_val = self.model.predict_proba(self.validation_data[0], verbose=0)[:,1]
        logs['val_gini'] = gini_normalized(self.validation_data[1][:,1], y_pred_prob_val)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return   
    
class loadbest_callback(keras.callbacks.Callback):
    def __init__(self, filename, verbose=0):
        self.filename = filename
        self.verbose = verbose
        
    def on_train_end(self, logs={}):
        self.model.load_weights(self.filename)
        if self.verbose > 0:
            print('\nBest model from training has been loaded.')
        return
    
def get_callbacks(filepath):
    return [gini_callback(), 
            EarlyStopping(monitor='val_gini', patience=1, mode='max'),
            ModelCheckpoint(filepath, monitor='val_gini', verbose=1, save_best_only=True, save_weights_only = True, mode='max'),
            loadbest_callback(filepath, verbose=1)] 

def fbeta_bce_loss(y_true, y_pred, beta = 2):

    beta_sq = beta ** 2
    tp_loss = K.sum(y_true * (1 - K.binary_crossentropy(y_pred, y_true)), axis=-1)
    fp_loss = K.sum((1 - y_true) * K.binary_crossentropy(y_pred, y_true), axis=-1)

    return - K.mean((1 + beta_sq) * tp_loss / ((beta_sq * K.sum(y_true, axis = -1)) + tp_loss + fp_loss))
#    
def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    
    # next, build a rank loss
    
    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * K.cast(K.less(y_true,1), 'float32'))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(K.cast(K.greater(y_true,0), 'float32')) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss