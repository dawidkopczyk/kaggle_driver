"""
@author: dawidkopczyk
Script that contains utilities for Kaggle Porto Competition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools

# Define the gini metric - from https://www.kaggle.com/tezdhar/faster-gini-calculation
def gini(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return gini(a, p) / gini(a, a)

def gini_lgb(a, p):
    score = gini(a, p) / gini(a, a)
    return 'gini', score, True

def gini_lgb2(preds, train_data):
    labels = train_data.get_label()
    score = gini_normalized(labels, preds)
    return 'gini', score, True

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    # xgboost scikit wrapper eval always minimized
    return 'gini', gini_score

def gini_catboost(pred, y):
    return gini_normalized(y, pred)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.draw()
    plt.pause(0.001)

def plot_correlation_matrix(data, columns):
    
    d = pd.DataFrame(data=data, columns=columns)
    
    # Compute the correlation matrix
    corr = d.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.draw()
    plt.pause(0.001)
    
def save_results(ids, pred, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,target\n")
        for i in range(0,ids.size):
            f.write("%d,%.20f\n" % (ids[i], pred[i])) 


# Bayesian Computing Ensemble Weights using Bayesian Model Averaging.
def bayesian_model_averaging(preds, scores):
  n, m = preds.shape
  prior = np.repeat(1.0 / m, m)
  log_likelihood = np.zeros(m)
  z = -np.inf
  for i in range(0,m):
    s = scores[i]
    log_likelihood[i] =  m * (s * np.log(s) + (1.0 - s) * np.log(1.0 - s))
    z = max(z, log_likelihood[i])
  weights = prior * np.exp(log_likelihood - z)
  weights /= np.sum(weights) 
  return preds.dot(weights) 