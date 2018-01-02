"""
@author: dawidkopczyk
Script that does the classifying models for Kaggle Porto Competition
"""

# data analysis and wrangling
import pandas as pd
import numpy as np

from autolearn import Classifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# utilis and params
from utils import gini_normalized, plot_confusion_matrix, gini_xgb, gini_lgb, gini_lgb2
from params import get_param_grid

# Change this
name_train = "input/train4.csv"
name_test = "input/test4.csv"

modelname = 'LightGBM'
outputname = 'lgbtest4'

N_SPLITS = 5
SEED = 2017
BAGS = 3
outset = "output/_lv0_" + outputname

flagHeldOutMode = False
flagSubMode = True
flagGridMode = False

# Import data
df_train = pd.read_csv(name_train)
data_train = df_train.drop(['target','id'], axis=1).values[:,1:]
target_train = df_train['target'].values
df_train = []

# Get params
param_grid_values = dict([a, x[0]] for a, x in get_param_grid(modelname).items())

clf = Classifier(modelname, num_bagged_est=BAGS, random_state=SEED)
clf.set_params(**param_grid_values) 

if modelname=="XGBoost":
    old_n_estimators = clf.get_params()['n_estimators']
    ret = clf.cross_validate(data_train, target_train, cv=StratifiedKFold(n_splits=4), scoring=gini_xgb, 
                               num_boost_round=old_n_estimators, stratified=True, 
                               early_stopping_rounds=100, maximize=True, verbose_eval=100)
    clf.set_params(n_estimators=ret['test-gini-mean'].idxmax()+1)
 
if modelname=="LightGBM":
    old_n_estimators = clf.get_params()['n_estimators']
    clf.set_params(objective='binary')
    ret = clf.cross_validate(data_train, target_train, cv=4, scoring=gini_lgb2, 
                               num_boost_round=old_n_estimators, stratified=True, 
                               early_stopping_rounds=100, verbose_eval=100)
    clf.set_params(n_estimators=np.array(ret['gini-mean']).argmax()+1)
        
# Data for stacking mode
if flagHeldOutMode: 
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state = SEED)
    y_pred_proba_held_out = clf.cross_val_predict_proba(data_train, target_train, cv=skf, scoring=gini_normalized)
 
    # Create predictions for training data
    np.save(outset + '_train', y_pred_proba_held_out)
        
    # Calculate scores of cv
    score = gini_normalized(target_train, y_pred_proba_held_out[:,1])
    print("CV Gini: {:.4f}".format(score))
    
    # Calculate confusion matrix of cv
    cnf_matrix = confusion_matrix(target_train, y_pred_proba_held_out[:,1] > 0.5)
    plot_confusion_matrix(cnf_matrix, classes={'0','1'}, title='Confusion matrix')
    
# Final model and submission mode
if flagSubMode:
        
    # Get test data
    df_test = pd.read_csv(name_test)
    data_test = df_test.drop(['id'], axis=1).values[:,1:]
    df_test = []
    
    # Create predictions for test data
    clf.fit(data_train, target_train)
    y_pred_proba_test = clf.predict_proba(data_test)
    np.save(outset + '_test', y_pred_proba_test)
    
    
