"""
@author: dawidkopczyk
Script that does meta modelling 
"""

import numpy as np
import pandas as pd

from autolearn import Classifier, StackingClassifier

from sklearn.model_selection import  StratifiedKFold 
from sklearn.metrics import confusion_matrix
from utils import gini_normalized, plot_confusion_matrix
from params import get_param_grid
         

base_estimators = [("output/_lv0_lgbtest2_train", "output/_lv0_lgbtest2_test"), 
                   ("output/_lv0_lgbtest4_train", "output/_lv0_lgbtest4_test"),
                   ("output/_lv0_lgbtest5_train", "output/_lv0_lgbtest5_test"),
                   ("output/_lv0_xgbtest4_train", "output/_lv0_xgbtest4_test")]

base_estimators = [Classifier('Linear', num_bagged_est=1, random_state=2017), 
                   Classifier('Linear', num_bagged_est=2, random_state=2017, C=0.1)]
  
# Import data
df_train = pd.read_csv("input/train.csv")
y_train = df_train['target'].values[0:50000]
X_train = df_train.drop(['target','id'], axis=1).values[0:50000,1:]
ids_train = df_train['id'].values
df_train = []

df_test = pd.read_csv("input/test.csv")
X_test = df_test.drop(['id'], axis=1).values[0:50000,1:]
ids_test = df_test['id'].values
df_test = [] 

modelname = 'Linear'
outputname = 'lrfinal'

N_SPLITS = 5
SEED = 2017
BAGS = 15
outset = "output/_lv1_" + outputname

flagHeldOutMode = False
flagSubMode = True
flagGridMode = False

## Get params
param_grid_values = dict([a, x[0]] for a, x in get_param_grid(modelname).items())

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state = SEED)
stacking_clf = StackingClassifier(modelname, num_bagged_est=BAGS, random_state=SEED,
                                  base_estimators=base_estimators,
                                  base_cv=skf, base_scoring = gini_normalized, 
                                  base_copy_idx=None, base_save=True,
                                  base_drop_first=True, stacking_verbose=True)
stacking_clf.set_params(**param_grid_values)  
     
# Data for stacking mode
if flagHeldOutMode: 
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state = SEED)
    y_pred_proba_held_out = stacking_clf.cross_val_predict_proba(X_train, y_train, cv=skf, scoring=gini_normalized)
 
    # Create predictions for training data
    np.save(outset + '_train', y_pred_proba_held_out)
        
    # Calculate scores of cv
    score = gini_normalized(y_train, y_pred_proba_held_out[:,1])
    print("CV Gini: {:.4f}".format(score))
    
    # Calculate confusion matrix of cv
    cnf_matrix = confusion_matrix(y_train, y_pred_proba_held_out[:,1] > 0.5)
    plot_confusion_matrix(cnf_matrix, classes={'0','1'}, title='Confusion matrix')
    
# Final model and submission mode
if flagSubMode:
    
    # Create predictions for test data
    stacking_clf.fit(X_train, y_train)
    y_pred_proba_test = stacking_clf.predict_proba(X_test)
    np.save(outset + '_test', y_pred_proba_test)
    