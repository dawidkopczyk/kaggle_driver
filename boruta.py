"""
@author: Olorozrabiaka
Script that does the boruta for Kaggle Porto Competition
"""

# transformations
import pandas as pd
import numpy as np

# Boruta
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

train = pd.read_csv("train.csv", na_values=["-1", "-1.0"])

# train.info()
# train.describe()

# sum of missing cols per row
n_miss = train.isnull().sum(axis=1)
train = pd.read_csv("train.csv")
train["n_miss"] = n_miss

# sums of binary features
ind_bin_cols = [col for col in train.columns if 'ind' in col and 'bin' in col]
calc_bin_cols = [col for col in train.columns if 'calc' in col and 'bin' in col]

train["ind_bin_sum"] = train[ind_bin_cols].sum(axis=1)
train["calc_bin_sum"] = train[calc_bin_cols].sum(axis=1)

# brute NaN replacement
# train.fillna(value=999, inplace=True)

"""
BORUTA
"""

# drop id
train.drop(['id'], axis=1, inplace=True)

# create numpy arrays
X = train.ix[:, train.columns != 'target'].as_matrix()
y = train['target'].as_matrix()

# define classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, max_iter=20)

# abort RuntimeWarning: invalid value encountered in greater
np.seterr(invalid='ignore')

# run Boruta
feat_selector.fit(X, y)

features = pd.DataFrame(feat_selector.support_)

names = pd.DataFrame(train.ix[:, train.columns != 'target'].columns)

important_features = pd.concat([features.reset_index(drop=True), names], axis=1)

important_features.to_csv('boruta_outcome_negativenan.csv', sep=',')

# check ranking of features
ranking = pd.DataFrame(feat_selector.ranking_)

ranking_features = pd.concat([ranking.reset_index(drop=True),
                              names], axis=1)

ranking_features.to_csv('boruta_ranking_negativenan.csv', sep=',')

tentative = pd.DataFrame(feat_selector.support_weak_)

tentative_features = pd.concat([tentative.reset_index(drop=True),
                                names], axis=1)

tentative_features.to_csv('boruta_tentative_negativenan.csv', sep=',')