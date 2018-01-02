"""
@author: Olorozrabiaka, dawidkopczyk
Script that does the dataprep for Kaggle Porto Competition
"""

# transformations
import pandas as pd
from sklearn.preprocessing import StandardScaler

# read data with -1 as NaNs
#train = pd.read_csv("train.csv", na_values=["-1", "-1.0"])
#test = pd.read_csv("test.csv", na_values=["-1", "-1.0"])

# read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# separate target variable
train_y = train[['target','id']]

# drop target variable from original set
train.drop(['target'], axis=1, inplace=True)

# concatenate train and test for feature engineering
df = pd.concat([train,test])
train, test = [], []

# sum of missing cols per row
df['n_miss'] = df.isnull().sum(axis=1)

# sums of binary features
# ind and calc features separately
ind_bin_cols = [col for col in df.columns if 'ind' in col and 'bin' in col]
calc_bin_cols = [col for col in df.columns if 'calc' in col and 'bin' in col]
    
df["ind_bin_sum"] = df[ind_bin_cols].sum(axis=1)
df["calc_bin_sum"] = df[calc_bin_cols].sum(axis=1)

# numeric
num_cols = [col for col in df.columns if '_calc_' in  col and '_bin' not in col]

# read important features from Boruta
important = pd.read_csv("boruta_outcome_negativenan.csv")

# read tentative features from Boruta
tentative = pd.read_csv("boruta_tentative_negativenan.csv")

# select column name and boolean identifier of importance
features = important.ix[:, 1:]

# rename columns
features.columns = ['Important', 'Name']

# if feature tentative then important
features['Important'] = features['Important'] + tentative.iloc[:, 1]

# create list of important featurs names and id
selected_features = list(features[(features['Important'] == True)]['Name']) + ['id'] + ['ps_ind_02_cat'] + num_cols

# selecto only important features and id
df_selected = df[selected_features]

# weird interaction
df_selected['weird'] = df_selected['ps_car_13'] * df_selected['ps_reg_03']

# binary nan flag for significant difference
df_selected['ps_ind_02_cat_null'] = 0
df_selected.loc[df_selected['ps_ind_02_cat'] ==- 1,'ps_ind_02_cat_null'] = 1
    
# standarize numeric variables
scaler = StandardScaler()
df_selected[num_cols] = scaler.fit_transform(df_selected[num_cols])
   
# ohe
cat_cols = [col for col in df_selected.columns if 'cat' in col]
df_selected = pd.get_dummies(data=df_selected, columns=cat_cols, drop_first=False)

# merge target variable
df_selected = pd.merge(left=df_selected, right=train_y, how='left', on='id')

# extract test data where target is missing
test_selected = df_selected[df_selected['target'].isnull()]
test_selected.drop(['target'], axis=1, inplace=True)
cols_test_selected = test_selected.columns.tolist()
cols_test_selected.insert(0, cols_test_selected.pop(cols_test_selected.index('id')))
test_selected = test_selected.reindex(columns=cols_test_selected)

# extract train data where target is not missing
train_selected = df_selected[df_selected['target'].isnull() == False]
cols_train_selected = train_selected.columns.tolist()
cols_train_selected.insert(0, cols_train_selected.pop(cols_train_selected.index('id')))
cols_train_selected.insert(1, cols_train_selected.pop(cols_train_selected.index('target')))
train_selected = train_selected.reindex(columns=cols_train_selected)

# save modified data sets to_csv
train_selected.to_csv('train_selected.csv',sep=',')
test_selected.to_csv('test_selected.csv',sep=',')