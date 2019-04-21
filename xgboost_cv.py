"""This script aims to produce a MAD versus training sample size
plot. Here is a list of parameters that this script uses:

cat_filename: catalog filename
sample_sizes: the list of training sample sizes of interests
features_name: list of columns to be used as features
label_name: column to be used as label
Regressor: the Regressor class to use that supports sklearn-like syntax
output_dir: directory to saved the plot and data
output_prefix: filename prefix to identify the outputs

The script produces two files as output
{output_dir}/{output_prefix}.npy: MAD data for future plotting
{output_dir}/{output_prefix}.png: MAD vs sample size plot

"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

#####################
# define parameters #
#####################

# input parameters
cat_filename = 'data/Catalog_Graham+2018_10YearPhot.dat'

# model parameters
train_sample_size = int(1E5)  # 1E5
test_sample_size = int(1E5)  # size of test sample
features_name = ['u10', 'g10', 'r10', 'i10', 'z10', 'y10']
labels_name = 'redshift'

# regression parameters
Regressor = XGBRegressor
parameters = {
    "n_estimators": [100, 200, 500, 1000, 2000],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.1, 0.01, 0.001],
}

#####################
# utility functions #
#####################

def load_data(cat_filename):
    """Loads a given number of rows from catalog and return test and train"""
    # load catalog data
    cat_data = pd.DataFrame(np.array(pd.read_csv(cat_filename,
                                                 delim_whitespace=True,
                                                 comment='#',
                                                 header=None)))
    # give column names
    cat_data.columns = ['id','redshift','tu','tg','tr','ti','tz','ty',\
                        'u10','uerr10','g10','gerr10','r10','rerr10',\
                        'i10','ierr10','z10','zerr10','y10','yerr10']

    return cat_data

def prepare_data(cat_train, cat_test, use_scaler=True):
    """This function takes in a catalog dataframe and split it into 
    training and testing set that can be supplied to machine learning
    models such as those in sklearn"""
    X_train = cat_train[features_name].values
    y_train = cat_train[labels_name].values

    X_test = cat_test[features_name].values
    y_test = cat_test[labels_name].values

    # if using standard scaler
    if use_scaler:
        # initialize standard scaler
        scaler = StandardScaler()

        # train scaler on training data
        scaler.fit(X_train)

        # scale both train and test using this scaler
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def MAD(data, axis=0):
    """This function calculates the median absolute deviation"""
    absdev = np.abs(data - np.expand_dims(np.median(data, axis=axis), axis))
    return np.median(absdev, axis=axis)/0.6745

def loss_func(y_pred, y_truth):
    """This function calculates a loss function based on truth and prediction.
    loss = (z_photo - z_spec) / (1 + z_spec)
    and z_photo -> y_pred
        z_spec  -> y_truth
    """
    return (y_pred-y_truth)/(1+y_truth)

def target_func(x, c, A, n):
    """Target function for curve fitting"""
    return c + A*x**n

################
# main program #
################

# load data: first load the maximum number of sample sizes that will
# be used
print("Loading data...")
cat_data = load_data(cat_filename)

# select a subset to be used for our training and testing
# randomly sample a max(train_sample_sizes) + test_sample_size subset
total_sample_size = int(round(train_sample_size + test_sample_size))
cat_data = cat_data.sample(total_sample_size)

# split into two groups: train and test
cat_train = cat_data.head(int(round(train_sample_size)))
cat_test = cat_data.tail(test_sample_size)

# initialize an empty array to store the MAD for different sample sizes
sample_size = int(round(train_sample_size))

# first round it to nearest integar if not already 
sample_size = int(round(sample_size))
print("sample_size: %d" % sample_size)

# sample the required number of data from catalog
print("-> Randomly sampling %d data from catalog..." % sample_size)
cat_train_sample = cat_train.sample(sample_size)

# Get training and testing data and labels
print("-> Spliting into train and test sets...")
X_train, X_test, y_train, y_test = prepare_data(cat_train_sample, cat_test)

# train model
print("-> Training model...")

# use a grid search to narrow down the best parameters
model = GridSearchCV(estimator=Regressor(), param_grid=parameters)
model.fit(X_train, y_train)

# test model
print(model.best_params_)

# assign value to MAD list
print("Done!")
