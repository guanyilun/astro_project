"""This script aims to produce a MAD versus training sample size
plot. Here are a list of parameters that this script uses:

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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

#####################
# define parameters #
#####################

# input parameters
cat_filename = 'data/Catalog_Graham+2018_10YearPhot.dat'

# training parameters
sample_sizes = np.logspace(3, 6, 30)  # 1E3 -> 1E6
features_name = ['u10', 'g10', 'r10', 'i10', 'z10', 'y10']
labels_name = 'redshift'
Regressor = RandomForestRegressor

# output parameters
output_dir = 'outputs'
output_prefix = 'randfor'

#####################
# utility functions #
#####################

def load_data(cat_filename, nsamples=10000):
    """Loads a given number of rows from catalog and return test and train"""
    # load catalog data
    cat_data = pd.DataFrame(np.array(pd.read_csv(cat_filename,
                                                 delim_whitespace=True,
                                                 comment='#',
                                                 # nrows=nsamples,
                                                 header=None)))
    # give column names
    cat_data.columns = ['id','redshift','tu','tg','tr','ti','tz','ty',\
                        'u10','uerr10','g10','gerr10','r10','rerr10',\
                        'i10','ierr10','z10','zerr10','y10','yerr10']

    return cat_data

def prepare_data(cat_data, test_size=0.2, use_scaler=True):
    """This function takes in a catalog dataframe and split it into 
    training and testing set that can be supplied to machine learning
    models such as those in sklearn"""
    X = cat_data[features_name].values
    y = cat_data[labels_name].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

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

def loss(y_pred, y_truth):
    """This function calculates a loss function based on truth and prediction."""
    return (y_truth-y_pred)/(1+y_pred)

def target_func(x, c, A, n):
    """Target function for curve fitting"""
    return c + A*x**n

################
# main program #
################

# load data: first load the maximum number of sample sizes that will
# be used
print("Loading data...")
cat_data = load_data(cat_filename, nsamples=max(sample_sizes))

# initialize an empty array to store the MAD for different sample sizes
mads = np.zeros(len(sample_sizes))

# loop over sample_size
for i, sample_size in enumerate(sample_sizes):
    # first round it to nearest integar
    sample_size = int(round(sample_size))
    print("sample_size: %d" % sample_size)

    # sample the required number of data from catalog
    print("-> Randomly sampling %d data from catalog..." % sample_size)
    cat_data_sample = cat_data.sample(sample_size)

    # Get training and testing data and labels
    print("-> Spliting into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_data(cat_data_sample)

    # train model
    print("-> Training model...")
    model = Regressor()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    y_truth = y_test

    # plt.figure(figsize=(10,8))
    # plt.plot(y_pred, y_truth, 'k.', alpha=0.3)
    # plt.xlabel("Pred-z")
    # plt.ylabel("True-z")
    # plt.savefig("plots/%d.png" % sample_size)

    # evaluating performance
    print("-> Evaluating performance...")
    mad = MAD(loss(y_pred, y_test))
    print("-> Performance MAD: %.4f" % mad)
    
    # assign value to MAD list
    mads[i] = mad


# check if output_dir exists
if not os.path.exists(output_dir):
    print("Path %s not found, creating now..." % output_dir)
    os.makedirs(output_dir)

# save data for future processing
data_filename = os.path.join(output_dir, "%s.npy" % output_prefix)
print("Saving %s" % data_filename)
output_data = np.stack([sample_sizes, mads], axis=0)
np.save(data_filename, output_data)

# save plots
fig_filename = os.path.join(output_dir, "%s.png" % output_prefix)
print("Saving %s" % fig_filename)
plt.figure(figsize=(12,9))
plt.plot(sample_sizes, mads, 'b.')

# curve fitting
popt, pcov = curve_fit(target_func, sample_sizes, mads, bounds=([0,0,-1],[5,5,0]))
c, A, n = popt
label = r"$%.4f + %.4f N^{%.1f}$" % (c, A, n)
plt.plot(sample_sizes, target_func(sample_sizes, *popt), 'k-', label=label)

plt.xlabel("training sample sizes")
plt.ylabel("MAD")
plt.xscale("log")
plt.legend()
plt.savefig(fig_filename)

print("Done!")
