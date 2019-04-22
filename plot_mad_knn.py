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
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from sklearn.neighbors import KNeighborsRegressor
from utils import *

#####################
# define parameters #
#####################

# model parameters
train_sample_sizes = np.logspace(3, 6, 30)  # 1E3 -> 1E6
test_sample_size = int(1E5)  # size of test sample

# regressor parameters
Regressor = KNeighborsRegressor
parameters = {
    "n_neighbors": 7,
    "leaf_size": 20,
    "p": 2,
}

# output parameters
output_dir = 'outputs'
output_prefix = 'knn_tuned'

###############
# plot styles #
###############

mpl.style.use('classic')
plt.rc('font', family='serif', serif='Times')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

################
# main program #
################

# load data: first load the maximum number of sample sizes that will
# be used
print("Loading data...")
cat_data = load_data(cat_filename)

# select a subset to be used for our training and testing
# randomly sample a max(train_sample_sizes) + test_sample_size subset
total_sample_size = int(round(max(train_sample_sizes) + test_sample_size))
cat_data = cat_data.sample(total_sample_size)

# split into two groups: train and test
cat_train = cat_data.head(int(round(max(train_sample_sizes))))
cat_test = cat_data.tail(test_sample_size)

# initialize an empty array to store the MAD for different sample sizes
mads = np.zeros(len(train_sample_sizes))

# initialize empty arrays to store the truth and predicted z values
y_pred_list = []
y_truth_list = []

# loop over sample_size
for i, sample_size in enumerate(train_sample_sizes):
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
    model = Regressor(**parameters)
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    y_truth = y_test

    # evaluating performance
    print("-> Evaluating performance...")
    mad = loss_func(y_pred, y_test)
    print("-> Performance MAD: %.4f" % mad)

    # save values to list for furthur processing
    mads[i] = mad
    y_pred_list.append(y_pred)
    y_truth_list.append(y_truth)
    
# check if output_dir exists
if not os.path.exists(output_dir):
    print("Path %s not found, creating now..." % output_dir)
    os.makedirs(output_dir)

# save data for future processing
data_filename = os.path.join(output_dir, "%s.npy" % output_prefix)
print("Saving data: %s" % data_filename)
output_data = np.stack([train_sample_sizes, mads], axis=0)
np.save(data_filename, output_data)

data_filename = os.path.join(output_dir, "%s_pred.npy" % output_prefix)
print("Saving data: %s" % data_filename)
output_data = np.stack(y_pred_list, axis=0)
np.save(data_filename, output_data)

data_filename = os.path.join(output_dir, "%s_truth.npy" % output_prefix)
print("Saving data: %s" % data_filename)
output_data = np.stack(y_truth_list, axis=0)
np.save(data_filename, output_data)

# save plots
fig_filename = os.path.join(output_dir, "%s.png" % output_prefix)
print("Saving plot: %s" % fig_filename)
plt.figure(figsize=(12,9))
plt.plot(train_sample_sizes, mads, 'b.')

# curve fitting
popt, pcov = curve_fit(target_func, train_sample_sizes, mads, bounds=([0,0,-1],[5,5,0]))
c, A, n = popt
label = r"$%.4f + %.4f N^{%.1f}$" % (c, A, n)
plt.plot(train_sample_sizes, target_func(train_sample_sizes, *popt), 'k-', label=label)

plt.xlabel("training sample sizes")
plt.ylabel("NMAD")
plt.xscale("log")
plt.legend()
plt.savefig(fig_filename)

print("Done!")
