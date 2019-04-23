"""This script plots the z_spec vs z_phot scatter data for a given
algorithm"""

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import os
from utils import loss_func

#####################
# define parameters #
#####################

input_dir = "./outputs/"
output_dir = "./outputs/knn/hist/"
prefix = "knn_tuned"

# decide if we want to plot all
plot_all = True

# otherwise choose an index to plot since we have many different
# testing samples
index_to_plot = 10

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

# load data
if not os.path.exists(input_dir):
    print("Error: input path doesn't exist!")

# get list of sample sizes and their corresponding scores
filename = os.path.join(input_dir, "%s.npy" % prefix)
if not os.path.exists(filename):
    print("Error: %s doesn't exists, exiting..." % filename)
data = np.load(filename)
sample_sizes = data[0,:]
mads = data[1,:]

# get the prediction results (photo-z)
filename = os.path.join(input_dir, "%s_pred.npy" % prefix)
if not os.path.exists(filename):
    print("Error: %s doesn't exists, exiting..." % filename)
y_preds = np.load(filename)

# get the true redshift (spec-z)
filename = os.path.join(input_dir, "%s_truth.npy" % prefix)
if not os.path.exists(filename):
    print("Error: %s doesn't exists, exiting..." % filename)
y_truths = np.load(filename)

# check if the folder exists
if not os.path.exists(output_dir):
    print("Folder %s doesn't existing, creating one now..." % output_dir)
    os.makedirs(output_dir)
    
plt.figure(figsize=(12,9))
for i in range(len(sample_sizes)):
    xdata = y_truths[i, :]
    ydata = y_preds[i, :]

    bad = np.ravel(np.where(np.abs(ydata-xdata)/(1+xdata)>=0.1))
    good = np.ravel(np.where(np.abs(ydata-xdata)/(1+xdata)<0.1))
    
    nmad = loss_func(ydata, xdata)
    outlier_ratio = len(bad)*100.0/len(xdata)
    
    plt.hist(y_truths[i], bins=100, histtype='step',lw=3,label=r'$z_{\rm spec}$');
    plt.hist(y_preds[i], bins=100, histtype='step',lw=3,label=r'$z_{\rm phot}$');
    plt.legend(loc="best", fontsize=28)
    plt.tick_params(labelsize=18)
    plt.xlabel(r"$z_{\rm spec}$", fontsize=28)
    plt.ylim(0, 4500)
    plt.title("Sample size = %d \n" % sample_sizes[i] + 
              r"$\sigma_{\rm NMAD} = %.3f$" % nmad + "\n" + \
              r"$(\Delta z) > 0.1(1+z)$ outliers = %.3f%%" % outlier_ratio)

    # save plot
    filename = os.path.join(output_dir, "%s_%d.png" % (prefix, i))
    print("Saving plot: %s" % filename)
    plt.savefig(filename)
    plt.clf()
