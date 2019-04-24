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
output_dir = "./outputs/randfor_diff"
prefix = "randfor_diff"

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

# plotting function
def generate_corr_plot(index_to_plot):
    """A function to generate the z_photo vs z_spec comparison
    
    Args:
        index_to_plot: an index to plot
        postfix: a post fix for the output filename
    """
    plt.figure(figsize=(10,10))
    xdata = y_truths[index_to_plot, :]
    ydata = y_preds[index_to_plot, :]

    bad = np.ravel(np.where(np.abs(ydata-xdata)/(1+xdata)>=0.1))
    good = np.ravel(np.where(np.abs(ydata-xdata)/(1+xdata)<0.1))
    
    nmad = loss_func(ydata, xdata)
    outlier_ratio = len(bad)*100.0/len(xdata)
    
    plt.plot(xdata[good], ydata[good], 'go', label="Accepted",
             alpha=0.2, markersize=2)
    plt.plot(xdata[bad], ydata[bad], 'ro', label="Outlier",
             alpha=0.2, markersize=2)

    plt.plot(xdata, xdata, 'k-')
    plt.plot(xdata, xdata+0.1*(1+xdata), 'k--')
    plt.plot(xdata, xdata-0.1*(1+xdata), 'k--')

    plt.title("Sample size = %d \n" % sample_sizes[index_to_plot] + 
              r"$\sigma_{\rm NMAD} = %.3f$" % nmad + "\n" + \
              r"$(\Delta z) > 0.1(1+z)$ outliers = %.3f%%" % outlier_ratio)
              
    plt.xlim(-0.1, 4,1)
    plt.ylim(-0.1, 4,1)    
    plt.xlabel(r"$z_{\rm spec}$", fontsize=20)
    plt.ylabel(r"$z_{\rm phot}$", fontsize=20)
    plt.legend(loc="upper left")

    output_filename = os.path.join(output_dir, "%s_corr_%d.png" %
                                   (prefix, index_to_plot))
    print("Saving plot: %s" % output_filename)
    plt.savefig(output_filename)
    plt.close()
    

# check if the folder exists
if not os.path.exists(output_dir):
    print("Folder %s doesn't existing, creating one now..." % output_dir)
    os.makedirs(output_dir)

# check if we want to plot all
if not plot_all:
    generate_corr_plot(index_to_plot)
else:
    for index_to_plot in range(len(sample_sizes)):
        generate_corr_plot(index_to_plot)
print("Done!")
