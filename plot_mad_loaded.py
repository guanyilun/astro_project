"""This script loads a saved data and produce plots"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

##############
# parameters #
##############

data_filename = "outputs/randfor.npy"
output_dir = "outputs"
output_filename="randfor"

########
# main #
########

def target_func(x, c, A, n):
    """Target function for curve fitting"""
    return c + A*x**n

# save data for future processing
print("loading %s" % data_filename)
input_data = np.load(data_filename)

sample_sizes = input_data[0,:]
mads = input_data[1,:]

# save plots
fig_filename = os.path.join(output_dir, "%s.png" % output_filename)
print("Saving %s" % fig_filename)
plt.figure(figsize=(12,9))
plt.plot(sample_sizes, mads, 'b.')

# curve fitting
popt, pcov = curve_fit(target_func, sample_sizes, mads, bounds=([0,0,-1],[3,3,0]))
c, A, n = popt
label = r"$%.4f + %.4f N^{%.1f}$" % (c, A, n)
plt.plot(sample_sizes, target_func(sample_sizes, *popt), 'k-', label=label)

plt.xlabel("training sample sizes")
plt.ylabel("MAD")
plt.xscale("log")
plt.legend()
plt.savefig(fig_filename)
