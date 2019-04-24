import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

# input parameters
cat_filename = 'data/Catalog_Graham+2018_10YearPhot.dat'
# features_name = ['u10', 'g10', 'r10', 'i10', 'z10', 'y10']
features_name = ['u10', 'g-u', 'r-g', 'i-r', 'z-i', 'y-z']
labels_name = 'redshift'

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

    #ugrizy
    cat_data['g-u'] = cat_data['g10'] - cat_data['u10']
    cat_data['r-g'] = cat_data['r10'] - cat_data['g10']
    cat_data['i-r'] = cat_data['i10'] - cat_data['r10']
    cat_data['z-i'] = cat_data['z10'] - cat_data['i10']
    cat_data['y-z'] = cat_data['y10'] - cat_data['z10']
    
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
    return MAD((y_pred-y_truth)/(1+y_truth))

def target_func(x, c, A, n):
    """Target function for curve fitting"""
    return c + A*x**n

score_func = make_scorer(loss_func, greater_is_better=False)

