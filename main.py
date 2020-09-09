import numpy as np
import pandas as pd
from util import lin_reg_mod, plot_all, pair_hist_plot
import os
import sys
import getopt

basepath = "/Users/cookiemonster/PycharmProjects/python-checkin/exportedData"
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # trouble shooting file path issues
    # for entry in os.listdir(basepath):
    #     if os.path.isfile(os.path.join(basepath, entry)):
    #         print(entry)

    X_in = np.loadtxt(basepath + '/X_data.csv', dtype="int", delimiter=",")
    y_out = np.loadtxt(basepath + '/y_data.csv', dtype="float", delimiter=",")
    df = pd.read_csv(basepath + '/all.csv', delimiter=",")
    
    # print(df)
    # debugging
    # print(X_in)
    # print(y_out)
    # print(all)

    # split data, targets into training/testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X_in, y_out, random_state=0)

    # inspect shapes
    # print("X_train shape: {}".format(X_train.shape))
    # print("y_train shape: {}".format(y_train.shape))
    # print("X_test shape: {}".format(X_test.shape))
    # print("y_test shape: {}".format(y_test.shape))

    ## lin reg working
    #lin_reg_mod(df, 'Steps', 'Weight')
    #lin_reg_mod(df, 'Carb', 'Fiber')
    # lin_reg_mod(df, 'Date', 'Calories') # need to convert to datetime object or sth first
    plot_all(df)
    # pair_hist_plot(X_train, y_train)     # todo: what even is this