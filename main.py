import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import getopt

basepath = "/Users/cookiemonster/PycharmProjects/python-checkin/exportedData"
from sklearn import linear_model
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # trouble shooting file path issues
    # for entry in os.listdir(basepath):
    #     if os.path.isfile(os.path.join(basepath, entry)):
    #         print(entry)

    X_in = np.loadtxt(basepath + '/X_data.csv', dtype="int", delimiter=",")
    y_out = np.loadtxt(basepath + '/y_data.csv', dtype="float", delimiter=",")

    # debugging
    # print(X_in)
    # print(y_out)

    # split data, targets into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_in, y_out, random_state=0)

    # inspect shapes
    # print("X_train shape: {}".format(X_train.shape))
    # print("y_train shape: {}".format(y_train.shape))
    # print("X_test shape: {}".format(X_test.shape))
    # print("y_test shape: {}".format(y_test.shape))

    # inspect data
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    scatMatrix = pd.plotting.scatter_matrix(df, c=y_train, figsize=(5, 5), marker='o',
                                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap="CMRmap_r")
    plt.show()
    #print(df)

