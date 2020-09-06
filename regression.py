import matplotlib.pyplot as plt
import numpy as np
import os
basepath = "/Users/cookiemonster/PycharmProjects/python-checkin/exportedData"
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":

    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)

    X_in = np.loadtxt(basepath + '/X_data.csv', dtype="int", delimiter=",")
    y_out = np.loadtxt(basepath + '/y_data.csv', dtype="float", delimiter=",")

    print(X_in)
    print(y_out)