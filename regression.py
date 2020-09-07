import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def lin_reg_mod(X_train, X_test, y_train, y_test):
    # create linear regression obj
    reg = linear_model.LinearRegression()

    # train model with data sets
    reg.fit(X_train, y_train)

    # make predictions with test set
    y_pred = reg.predict(X_test)

    # coefficients
    print('Coefficients \n', reg.coef_)
    # mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # coefficient of determination (1 = perfect prediction)
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

    print('\nThis is X_test:\n{}'.format(X_test))
    print('\nMutating X_test with [:,1]:\n{}'.format(X_test[:, 1]))
    # plot output
    # plt.scatter(X_test[:,1], y_test, color='black')
    # plt.plot(X_test, y_pred, color='blue', linewidth=2)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()
