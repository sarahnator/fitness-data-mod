import matplotlib.pyplot as plt
import numpy as np
import os
basepath = "/Users/cookiemonster/PycharmProjects/python-checkin/exportedData"
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":

    # for entry in os.listdir(basepath):
    #     if os.path.isfile(os.path.join(basepath, entry)):
    #         print(entry)

    X_in = np.loadtxt(basepath + '/X_data.csv', dtype="int", delimiter=",")
    y_out = np.loadtxt(basepath + '/y_data.csv', dtype="float", delimiter=",")

    # print(X_in)
    # print(y_out)

    # split data, targets into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_in, y_out, random_state=0)
    # inspect shapes
    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test shape: {}".format(y_test.shape))

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
    print('\nMutating X_test with [:,1]:\n{}'.format(X_test[:,1]))
    # plot output
    plt.scatter(X_test[:,1], y_test, color='black')
    plt.scatter(X_test[:,0], y_test, color='blue')

   # plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()