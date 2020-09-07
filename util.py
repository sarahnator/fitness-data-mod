import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plot_all(X_train):
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    print(df)
    df.plot()
    plt.title('Steps & Calories')
    plt.show()


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

    # print('\nThis is X_test:\n{}'.format(X_test))
    # print('\nMutating X_test with [:,1]:\n{}'.format(X_test[:, 1]))
    # plot output
    plt.scatter(X_test[:, 0], y_test, color='black')
    # plt.plot(X_test, y_pred, color='blue', linewidth=2)
    plt.xticks(())
    plt.yticks(())
    plt.title('Cal vs Weight')
    plt.show()


def pair_hist_plot(X_train, y_train):
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    print(df)
    scatMatrix = pd.plotting.scatter_matrix(df, c=y_train, figsize=(5, 5), marker='o',
                                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap="CMRmap_r")

    plt.show()
