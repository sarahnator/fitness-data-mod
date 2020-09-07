import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def plot_all(X_train):
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    print(df)
    df.plot()
    plt.title('Steps & Calories')
    plt.show()


def lin_reg_mod(df, x, y):
    """
    Produces a least-squares linear regression model by plotting an attribute @param index against weight
    Produces statistics including linear coefficient, mean squared error, and coefficient of determination
    :param df: dataframe that contains combined mfp and fitbit data
    :param x: String of the set {'Date', 'Calories', 'Protein', 'Carb', 'Fat', 'Fiber', 'Steps, 'Distance'}. Functions as dependent variable.
    :param y: String of the set {'Date', 'Calories', 'Protein', 'Carb', 'Fat', 'Fiber', 'Steps, 'Distance'}. Functions as independent variable.
    """

    # create linear regression obj
    reg = linear_model.LinearRegression()

    # df to ndarray
    arr = df.values
    # use only one feature - don't try to not use np.newaxis or u get headache
    indices = {'Weight': 0, 'Date': 1, 'Calories': 2, 'Protein': 3, 'Carb': 4, 'Fat': 5, 'Fiber': 6, 'Steps': 7, 'Distance': 8}
    X_indexed = arr[:, np.newaxis, indices[x]]
    y_out = arr[:, np.newaxis, indices[y]]

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_indexed, y_out, random_state=0)

    # train model with data sets
    reg.fit(X_train, y_train)

    # make predictions with test set
    y_pred = reg.predict(X_test)

    # coefficients
    print(f"Coefficients: {reg.coef_}")
    # mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # coefficient of determination (1 = perfect prediction)
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

   # plot outputs
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=2)
    plt.title(f"{x} vs {y}")
    plt.xticks()
    plt.yticks()

    # labeling axes
    units = {'Weight': '[lb]', 'Distance': '[mi]', 'Carb': '[g]', 'Protein': '[g]', 'Fat': '[g]', 'Fiber': '[g]'}
    label1 = x
    label2 = y
    if x in units:
        label1 += (' ' + units[x])
    if y in units:
        label2 += (' ' + units[y])
    plt.ylabel(f"{label2}")
    plt.xlabel(f"{label1}")
    plt.show()


def pair_hist_plot(X_train, y_train):
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    print(df)
    scatMatrix = pd.plotting.scatter_matrix(df, c=y_train, figsize=(5, 5), marker='o',
                                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap="CMRmap_r")

    plt.show()
