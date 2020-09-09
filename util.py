import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def plot_macros(df):
    """
    Plots macronutrients, including fiber, and their averages on a single plot.
    TODO: make subplots for steps, distance, calories
    :param df: mfp and fitbit dataframe
    """

    strIndices = {'Weight': '0', 'Date': '1', 'Calories': '2', 'Protein': '3', 'Carb': '4', 'Fat': '5', 'Fiber': '6',
               'Steps': '7', 'Distance': '8'}
    macrosList = ['Carb', 'Protein', 'Fat', 'Fiber'] # fiber is not a macronutrient but for labeling purposes we will say yes
    macros = df[[strIndices['Carb'], strIndices['Protein'], strIndices['Fat'], strIndices['Fiber']]].copy()
    colors = ['b', 'r', 'y', 'm']

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    style = dict(size=10, color='gray') # unused

    for m in macrosList:
        # indexing
        index = macrosList.index(m)
        r = index // 2  # row
        c = index % 2   # col

        print(f"Average {m}:", macros[strIndices[m]].mean()) # print cool values

        # subplot stuff
        axs[r, c].set_title(f"{m}")
        axs[r, c].hlines(macros[strIndices[m]].mean(), -.5, 110, linestyles='dashed', color='gray')
        axs[r, c].annotate('Average', (45, macros[strIndices[m]].mean()))
        axs[r, c].tick_params(axis='both')
        axs[r, c].set_xlabel('Tracking Day Number')
        axs[r, c].set_ylabel('Quantity [g]')
        axs[r, c].plot(df[strIndices[m]], color=colors[index])

    fig.suptitle('Macronutrient Data', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(top=0.85)

    plt.show()

    ## difficult to read if all macronutrients clusted on single graph
    # fig, ax = plt.subplots()
    # macros.plot(ax=ax)
    # # df.plot()
    # for m in macrosList:
    #     print(f"Average {m}:", macros[strIndices[m]].mean())
    #     ax.hlines(macros[strIndices[m]].mean(), -.5, 110, linestyles='dashed')
    #     if m != 'Fiber':
    #         ax.annotate(f"Average {m}", (50, macros[strIndices[m]].mean() + 6))
    #     else:
    #         ax.annotate(f"Average {m}", (50, macros[strIndices[m]].mean() - 10))
    #
    # ax.legend(['Carbohydrate', 'Protein', 'Fat', 'Fiber'], fontsize=10)
    # ax.set_title('Macronutrients')
    # ax.tick_params(axis='both')
    # ax.set_xlabel('Tracking Day')
    # ax.set_ylabel('Quantity [g]')
    # plt.show()


def lin_reg_mod(df, x, y):
    """
    Produces a least-squares linear regression model by plotting attribute y as a function of attribute x
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
    linCoeff = f"Coefficients: {reg.coef_}"
    # mean squared error
    meanSqErr = 'Mean squared error: %.2f' % mean_squared_error(y_test, y_pred)
    # coefficient of determination (1 = perfect prediction)
    dCoef = 'Coefficient of determination: %.2f' % r2_score(y_test, y_pred)

   # # old plotting method plot outputs
   #  plt.scatter(X_test, y_test, color='black')
   #  plt.plot(X_test, y_pred, color='blue', linewidth=2)
   #  plt.title(f"{x} vs {y}")
   #  plt.xticks()
   #  plt.yticks()
   #
   #  # labeling axes
   #  units = {'Weight': '[lb]', 'Distance': '[mi]', 'Carb': '[g]', 'Protein': '[g]', 'Fat': '[g]', 'Fiber': '[g]'}
   #  label1 = x
   #  label2 = y
   #  if x in units:
   #      label1 += (' ' + units[x])
   #  if y in units:
   #      label2 += (' ' + units[y])
   #  plt.ylabel(f"{label2}")
   #  plt.xlabel(f"{label1}")
   #  plt.show()

    # plot outputs
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='black')
    ax.plot(X_test, y_pred, color='blue', linewidth=2)
    ax.set_title(f"{x} vs {y}")
    ax.tick_params(axis='both')

    # labeling axes
    units = {'Weight': '[lb]', 'Distance': '[mi]', 'Carb': '[g]', 'Protein': '[g]', 'Fat': '[g]', 'Fiber': '[g]'}
    label1 = x
    label2 = y
    if x in units:
        label1 += (' ' + units[x])
    if y in units:
        label2 += (' ' + units[y])
    ax.set_ylabel(f"{label2}")
    ax.set_xlabel(f"{label1}")

    # add data below fig
    textBox = dict(boxstyle='round, pad=1', facecolor='none', edgecolor='black')
    textStr = f"{linCoeff}\n{meanSqErr}\n{dCoef}"
    ax.annotate(textStr, xy=(0.5, 0),
            # credit: https: // stackoverflow.com / questions / 17086847 / box - around - text - in -matplotlib
            # Interpret the x as axes coords, and the y as figure coords
            xycoords=('axes fraction', 'figure fraction'),

            # The distance from the point that the text will be at
            xytext=(0, 10),
            # Interpret `xytext` as an offset in points...
            textcoords='offset points',

            # Any other text parameters we'd like
            size=10, ha='center', va='bottom', bbox=textBox, weight='bold')

    plt.subplots_adjust(bottom=0.25)
    plt.show()

def pair_hist_plot(X_train, y_train):
    df = pd.DataFrame(X_train, columns=['Calories', 'Steps'])
    print(df)
    scatMatrix = pd.plotting.scatter_matrix(df, c=y_train, figsize=(5, 5), marker='o',
                                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap="CMRmap_r")

    plt.show()
