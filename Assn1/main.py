import itertools
from RegressionFunctions import stepwise_selection
import plottools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def q_1_1():
    # Use a breakpoint in the code line below to debug your script.
    df = pd.read_csv("Assn1/data/corr.csv")
    # Calculate Pearson correlation for each of the following pair of variables:
    #   x and y,
    #   x2 and y2
    #   x3 and y3
    coeff_x_y = np.corrcoef(df["x"], df["y"])[0, 1]
    coeff_x2_y2 = np.corrcoef(df["x2"], df["y2"])[0, 1]
    coeff_x3_y3 = np.corrcoef(df["x3"], df["y3"])[0, 1]
    print("x and y Pearson correlation:         {num}".format(num=coeff_x_y))
    print("x2 and y2 Pearson correlation:       {num}".format(num=coeff_x2_y2))
    print("x3 and y3 Pearson correlation:       {num}".format(num=coeff_x3_y3))


def q_1_2():
    df = pd.read_csv("Assn1/data/corr.csv")
    coeff_x_y = np.corrcoef(df["x"], df["y"])[0, 1]
    coeff_x2_y2 = np.corrcoef(df["x2"], df["y2"])[0, 1]
    coeff_x3_y3 = np.corrcoef(df["x3"], df["y3"])[0, 1]
    series = [
            [['x', 'y'], coeff_x_y],
            [['x2', 'y2'], coeff_x2_y2],
            [['x3', 'y3'], coeff_x3_y3]]

    # dataframe consisting of x,y,corr
    for d in series:
        x = d[0][0]
        y = d[0][1]

        plot = plottools.plot_results(
            df[[x, y]], # select x and y from dataframe
            x,
            y,
            '{xLab} v {yLab}'.format(xLab=x,yLab=y),
            True)
        plottools.plot_fit(plot, df[x], df[y])
        plt.savefig('Assn1/Plots/' + '{xLab} v {yLab}'.format(xLab=x,yLab=y) + '.jpg')
        plt.close()
    return 0


def q_2_a():
    df = pd.read_csv("Assn1/data/mlr.csv")
    fig = plt.figure()
    # dataframe consisting of x,y,corr
    for d in df.iloc[:,0:5].columns:
        x = range(df.shape[0])
        y = df[d]
        plt.plot(x, y, '-o', label=d)
        plt.ylabel(d)

    plt.xlabel('t')
    plt.legend()
    plt.savefig('Assn1/Plots/Q2_a.jpg')
    plt.close()


def q_2_b():
    df = pd.read_csv("Assn1/data/mlr.csv")
    data_norm = (df - df.mean()) / df.std()

    # normalize data
    X_norm = data_norm.iloc[:,0:5].values
    y_norm = data_norm['y']

    # make MLR model, fit to the data, and predict y
    lm_MLR = linear_model.LinearRegression()
    model = lm_MLR.fit(X_norm, y_norm)
    ypred_MLR = lm_MLR.predict(X_norm)  # y predicted by MLR
    intercept_MLR = lm_MLR.intercept_  # intercept predicted by MLR
    coef_MLR = lm_MLR.coef_  # regression coefficients in MLR model
    R2_MLR = lm_MLR.score(X_norm, y_norm)  # R-squared value from MLR model
    corr_coef = np.corrcoef(ypred_MLR, y_norm)  # Pearson correlation between modeled and
    print('correlation = ' + str(corr_coef[0, 1]))
    print('MLR coeff = ' + str(coef_MLR))
    # visualize MLR model performance
    ax = plt.subplot()
    ax.scatter(y_norm, ypred_MLR)

    ### We want to draw a 1:1 line but don't know the lims to draw it over -- so we query the plot lims,
    ### Draw a line accordingly and ensure to set the lims
    lims = ax.get_xlim()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, ls="--", c="pink")

    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.set_title(f'MLR Model Results: R$^2$ = {round(R2_MLR, 2)}')
    plt.savefig('Assn1/Plots/Q2_b.jpg')

def q_2_c():
    df = pd.read_csv("Assn1/data/mlr.csv")

    # normalize data
    data_norm = (df - df.mean()) / df.std()
    X_norm = data_norm.iloc[:,0:5]
    y_norm = data_norm['y']

    # use stepwise regression to find which predictors to use
    result = stepwise_selection(X_norm, y_norm)
    print('resulting features:')
    print(result)

    # do MLR using predictors chosen from stepwise regression
    lm_step = linear_model.LinearRegression()
    model_step = lm_step.fit(X_norm[result], y_norm)
    ypred_step = lm_step.predict(X_norm[result])  # y predicted by MLR
    intercept_step = lm_step.intercept_  # intercept predicted by MLR
    coef_step = lm_step.coef_  # regression coefficients in MLR model
    R2_step = lm_step.score(X_norm[result], y_norm)  # R-squared value from MLR model

    print('Stepwise MLR coeff = ' + str(coef_step))
    print('Stepwise R2 coeff = ' + str(R2_step))


    # visualize stepwise model performance
    ax = plt.subplot()
    ax.scatter(y_norm, ypred_step)
    lims = ax.get_xlim()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, ls="--", c="grey")
    ax.set_xlabel('Measured g')
    ax.set_ylabel('Modelled g')
    ax.set_title(f'Stepwise Model Results: R$^2$ = {round(R2_step, 2)}')
    plt.savefig('Assn1/Plots/Q2_c.jpg')
    # Apply calibration - validation approach on standardized input
    # goal: loop through every combination of normalized predictors, make linear model, and find one with best performance
    R2_best = []
    combo_best = []
    for kk in range(1, 5):  # for each total number of predictors to use in model (from 1 predictor to 5)
        v0 = range(np.shape(X_norm)[1])
        combinations = list(itertools.combinations(range(np.shape(X_norm)[1]), kk))  # all possible combinations of kk total predictors
        R2_test = []
        for ind in range(len(combinations)):  # for each combination of predictors, make MLR model and compute R ^ 2
            test_vars = np.array(combinations[ind])
            X_test = X_norm.iloc[::2, test_vars]  # calibation sample consists of all odd indices in the data
            y_test = y_norm.iloc[::2]
            X_valid = X_norm.iloc[1::2, test_vars]  # validation sample consists of all even indices in the data
            y_valid = y_norm.iloc[1::2]
            lm_test = linear_model.LinearRegression()
            model_test = lm_test.fit(X_test, y_test)
            ypred_test = lm_test.predict(X_test)  # y predicted by MLR
            R2_test.append(lm_test.score(X_valid, y_valid))  # R-squared value from MLR model
            R2_best.append(np.max(R2_test))
            combo_best.append(combinations[np.argmax(R2_test)])
        R2_best_final = np.max(R2_best)
        combo_best_final = combo_best[np.argmax(R2_best)]
        print('The best combination of predictors is: ')
        print(list(X_norm.columns[np.asarray(combo_best_final)]))

        # build linear model using the best combination of predictors
        X_calib_valid = X_norm.iloc[:, np.asarray(combo_best_final)]
        lm_calib_valid = linear_model.LinearRegression()
        model_calib_valid = lm_calib_valid.fit(X_calib_valid, y_norm)
        ypred_calib_valid = lm_calib_valid.predict(X_calib_valid)  # y predicted by MLR
        intercept_calib_valid = lm_calib_valid.intercept_  # intercept predicted by MLR
        coef_calib_valid = lm_calib_valid.coef_  # regression coefficients in MLR model
        R2_calib_valid = lm_calib_valid.score(X_calib_valid, y_norm)  # R-squared value from MLR model



if __name__ == '__main__':
    # q_1_1()
    # q_1_2()
    # q_2_a()
    # q_2_b()
    q_2_c()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

