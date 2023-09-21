import os
import statistics
from pathlib import Path

import learn

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
    plt.savefig('Assn1/Plots/Q2_1.jpg')
    plt.close()


def q_2_b():
    df = pd.read_csv("Assn1/data/mlr.csv")
    X = df.iloc[:,0:5].values
    y = df['y']
    # make MLR model, fit to the data, and predict y
    lm_MLR = linear_model.LinearRegression()
    model = lm_MLR.fit(X, y)
    ypred_MLR = lm_MLR.predict(X)  # y predicted by MLR
    intercept_MLR = lm_MLR.intercept_  # intercept predicted by MLR
    coef_MLR = lm_MLR.coef_  # regression coefficients in MLR model
    R2_MLR = lm_MLR.score(X, y)  # R-squared value from MLR model
    corr_coef = np.corrcoef(ypred_MLR, y)  # Pearson correlation between modeled and
    print('correlation = ' + str(corr_coef[0, 1]))
    print('MLR coeff = ' + str(coef_MLR[0, 1]))
    # visualize MLR model performance
    ax = plt.subplot()
    ax.scatter(y, ypred_MLR)

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # q_1_1()
    # q_1_2()
    # q_2_a()
    q_2_b()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
