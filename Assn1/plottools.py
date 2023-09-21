import matplotlib.pyplot as plt  # To visualize
from typing import Union, List, Tuple

from sklearn import linear_model
import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress


def plot_results(datas_and_fits, xlabel, ylabel, title, save):
    plt.scatter(datas_and_fits[xlabel], datas_and_fits[ylabel])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if False == save:
        plt.savefig('Plots/' + title + '.jpg')
        plt.close()
    return plt

# def plot_results(datas_and_fits, xlabel, ylabel, title):
#     plt.scatter(datas_and_fits[xlabel], datas_and_fits[ylabel])
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.savefig('Plots/' + title + '.jpg')
#     plt.close()

def plot_results_with_fit(datas_and_fits, xlabel, ylabel, title, y_pred):
    plt.scatter(datas_and_fits[xlabel], datas_and_fits[ylabel])
    plt.plot(datas_and_fits[xlabel], y_pred, color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('Plots/' + title + '.jpg')
    plt.close()


def plot_multiple(datas_and_fits, x_label, y_label, plot_fit):
    for d in datas_and_fits:
        test = d[0]
        plt.scatter(test[0], test[1])
        if plot_fit:
            plt.plot(test[0], test[2], label=d[1], color='red')


def plot_fit(ax: plt.Axes,
             x_vals: List[Union[int, float, np.float64]],
             y_vals: List[Union[int, float, np.float64]],
             low_fit: Union[None, int, float, np.float64] = None,
             high_fit: Union[None, int, float, np.float64] = None,
             color: str = 'C0'):
    """
    Plots a fitted line to the matplotlib Axes object passed to it.
    :param ax: matplotlib axes object to plot to.
    :param x_vals: List of numeric data types. Can also take an numpy
    ndarray, although type checking will complain. The independent values.
    :param y_vals: List of numeric data types. Can also take an numpy
    ndarray, although type checking will complain. The dependent values.
    :param low_fit: Minimum end of range to fit regression line to. All data
    points associated with values in x_values less than or equal to
    this value will be excluded from the fit. If no value is passed no data
    will be trimmed from the lower end of the data.
    :param high_fit: Maximum end of range to fit regression line to. All data
    points associated with values in x_values greater than or equal to
    this value will be excluded from the fit. If no value is passed no data
    will be trimmed from the upper end of the data.
    :param color: color of line to plot. Any string accepted by matplotlib
    may be used.
    :return: Returns the parameters of the fit.
    """
    # Validate x_vals, y_vals input
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must be equal in length")
    if len(x_vals) < 2:
        raise ValueError("We can't fit a line to data consisting of less than "
                         "two values.")
    # Set low and high cutoffs to minimum and maximum of range if not defined.
    if low_fit is None:
        low_fit = np.array([min(x_vals)])
        low_fit = np.nan_to_num(low_fit)[0]
    if high_fit is None:
        high_fit = np.array([max(x_vals)])
        high_fit = np.nan_to_num(high_fit)[0]
    # Trim data outside of range:
    trimmed_x_vals = np.array([x for x in x_vals
                               if low_fit <= x <= high_fit])
    trimmed_y_vals = np.array([y for x, y in zip(x_vals, y_vals)
                               if low_fit <= x <= high_fit])
    trimmed_x_vals = np.nan_to_num(trimmed_x_vals)
    trimmed_y_vals = np.nan_to_num(trimmed_y_vals)
    # Fit line and plot
    slope, intercept, r, p, stderr = linregress(trimmed_x_vals, trimmed_y_vals)
    fit_x_vals = [min(trimmed_x_vals), max(trimmed_x_vals)]
    fit_y_vals = [x * slope + intercept for x in fit_x_vals]
    ax.plot(fit_x_vals, fit_y_vals,
            label=f"({fit_x_vals[0]:0.2f}, {fit_x_vals[1]:0.2f}), "
                  f"m={slope:0.2f}, b={intercept:0.2f}",
            color=color)
    return slope, intercept, r, p, stderr

