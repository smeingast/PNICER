from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'


# ----------------------------------------------------------------------
# Import stuff
import numpy as np


# ----------------------------------------------------------------------
# Define helper functions for slope determination
def get_covar(xi, yi):
    """
    Calculate sample covariance
    :param xi: x data
    :param yi: y data
    :return: sample covariance
    """
    return np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / len(xi)


def get_beta_ols(xj, yj):
    """
    Get slope of ordinary least squares fit
    :param xj: x data
    :param yj: y data
    :return: slope of linear fit
    """
    return get_covar(xi=xj, yi=yj) / np.var(xj)


def get_beta_bces(x_sc, y_sc, cov_err_sc, var_err_sc):
    upper = get_covar(x_sc, y_sc) - cov_err_sc
    lower = np.var(x_sc) - var_err_sc
    return upper / lower


def get_beta_lines(x_sc, y_sc, x_cf, y_cf, cov_err_sc, cov_err_cf, var_err_sc, var_err_cf):
    """
    Get slope of distribution with LINES
    :param x_sc: x data science field
    :param y_sc: y data science field
    :param x_cf: x data control field
    :param y_cf: y data control field
    :param cov_err_sc: covariance of errors science field
    :param cov_err_cf: covariance of errors control field
    :param var_err_sc: variance in x science field
    :param var_err_cf: variance in y control field
    :return: slope of linear fit
    """
    upper = get_covar(x_sc, y_sc) - get_covar(x_cf, y_cf) - cov_err_sc + cov_err_cf
    lower = np.var(x_sc) - np.var(x_cf) - var_err_sc + var_err_cf
    return upper / lower
