from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'


# ----------------------------------------------------------------------
# Import stuff
import numpy as np

from astropy.io import fits
from pnicer import Magnitudes


# ----------------------------------------------------------------------
# PNICER initialization functions
def pnicer_ini(skip_science, skip_control, n_features=5, color=False, sfil=None, cfil=None):

    # Type assertion
    assert isinstance(skip_science, int) & isinstance(skip_control, int) & isinstance(n_features, int)

    # Define file paths
    science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
    control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"

    # Load data
    science_dummy = fits.open(science_path)[1].data
    control_dummy = fits.open(control_path)[1].data

    if sfil is None:
        sfil = np.arange(len(science_dummy["GLON"]))
    if cfil is None:
        cfil = np.arange(len(control_dummy["GLON"]))

    # Coordinates
    science_glon = science_dummy["GLON"][sfil][::skip_science]
    science_glat = science_dummy["GLAT"][sfil][::skip_science]
    control_glon = control_dummy["GLON"][cfil][::skip_control]
    control_glat = control_dummy["GLAT"][cfil][::skip_control]

    # Definitions
    features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
    errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
    features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

    # Photometry
    science_data = [science_dummy[n][sfil][::skip_science] for n in features_names[:n_features]]
    science_error = [science_dummy[n][sfil][::skip_science] for n in errors_names[:n_features]]
    control_data = [control_dummy[n][cfil][::skip_control] for n in features_names[:n_features]]
    control_error = [control_dummy[n][cfil][::skip_control] for n in errors_names[:n_features]]

    # Initialize data with PNICER
    science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                         lon=science_glon, lat=science_glat, names=features_names)
    control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                         lon=control_glon, lat=control_glat, names=features_names)

    if color:
        science = science.mag2color()
        control = control.mag2color()

    return science, control


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
