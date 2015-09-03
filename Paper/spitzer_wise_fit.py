from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


# ----------------------------------------------------------------------
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl

# from scipy import optimize
from scipy.odr import Model, RealData, ODR
from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
# Define file paths
sxw_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_X_WISE.fits"


# ----------------------------------------------------------------------
# Convenience function for plotting
def get_sign(val):
    if np.sign(val) == -1:
        return "-"
    elif np.sign(val) == 1:
        return "+"
    else:
        return ""


# ----------------------------------------------------------------------
# Load colorbrewer colormaps
cmap = brewer2mpl.get_map('RdBu', 'Diverging', number=4, reverse=True).get_mpl_colormap(N=20, gamma=1)


# ----------------------------------------------------------------------
# Read data
skip = 1
sxw_data = fits.open(sxw_path)[1].data

irac1 = sxw_data["IRAC1"][::skip]
irac1err = sxw_data["IRAC1_err"][::skip]
irac2 = sxw_data["IRAC2"][::skip]
irac2err = sxw_data["IRAC2_err"][::skip]
wise1 = sxw_data["w1mpro"][::skip]
wise1_err = sxw_data["w1sigmpro"][::skip]
wise2 = sxw_data["w2mpro"][::skip]
wise2_err = sxw_data["w2sigmpro"][::skip]


i1w1_filter = np.isfinite(irac1) & np.isfinite(wise1) & np.isfinite(irac1err) & np.isfinite(wise1_err)
i2w2_filter = np.isfinite(irac2) & np.isfinite(wise2) & np.isfinite(irac2err) & np.isfinite(wise2_err)


# ----------------------------------------------------------------------
# Define linear function
def f(vec, val):
    """Linear function y = m*x + b
    :param vec: vector of the parameters
    :param val: array of the current x values
    """
    return vec[0] * val + vec[1]


# ----------------------------------------------------------------------
# Perform model fits to IRAC1 vs. WISE1 and IRAC2 vs. WISE2
model = Model(f)

fit = []
for x, y, xerr, yerr in zip([wise1[i1w1_filter], wise2[i2w2_filter]],
                            [irac1[i1w1_filter], irac2[i2w2_filter]],
                            [wise1_err[i1w1_filter], wise2_err[i2w2_filter]],
                            [irac1err[i1w1_filter], irac2err[i2w2_filter]]):

    fit_data = RealData(x, y, sx=xerr, sy=yerr)
    fit_odr = ODR(fit_data, model, beta0=[1., 0.])
    fit.append(fit_odr.run())

# for f in fit:
#     print(f.beta, f.sd_beta)

# Get solutions
# iw1_m, iw1_b = iw1_out.beta
# Get errors
# iw1_merr, iw1_berr = iw1_out.sd_beta


# ----------------------------------------------------------------------
# Plot results
fig = plt.figure(figsize=(10, 15))

# Create plot grid with Gridspec
grid = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.01, wspace=0.01,
                height_ratios=[1, 1, 1], width_ratios=[1, 1, 0.05])

# Add axes to figure
ax0 = plt.subplot(grid[0])
ax1 = plt.subplot(grid[1])
cax = plt.subplot(grid[2])

x_fit = np.linspace(3, 19, 2)

s = ax0.scatter(wise1[i1w1_filter], irac1[i1w1_filter],
                c=np.sqrt(irac1err[i1w1_filter] ** 2 + wise1_err[i1w1_filter] ** 2), s=5,
                alpha=0.5, lw=0, cmap=cmap, vmin=0, vmax=0.4)
ax0.plot(x_fit, fit[0].beta[0] * x_fit + fit[0].beta[1], '-k', lw=2, ls="--")
ax0.set_xlabel("WISE$_{3.4}$ (mag)")
ax0.set_ylabel("IRAC (mag)")

# Write equation
ax0.annotate("IRAC$_{3.6}$ = " + str(np.around(fit[0].beta[0], 4)) + r"$\times$ WISE$_{3.4}$ " +
             get_sign(fit[0].beta[1]) + " " + str(np.abs(np.around(fit[0].beta[1], 4))),
             xy=(0.05, 0.97), xycoords="axes fraction", ha="left", va="top", size=12)

ax1.scatter(wise2[i2w2_filter], irac2[i2w2_filter],
            c=np.sqrt(irac2err[i2w2_filter] ** 2 + wise2_err[i2w2_filter] ** 2), s=5,
            alpha=0.5, lw=0, cmap=cmap, vmin=0, vmax=0.4)

ax1.plot(x_fit, fit[1].beta[0] * x_fit + fit[1].beta[1], '-k', lw=2, ls="--")
ax1.set_xlabel("WISE$_{4.6}$ (mag)")
# ax1.set_ylabel("IRAC2 (mag)")

# Write equation
ax1.annotate("IRAC$_{4.5}$ = " + str(np.around(fit[1].beta[0], 4)) + r"$\times$ WISE$_{4.6}$ " +
             get_sign(fit[1].beta[1]) + " " + str(np.abs(np.around(fit[1].beta[1], 4))),
             xy=(0.05, 0.97), xycoords="axes fraction", ha="left", va="top", size=12)

# Remove tick labels
ax1.axes.yaxis.set_ticklabels([])


for ax in [ax0, ax1]:

    # Plot limits
    ax.set_xlim(1, 21)
    ax.set_ylim(1, 21)

    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))


# Add colorbar
cb = plt.colorbar(cax=cax, mappable=s)
cb.set_label('Combined error (mag)')
# Set alpha to 1
cb.set_alpha(1)
cb.draw_all()
cb.set_ticks(MultipleLocator(0.1))

plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/spitzer_wise_fit.png", bbox_inches="tight", dpi=300)


# ----------------------------------------------------------------------
# AstroML stuff
# xdata = wise1[i1w1_filter]
# xdata_err = wise1_err[i1w1_filter]
# ydata = irac1[i1w1_filter]
# ydata_err = irac1err[i1w1_filter]

# xdata = irac2[i2w2_filter]
# ydata = wise2[i2w2_filter]
# xdata_err = irac2err[i2w2_filter]
# ydata_err = wise2_err[i2w2_filter]


# # ----------------------------------------------------------------------
# # Define some convenience functions
#
# # translate between typical slope-intercept representation, and the normal vector representation
# def get_m_b(beta):
#     b = np.dot(beta, beta) / beta[1]
#     m = -beta[0] / beta[1]
#     return m, b
#
#
# def get_beta(m, b):
#     denom = (1 + m * m)
#     return np.array([-b * m / denom, b / denom])
#
#
# def my_TLS_logL(v, X, dX):
#     """Compute the total least squares log-likelihood
#
#     This uses Hogg et al eq. 29-32
#
#     Parameters
#     ----------
#     v : ndarray
#         The normal vector to the linear best fit.  shape=(D,).
#         Note that the magnitude |v| is a stand-in for the intercept.
#     X : ndarray
#         The input data.  shape = [N, D]
#     dX : ndarray
#         The covariance of the errors for each point.
#         For diagonal errors, the shape = (N, D) and the entries are
#         dX[i] = [sigma_x1, sigma_x2 ... sigma_xD]
#         For full covariance, the shape = (N, D, D) and the entries are
#         dX[i] = Cov(X[i], X[i]), the full error covariance.
#
#     Returns
#     -------
#     logL : float
#         The log-likelihood of the model v given the data.
#
#     Notes
#     -----
#     This implementation follows Hogg 2010, arXiv 1008.4686
#     """
#     # check inputs
#     X, dX, v = map(np.asarray, (X, dX, v))
#     N, D = X.shape
#     assert v.shape == (D,)
#     assert dX.shape in ((N, D), (N, D, D))
#
#     v_norm = np.linalg.norm(v)
#     v_hat = v / v_norm
#
#     # eq. 30
#     delta = np.dot(X, v_hat) - v_norm
#
#     # eq. 31
#     if dX.ndim == 2:
#         # diagonal covariance
#         sig2 = np.sum(dX * v_hat ** 2, 1)
#     else:
#         # full covariance
#         sig2 = np.dot(np.dot(v_hat, dX), v_hat)
#
#     dummy = -0.5 * np.sum(np.log(2 * np.pi * sig2)) - np.sum(0.5 * delta ** 2 / sig2)
#     return dummy
#
# def my_convert_to_stdev(logL):
#     """
#     Given a grid of log-likelihood values, convert them to cumulative
#     standard deviation.  This is useful for drawing contours from a
#     grid of likelihoods.
#     """
#     sigma = np.exp(logL)
#
#     shape = sigma.shape
#     sigma = sigma.ravel()
#
#     # obtain the indices to sort and unsort the flattened array
#     i_sort = np.argsort(sigma)[::-1]
#     i_unsort = np.argsort(i_sort)
#
#     sigma_cumsum = sigma[i_sort].cumsum()
#     sigma_cumsum /= sigma_cumsum[-1]
#
#     return sigma_cumsum[i_unsort].reshape(shape)
#
#
# # ----------------------------------------------------------------------
# # Get data into shape
# data = np.vstack([xdata, ydata]).T
#
# # Create covariance matrix
# cov = np.zeros((len(xdata), 2, 2))
# cov[:, 0, 0] = xdata_err ** 2
# cov[:, 1, 1] = ydata_err ** 2
# cov[:, 0, 1] = cov[:, 1, 0] = 0.     # uncorrelated errors!
#
#
# # ----------------------------------------------------------------------
# # Find best-fit parameters
# min_func = lambda beta: -my_TLS_logL(beta, data, cov)
# beta_fit, beta_all = optimize.fmin(min_func, x0=[0, 0], maxiter=1000, maxfun=1000, disp=False, retall=True, xtol=1E-5)
#
# # Transform to (slope, intercept)
# m_fit, b_fit = get_m_b(beta_fit)
# print(m_fit, b_fit)
#
#
# # # ----------------------------------------------------------------------
# # # Plot the data and fits
# # fig, [ax, ax1] = plt.subplots(1, 2, figsize=[20, 10])
# #
# #
# # # ----------------------------------------------------------------------
# # # first let's visualize the data
# # ax.scatter(xdata, ydata, c='k', s=9)
# #
# # # ----------------------------------------------------------------------
# # # plot the best-fit line
# # x_fit = np.linspace(5, 20, 2)
# # ax.plot(x_fit, m_fit * x_fit + b_fit, '-k')
# # ax.set_xlabel("IRAC")
# # ax.set_ylabel("WISE")
# #
# #
# # # ----------------------------------------------------------------------
# # # plot the likelihood contour in m, b
# # m = np.linspace(0.9, 1.1, 10)
# # b = np.linspace(-1, 1, 10)
# # logL = np.zeros((len(m), len(b)))
# #
# # for i in range(len(m)):
# #     for j in range(len(b)):
# #         logL[i, j] = my_TLS_logL(get_beta(m[i], b[j]), data, cov)
# #         print(logL[i, j], my_convert_to_stdev(logL[i, j]))
# #         exit()
# #
# # ax1.contour(m, b, my_convert_to_stdev(logL.T), levels=(0.683, 0.955, 0.997), colors='k')
# #
# # # plt.show()
