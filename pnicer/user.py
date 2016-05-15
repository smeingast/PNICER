# ----------------------------------------------------------------------
# import stuff
import numpy as np

from pnicer.common import DataBase
from pnicer.utils import get_sample_covar, get_color_covar


# ----------------------------------------------------------------------
# noinspection PyProtectedMember
class Magnitudes(DataBase):

    def __init__(self, mag, err, extvec, coordinates=None, names=None):
        """
        Main class for users. Includes PNICER and NICER.

        Parameters
        ----------
        mag : list
            List of magnitude arrays. All arrays must have the same length.
        err : list
            List off magnitude error arrays.
        coordinates : SkyCoord, optional
            Astropy SkyCoord instance.
        extvec : list
            List holding the extinction components for each magnitude.
        names : list, optional
            List of magnitude (feature) names.

        """

        # Call parent
        super(Magnitudes, self).__init__(mag=mag, err=err, extvec=extvec, coordinates=coordinates, names=names)

        # Create color names
        self.colors_names = [self.features_names[k - 1] + "-" + self.features_names[k]
                             for k in range(1, self.n_features)]

    # ----------------------------------------------------------------------
    def mag2color(self):
        """
        Method to convert magnitude to color instances.

        Returns
        -------
        Colors
            Colors instance.

        """

        # Calculate colors
        colors = [self.features[k - 1] - self.features[k] for k in range(1, self.n_features)]

        # Calculate color errors
        colors_error = [np.sqrt(self.features_err[k - 1] ** 2 + self.features_err[k] ** 2)
                        for k in range(1, self.n_features)]

        # Color names
        color_extvec = [self.extvec.extvec[k - 1] - self.extvec.extvec[k] for k in range(1, self.n_features)]

        # Return Colors instance
        return Colors(mag=colors, err=colors_error, extvec=color_extvec, coordinates=self.coordinates,
                      names=self.colors_names)

    # ----------------------------------------------------------------------
    def _color_combinations(self):
        """
        Calculates a list of Colors instances for all combinations.

        Returns
        -------
        iterable
            List of Colors instances.

        """

        # Get all colors and then all combinations of colors
        return self.mag2color()._all_combinations(idxstart=1)

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov", add_colors=False):
        """
        Main PNICER method for magnitudes. Includes options to use combinations for input features, or convert them
        to colors.

        Parameters
        ----------
        control
            Control field instance. Same class as self.
        sampling : int, optional
            Sampling of grid relative to bandwidth of kernel. Default is 2.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
        add_colors : bool, optional
            Whether to also include the colors generated from the given magnitudes.

        Returns
        -------
        pnicer.extinction.Extinction
            Extinction instance with the calcualted extinction and errors.

        """

        if add_colors:

            # To create a color, we need at least two features
            if self.n_features < 2:
                raise ValueError("To use colors, at least two features are required")

            # Build all combinations by adding colors to parameter space; also require at least two magnitudes
            comb = zip(self._all_combinations(idxstart=2) + self._color_combinations(),
                       control._all_combinations(idxstart=2) + control._color_combinations())
        else:

            # Build combinations, but start with 2.
            comb = zip(self._all_combinations(idxstart=2), control._all_combinations(idxstart=2))

        # Call PNICER
        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)

    # ----------------------------------------------------------------------
    def nicer(self, control, n_features=None):
        """
        NICER routine as descibed in Lombardi & Alves 2001. Generalized for arbitrary input magnitudes

        Parameters
        ----------
        control
            Control field instance. Same class as self.
        n_features : int, optional
            If set, return only extinction values for sources with data for 'n' features.

        Returns
        -------
        pnicer.extinction.Extinction
            Extinction instance with the calcualted extinction and errors.

        """

        # Some checks
        self._check_class(ccls=control)
        if self.n_features != control.n_features:
            raise ValueError("Number of features do not match")

        # Features to be required can only be as much as input features
        if n_features is not None:
            assert n_features <= self.n_features, "Can't require more features than there are available"
            assert n_features > 0, "Must require at least one feature"

        # Get reddening vector
        k = [x - y for x, y in zip(self.extvec.extvec[:-1], self.extvec.extvec[1:])]

        # Calculate covariance matrix of control field
        cov_cf = np.ma.cov([np.ma.masked_invalid(control.features[l]) - np.ma.masked_invalid(control.features[l + 1])
                            for l in range(self.n_features - 1)])

        # Get intrisic color of control field
        color_0 = [np.nanmean(control.features[l] - control.features[l + 1]) for l in range(control.n_features - 1)]

        # Replace NaN errors with a large number
        errors = []
        for e in self.features_err:
            a = e.copy()
            a[~np.isfinite(a)] = 1000
            errors.append(a)

        # Calculate covariance matrix of errors in the science field
        cov_er = np.zeros([self.n_data, self.n_features - 1, self.n_features - 1])
        for i in range(self.n_features - 1):
            # Diagonal
            cov_er[:, i, i] = errors[i] ** 2 + errors[i + 1] ** 2
            # Other entries
            if i > 0:
                cov_er[:, i, i - 1] = cov_er[:, i - 1, i] = -errors[i] ** 2

        # Total covariance matrix
        cov = cov_cf + cov_er

        # Invert
        cov_inv = np.linalg.inv(cov)

        # Get b from the paper (equ. 12)
        upper = np.dot(cov_inv, k)
        b = upper.T / np.dot(k, upper.T)

        # Get colors
        scolors = np.array([self.features[l] - self.features[l + 1] for l in range(self.n_features - 1)])

        # Get those with no good color value at all
        bad_color = np.all(np.isnan(scolors), axis=0)

        # Write finite value for all NaNs (this makes summing later easier)
        scolors[~np.isfinite(scolors)] = 0

        # Put back NaNs for those with only bad colors
        scolors[:, bad_color] = np.nan

        # Equation 13 in the NICER paper
        ext = b[0, :] * (scolors[0, :] - color_0[0])
        for i in range(1, self.n_features - 1):
            ext += b[i, :] * (scolors[i, :] - color_0[i])

        # Calculate variance (has to be done in loop due to RAM issues!)
        first = np.array([np.dot(cov.data[idx, :, :], b.data[:, idx]) for idx in range(self.n_data)])
        var = np.array([np.dot(b.data[:, idx], first[idx, :]) for idx in range(self.n_data)])
        # Now we have to mask the large variance data again
        # TODO: This is the same as 1/denominator from Equ. 12
        var[~np.isfinite(ext)] = np.nan

        # Generate intrinsic source color list
        color_0 = np.repeat(color_0, self.n_data).reshape([len(color_0), self.n_data])
        color_0[:, ~np.isfinite(ext)] = np.nan

        if n_features is not None:
            mask = np.where(np.sum(np.vstack(self._features_masks), axis=0, dtype=int) < n_features)[0]
            ext[mask] = var[mask] = color_0[:, mask] = np.nan

        # ...and return :) Here, a Colors instance is returned!
        from pnicer.extinction import Extinction
        return Extinction(db=self.mag2color(), extinction=ext.data, variance=var, color0=color_0)

    # ----------------------------------------------------------------------
    # noinspection PyPackageRequirements
    @staticmethod
    def _get_beta(method, xdata, ydata, **kwargs):
        """
        Performs a linear fit with different user-specified methods.

        Parameters
        ----------
        method : str
            Method to be used. One of: 'lines', 'ols', 'bces', 'odr'.
        xdata : np.ndarray
            X data to be fit.
        ydata : np.ndarray
            Y data to be fit.
        kwargs
            Dictionary holding the data statistics (required for 'lines' and 'bces').

        Returns
        -------
        tuple(float, float)
            Tuple with the calculated slope and intercept.

        Raises
        ------
        ValueError
            If the given method is not supported.

        """

        # LINES
        if method.lower() == "lines":
            upper = get_sample_covar(xdata, ydata) - kwargs["cov_control"] - kwargs["science_err_covar"][1, 0] +\
                kwargs["control_err_covar"][1, 0]
            lower = np.var(xdata) - kwargs["science_err_covar"][0, 0] - kwargs["var_control"] + \
                kwargs["control_err_covar"][0, 0]
            beta = upper / lower

        # Ordinary least squares
        elif method.lower() == "ols":
            beta = get_sample_covar(xdata, ydata) / np.var(xdata)

        # BCES
        elif method.lower() == "bces":
            upper = get_sample_covar(xdata, ydata) - kwargs["science_err_covar"][1, 0]
            lower = np.var(xdata) - kwargs["science_err_covar"][0, 0]
            beta = upper / lower

        # Orthogonal distance regression
        elif method.lower() == "odr":

            from scipy.odr import Model, RealData, ODR

            # Define linear model
            def _linear_model(vec, val):
                return vec[0] * val + vec[1]

            # Perform ODR
            fit_model = Model(_linear_model)
            fit_data = RealData(xdata, ydata)  # sx=std_x, sy=std_y)
            bdummy = ODR(fit_data, fit_model, beta0=[1., 0.]).run()
            beta, ic = bdummy.beta
            # beta_err, ic_err = bdummy.sd_beta

        # If the given method is not suppoerted raise error.
        else:
            raise ValueError("Method {0:s} not supported (One of: 'lines', 'ols', 'bces', or 'odr')".format(method))

        # Calculate intercept (ODR has its own intercept)
        if method != "odr":
            ic = np.median(ydata) - beta * np.median(xdata)

        # Return slope and intercept
        # noinspection PyUnboundLocalVariable
        return beta, ic

    # ----------------------------------------------------------------------
    def color_excess_ratio(self, x_keys, y_keys, method="lines", control=None, kappa=1, sigma=3, err_iter=100,
                           qc=True):
        """
        Calculates the selective color excess rations (e.g.: E(J-H)/E(H-K)) for a given combinations of magnitudes. This
        slope is derived via different methods (LINES, BCES, OLS, or ODR).

        Parameters
        ----------
        x_keys : iterable
            List of magnitude keys to be put on the abscissa (e.g. for 2MASS ["Hmag", "Kmag"]).
        y_keys : iterable
            List of magnitude keys to be put on the ordinate (e.g. for 2MASS ["Jmag", "Hmag"]).
        method : str, optional
            Method to use for fitting the data. One of 'lines', 'bces', 'ols', 'odr'.
        control
            Control field instance (required for 'lines').
        kappa : int
            Number of clipping iterations in the fitting procedure. Default is 1.
        sigma : int, float, optional
            Sigma clipping factor in iterations. Default is 3.
        err_iter : int, optional
            Number of iterations for error calculation via bootstrapping. Default is 100.
        qc : bool, optional
            Whether to show a quality control plot of the results.

        Returns
        -------
        tuple(float, float, float)
            Tuple holding the slope, the error of the slope and the intercept of the fit.

        """

        # TODO: Check LINES since the fit looks very different from the other methods!

        # Add fit key if not set manually
        if isinstance(y_keys, str):
            y_keys = [x_keys[1], y_keys]

        # Some input checks
        if len(x_keys) != 2:
            raise ValueError("'base_keys' must be tuple or list with two entries")
        if (y_keys[0] not in self.features_names) | (y_keys[1] not in self.features_names):
            raise ValueError("'fit_keys' not found")
        if (x_keys[0] not in self.features_names) | (x_keys[1] not in self.features_names):
            raise ValueError("'base_keys' not found")
        if (kappa < 0) | (isinstance(kappa, int) is False):
            raise ValueError("'kappa' must be non-zero positive integer")
        if sigma <= 0:
            raise ValueError("'sigma' must be positive")
        if method.lower() == "lines":
            if control is None:
                raise ValueError("The LINES method requires control field data")

        # Get indices of requested keys
        x_idx, y_idx = self._name2index(name=x_keys), self._name2index(name=y_keys)

        # Create common masks for all given features for science data
        smask = self._custom_mask(names=x_keys + y_keys)

        # Apply mask
        xc_science = self.features[x_idx[0]][smask] - self.features[x_idx[1]][smask]
        yc_science = self.features[y_idx[0]][smask] - self.features[y_idx[1]][smask]
        x1_sc_err, x2_sc_err = self.features_err[x_idx[0]][smask], self.features_err[x_idx[1]][smask]
        y1_sc_err, y2_sc_err = self.features_err[y_idx[0]][smask], self.features_err[y_idx[1]][smask]

        # Create dictionary to pass to beta routine
        beta_dict = {}

        # If a control field is given:
        if control is not None:

            # Sanity check
            self._check_class(ccls=control)

            # Get combined mask for control field
            cmask = control._custom_mask(names=x_keys + y_keys)

            # Shortcuts for control field terms
            xc_control = control.features[x_idx[0]][cmask] - control.features[x_idx[1]][cmask]
            yc_control = control.features[y_idx[0]][cmask] - control.features[y_idx[1]][cmask]

            # Determine control field error covariance matrix
            beta_dict["control_err_covar"] = get_color_covar(*[control.features_err[x_idx[i]][cmask] for i in range(2)],
                                                             *[control.features_err[y_idx[i]][cmask] for i in range(2)],
                                                             *x_idx, *y_idx)

            # And sample covariance
            beta_dict["var_control"] = np.var(xc_control)
            beta_dict["cov_control"] = get_sample_covar(xc_control, yc_control)

        else:
            xc_control = yc_control = None

        # Dummy mask for first iteration
        smask = np.arange(len(xc_science))

        # Start iterations
        beta, ic = 0., 0.
        for _ in range(kappa + 1):

            # Mask data (used for iterations)
            xc_science, yc_science = xc_science[smask], yc_science[smask]
            x1_sc_err, x2_sc_err = x1_sc_err[smask], x2_sc_err[smask]
            y1_sc_err, y2_sc_err = y1_sc_err[smask], y2_sc_err[smask]

            # Determine covariance matrix of errors for science field
            beta_dict["science_err_covar"] = get_color_covar(x1_sc_err, x2_sc_err, y1_sc_err, y2_sc_err, *x_idx, *y_idx)

            # Determine slope and intercept
            beta, ic = self._get_beta(method=method, xdata=xc_science, ydata=yc_science, **beta_dict)

            # Get orthogonal distance to the fit
            dis = np.abs(beta * xc_science - yc_science + ic) / np.sqrt(beta ** 2 + 1)

            # Apply sigma clipping
            smask = dis - np.median(dis) < sigma * np.std(dis)

        # Do the same for random splits to get errors
        beta_err = []

        # Prepare dictionaries
        beta_dict_1 = {k: beta_dict[k] for k in beta_dict if k not in ["science_err_covar"]}
        beta_dict_split = [beta_dict_1, beta_dict_1.copy()]

        # Do as many iterations as requested
        for _ in range(err_iter):

            # Define random index array...
            ridx_sc = np.random.permutation(len(xc_science))

            # ...and split in half
            ridx = ridx_sc[0::2], ridx_sc[1::2]

            # Define samples
            xc_sc_split, yc_sc_split = [xc_science[i] for i in ridx], [yc_science[i] for i in ridx]
            x1_sc_err_split, x2_sc_err_split = [x1_sc_err[i] for i in ridx], [x2_sc_err[i] for i in ridx]
            y1_sc_err_split, y2_sc_err_split = [y1_sc_err[i] for i in ridx], [y2_sc_err[i] for i in ridx]

            # Calculate covariance matrices
            cov_sc_err_split = [get_color_covar(a, b, c, d, *x_idx, *y_idx) for a, b, c, d in
                                zip(x1_sc_err_split, x2_sc_err_split, y1_sc_err_split, y2_sc_err_split)]

            # ...and put them into the dictionaries
            for idx in range(2):
                beta_dict_split[idx]["science_err_covar"] = cov_sc_err_split[idx]

            # Get beta
            beta_i = [self._get_beta(method=method, xdata=x, ydata=y, **d)
                      for x, y, d in zip(xc_sc_split, yc_sc_split, beta_dict_split)]

            # Append beta values
            beta_err.append(np.std(list(zip(*beta_i))[0]))

        # Get final error estimate
        beta_err = 1.25 * np.sum(beta_err) / (np.sqrt(2) * err_iter)

        # Generate QC plot
        if qc:
            self._plot_exction_ratio(beta=beta, ic=ic, x_science=xc_science, y_science=yc_science, x_control=xc_control,
                                     y_control=yc_control)

        # Return fit and data values
        return beta, beta_err, ic

    # ----------------------------------------------------------------------
    @staticmethod
    def _plot_exction_ratio(beta, ic, x_science, y_science, x_control=None, y_control=None):
        """
        Generates the qc plot for the extinction ratio fit.

        Parameters
        ----------
        beta : float
            Slope of fit.
        ic : float
            Intercept of fit.
        x_science : np.ndarray
            X data for science field.
        y_science : np.ndarray
            Y data for science field.
        x_control : np.ndarray, optional
            X data for control field.
        y_control : np.ndarray, optional
            Y data for control field.

        """

        # Import
        import matplotlib.pyplot as plt

        # Create figure
        if x_control is not None:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[20, 10])
        else:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
            ax = [ax0]

        # Plot science field data
        ax[0].scatter(x_science, y_science, s=5, c="black", lw=0)

        # Plot control field data
        if x_control is not None:
            ax[1].scatter(x_control, y_control, s=5, c="black", lw=0)

        # Plot fit
        xd = np.arange(np.min(x_science) - 0.5, np.max(x_science) + 0.5, 1)
        ax[0].plot(xd, beta * xd + ic)

        # Show plot
        plt.show()


# ----------------------------------------------------------------------
# noinspection PyProtectedMember
class Colors(DataBase):

    def __init__(self, mag, err, extvec, coordinates=None, names=None):
        """
        Basically the same as magnitudes without NICER. Naturally the PNICER implementation does not allow to convert
        to colors.

        Parameters
        ----------
        mag
        err
        extvec
        coordinates
        names

        Returns
        -------

        """
        # TODO: Add docstring

        # Call parent
        super(Colors, self).__init__(mag=mag, err=err, extvec=extvec, coordinates=coordinates, names=names)

        # Add attributes
        self.colors_names = self.features_names

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov"):
        """
        PNICER call method for colors.

        Parameters
        ----------
        control
            Control field instance.
        sampling : int, optional
            Sampling of grid relative to bandwidth of kernel. Default is 2.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.

        """

        comb = zip(self._all_combinations(idxstart=1), control._all_combinations(idxstart=1))
        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)
