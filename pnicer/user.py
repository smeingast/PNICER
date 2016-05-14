# ----------------------------------------------------------------------
# import stuff
import numpy as np

from pnicer.common import DataBase
from pnicer.utils import get_covar


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

        # Some assertions
        if self.__class__ != control.__class__:
            raise ValueError("control and instance class do not match")
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
    def get_beta_lines(self, base_keys, fit_key, control, kappa=2, sigma=3, err_iter=1000):

        # TODO: Add docstring
        # TODO: Improve or remove!

        # Some input checks
        if len(base_keys) != 2:
            raise ValueError("'base_keys' must be tuple or list with two entries")
        self._check_class(ccls=control)
        if fit_key not in self.features_names:
            raise ValueError("'fit_key' not found")
        if (base_keys[0] not in self.features_names) | (base_keys[1] not in self.features_names):
            raise ValueError("'base_keys' not found")
        if (kappa < 0) | (isinstance(kappa, int) is False):
            raise ValueError("'kappa' must be non-zero positive integer")
        if sigma <= 0:
            raise ValueError("'sigma' must be positive")

        # Get indices of requested keys
        base_idx = (self.features_names.index(base_keys[0]), self.features_names.index(base_keys[1]))
        fit_idx = self.features_names.index(fit_key)

        # Create common masks for all given features
        smask = np.prod(np.vstack([self._features_masks[i] for i in base_idx + (fit_idx,)]), axis=0, dtype=bool)
        cmask = np.prod(np.vstack([control._features_masks[i] for i in base_idx + (fit_idx,)]), axis=0, dtype=bool)

        # Shortcuts for control field terms which are not clipped
        xdata_control = control.features[base_idx[0]][cmask] - control.features[base_idx[1]][cmask]
        ydata_control = control.features[base_idx[1]][cmask] - control.features[fit_idx][cmask]

        var_err_control = np.mean(control.features_err[base_idx[0]][cmask]) ** 2 + \
            np.mean(control.features_err[base_idx[1]][cmask]) ** 2
        cov_err_control = -np.mean(control.features_err[base_idx[1]][cmask]) ** 2
        var_control = np.var(xdata_control)
        cov_control = get_covar(xdata_control, ydata_control)

        x1_sc, x2_sc = self.features[base_idx[0]][smask], self.features[base_idx[1]][smask]
        x1_sc_err, x2_sc_err = self.features_err[base_idx[0]][smask], self.features_err[base_idx[1]][smask]
        y1_sc, y2_sc = x2_sc, self.features[fit_idx][smask]

        # Dummy mask for first iteration
        smask = np.arange(len(x1_sc))

        # Shortcut for data
        beta, ic = 0., 0.
        for _ in range(kappa + 1):

            # Mask data
            x1_sc, x2_sc = x1_sc[smask], x2_sc[smask]
            x1_sc_err, x2_sc_err = x1_sc_err[smask], x2_sc_err[smask]
            y1_sc, y2_sc = y1_sc[smask], y2_sc[smask]

            xdata_science, ydata_science = x1_sc - x2_sc, y1_sc - y2_sc

            # Determine (Co-)variance terms of errors for science field
            var_err_science = np.mean(x1_sc_err) ** 2 + np.mean(x2_sc_err) ** 2
            cov_err_science = -np.mean(x2_sc_err) ** 2

            # Determine beta for the given data
            # LINES
            upper = get_covar(xdata_science, ydata_science) - cov_control - cov_err_science + cov_err_control
            lower = np.var(xdata_science) - var_err_science - var_control + var_err_control
            beta = upper / lower

            # OLS
            # beta = get_covar(xdata_science, ydata_science) / np.var(xdata_science)

            # BCES
            # upper = get_covar(xdata_science, ydata_science) - cov_err_science
            # lower = np.var(xdata_science) - var_err_science
            # beta = upper / lower

            # Get intercept of linear fit through median
            ic = np.median(ydata_science) - beta * np.median(xdata_science)

            # ODR
            # from scipy.odr import Model, RealData, ODR
            # fit_model = Model(linear_model)
            # fit_data = RealData(xdata_science, ydata_science)#, sx=std_x, sy=std_y)
            # bdummy = ODR(fit_data, fit_model, beta0=[1., 0.]).run()
            # beta, ic = bdummy.beta
            # beta_err, ic_err = bdummy.sd_beta

            # Get orthogonal distance to this line
            dis = np.abs(beta * xdata_science - ydata_science + ic) / np.sqrt(beta ** 2 + 1)

            # 3 sig filter
            smask = dis - np.median(dis) < sigma * np.std(dis)
            # smask = dis - np.median(dis) < 0.15

        # Do the same for random splits to get errors
        beta_i = []
        for _ in range(err_iter):

            # Define random index array
            ridx_sc = np.random.permutation(len(x1_sc))

            # Split array in two equally sized parts and get data
            x1_sc_1, x2_sc_1 = x1_sc[ridx_sc[0::2]], x2_sc[ridx_sc[0::2]]
            x1_sc_2, x2_sc_2 = x1_sc[ridx_sc[1::2]], x2_sc[ridx_sc[1::2]]
            x1_sc_err_1, x2_sc_err_1 = x1_sc_err[ridx_sc[0::2]], x2_sc_err[ridx_sc[0::2]]
            x1_sc_err_2, x2_sc_err_2 = x1_sc_err[ridx_sc[1::2]], x2_sc_err[ridx_sc[1::2]]
            y1_sc_1, y2_sc_1 = y1_sc[ridx_sc[0::2]], y2_sc[ridx_sc[0::2]]
            y1_sc_2, y2_sc_2 = y1_sc[ridx_sc[1::2]], y2_sc[ridx_sc[1::2]]

            # Determine variance terms
            var_sc_1, var_sc_2 = np.nanvar(x1_sc_1 - x2_sc_1), np.nanvar(x1_sc_2 - x2_sc_2)
            var_sc_err_1 = np.nanmean(x1_sc_err_1) ** 2 + np.nanmean(x2_sc_err_1) ** 2
            var_sc_err_2 = np.nanmean(x1_sc_err_2) ** 2 + np.nanmean(x2_sc_err_2) ** 2

            # Determine covariance terms
            # TODO: This probably only works when no NaNs are present...
            cov_sc_1 = get_covar(xi=x1_sc_1 - x2_sc_1, yi=y1_sc_1 - y2_sc_1)
            cov_sc_2 = get_covar(xi=x1_sc_2 - x2_sc_2, yi=y1_sc_2 - y2_sc_2)
            cov_sc_err_1, cov_sc_err_2 = -np.nanmean(x1_sc_err_1) ** 2, -np.nanmean(x1_sc_err_2) ** 2

            upper1 = cov_sc_1 - cov_sc_err_1 - cov_control + cov_err_control
            lower1 = var_sc_1 - var_sc_err_1 - var_control + var_err_control
            beta1 = upper1 / lower1

            upper2 = cov_sc_2 - cov_sc_err_2 - cov_control + cov_err_control
            lower2 = var_sc_2 - var_sc_err_2 - var_control + var_err_control
            beta2 = upper2 / lower2

            # Append beta values
            beta_i.append(np.std([beta1, beta2]))

        # Get error estimate
        beta_err = 1.25 * np.sum(beta_i) / (np.sqrt(2) * 1000)

        # Return fit and data values
        return beta, beta_err, ic, x1_sc, x2_sc, y2_sc


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
