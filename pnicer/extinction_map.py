# -----------------------------------------------------------------------------
# Import stuff
import numpy as np

from copy import copy
from astropy import wcs
from astropy.io import fits


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ExtinctionMap:

    # -----------------------------------------------------------------------------
    def __init__(self, map_ext, map_header=None, prime_header=None):

        # Map
        self.map_ext = map_ext

        # Headers
        self.prime_header = fits.Header() if prime_header is None else prime_header
        self.map_header = fits.Header() if map_header is None else map_header

    # -----------------------------------------------------------------------------
    @property
    def map_shape(self):
        """ Shape of map array. """
        return self.map_ext.shape

    # -----------------------------------------------------------------------------
    # noinspection PyTypeChecker
    @property
    def map_mask(self):
        """ Mask for bad entries in map. """
        try:
            # TODO: Check this
            return np.isnan(self.map_ext)
        except TypeError:
            return np.equal(self.map_ext, None)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class DiscreteExtinctionMap(ExtinctionMap):

    # -----------------------------------------------------------------------------
    def __init__(self, map_ext, map_var, map_header, prime_header=None, map_num=None, map_rho=None):
        """
        Extinction map class.

        Parameters
        ----------
        map_ext : np.ndarray
            2D Extintion map.
        map_var : np.ndarray
            2D Extinction variance map.
        map_header : astropy.fits.Header
            Header of grid from which extinction map was built.
        map_num : np.ndarray, optional
            2D source count map.
        map_rho : np.ndarray, optional
            2D source density map.

        """

        # Set instance attributes
        super(DiscreteExtinctionMap, self).__init__(map_ext=map_ext, map_header=map_header, prime_header=prime_header)
        self.map_var = map_var
        self.map_num = np.full_like(self.map_ext, fill_value=np.nan, dtype=np.uint32) if map_num is None else map_num
        self.map_rho = np.full_like(self.map_ext, fill_value=np.nan, dtype=np.float32) if map_num is None else map_rho

        # Sanity check for dimensions
        if (self.map_ext.ndim != 2) | (self.map_var.ndim != 2):
            raise TypeError("Input must be 2D arrays")

    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_vlim(data, percentiles, r=10):
        vmin = np.floor(np.percentile(data[np.isfinite(data)], percentiles[0]) * r) / r
        vmax = np.ceil(np.percentile(data[np.isfinite(data)], percentiles[1]) * r) / r
        return vmin, vmax

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    def plot_map(self, path=None, figsize=10):
        """
        Method to plot extinction map.

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. "/path/to/image.png". Default is None.
        figsize : int, float, optional
            Figure size for plot. Default is 10.

        """

        # Import
        import matplotlib
        import matplotlib.pyplot as plt

        # If the density should be plotted, set to 4
        nfig = 3

        fig = plt.figure(figsize=[figsize, nfig * 0.9 * figsize * (self.map_shape[0] / self.map_shape[1])])
        grid = matplotlib.gridspec.GridSpec(ncols=2, nrows=nfig, bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.08,
                                            wspace=0, height_ratios=[1] * nfig, width_ratios=[1, 0.05])

        # Set cmap
        cmap = copy(matplotlib.cm.binary)
        cmap.set_bad("#DC143C", 1.)

        for idx in range(0, nfig * 2, 2):

            ax = plt.subplot(grid[idx], projection=wcs.WCS(self.map_header))
            cax = plt.subplot(grid[idx + 1])

            # Plot Extinction map
            if idx == 0:
                vmin, vmax = self._get_vlim(data=self.map_ext, percentiles=[0.1, 90], r=100)
                im = ax.imshow(self.map_ext, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="Extinction (mag)")

            # Plot error map
            elif idx == 2:
                vmin, vmax = self._get_vlim(data=np.sqrt(self.map_var), percentiles=[1, 90], r=100)
                im = ax.imshow(np.sqrt(self.map_var), origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax,
                               cmap=cmap)
                if self.prime_header["METRIC"] == "median":
                    fig.colorbar(im, cax=cax, label="MAD (mag)")
                else:
                    fig.colorbar(im, cax=cax, label="Error (mag)")

            # Plot source count map
            elif idx == 4:
                vmin, vmax = self._get_vlim(data=self.map_num, percentiles=[1, 99], r=1)
                im = ax.imshow(self.map_num, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="N")

            elif idx == 6:
                vmin, vmax = self._get_vlim(data=self.map_rho, percentiles=[1, 99], r=1)
                im = ax.imshow(self.map_rho, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label=r"$\rho$")

            # Grab axes
            lon, lat = ax.coords[0], ax.coords[1]

            # Add axes labels
            if idx == (nfig - 1) * 2:
                lon.set_axislabel("Longitude")
            lat.set_axislabel("Latitude")

            # Hide tick labels
            if idx != (nfig - 1) * 2:
                lon.set_ticklabel_position("")

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------------------
    def save_fits(self, path, clobber=True):
        """
        Save extinciton map as FITS file.

        Parameters
        ----------
        path : str
            File path. e.g. "/path/to/table.fits".
        clobber : bool, optional
            Whether to overwrite exisiting files. Default is True.

        """

        # Create HDU list
        # noinspection PyTypeChecker
        hdulist = fits.HDUList([fits.PrimaryHDU(header=self.prime_header),
                                fits.ImageHDU(data=self.map_ext, header=self.map_header),
                                fits.ImageHDU(data=self.map_var, header=self.map_header),
                                fits.ImageHDU(data=self.map_num, header=self.map_header),
                                fits.ImageHDU(data=self.map_rho, header=self.map_header)])

        # Write
        hdulist.writeto(path, clobber=clobber)
