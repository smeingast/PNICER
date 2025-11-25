# -----------------------------------------------------------------------------
# Import stuff
import numpy as np

from astropy.io import fits
from pnicer.utils.algebra import round_partial


class ExtinctionCube:
    def __init__(self, cube, header=None):
        self.cube = cube
        self.header = header if header is not None else fits.Header(self.cube)

        # Get third axis range
        if self.header is not None:
            self.crange3 = (
                np.arange(self.cube.shape[0]) * self.header["CDELT3"]
                + self.header["CRVAL3"]
                - (self.header["CRPIX3"] - 1) * self.header["CDELT3"]
            )
        else:
            self.crange3 = np.arange(self.cube.shape[0])

    @property
    def shape(self):
        return self.cube.shape

    def save_fits(self, path, overwrite=True, precision=1e-4):
        """
        Save extinction cube as FITS file.

        Parameters
        ----------
        path : str
            File path. e.g. "/path/to/table.fits".
        overwrite : bool, optional
            Whether to overwrite exisiting files. Default is True.
        precision : float, optional
            The precision of the data in the output file. Default is 1E-4.

        """

        hdu = fits.PrimaryHDU(
            round_partial(self.cube, precision=precision), header=self.header
        )
        hdu.writeto(path, overwrite=overwrite)
