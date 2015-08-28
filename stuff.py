from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


# # ----------------------------------------------------------------------
# # Concatenate WISE and VISION Coordinates
# import numpy as np
# from astropy.io import fits
#
#
# tab_path = "/Users/Antares/Desktop/bla.fits"
#
# tab = fits.open(tab_path)[1].data
#
#
# ra = np.nanmean([tab["RA_VISION"], tab["RA_WISE"]], axis=0)
# dec = np.nanmean([tab["DEC_VISION"], tab["DEC_WISE"]], axis=0)
# print(np.sum(~np.isfinite(ra)))
# print(np.sum(~np.isfinite(dec)))
#
# col0 = fits.Column(name="IDX", format="J", array=np.arange(len(ra))+1)
# col1 = fits.Column(name="RA", format='D', array=ra)
# col2 = fits.Column(name="DEC", format='D', array=dec)
#
# cols = fits.ColDefs([col0, col1, col2])
# tbhdu = fits.BinTableHDU.from_columns(cols)
#
# tbhdu.writeto("/Users/Antares/Desktop/bla_new.fits")


# ----------------------------------------------------------------------
# Plot NICER vs NICEST from two maps
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

nicer_path = "/Users/Antares/Desktop/test.fits"
nicest_path = "/Users/Antares/Desktop/test_nicest.fits"

nicer = fits.open(nicer_path)[1].data
nicest = fits.open(nicest_path)[1].data



plt.scatter(nicest, nicest - nicer, lw=0, s=5, alpha=0.8)
plt.show()