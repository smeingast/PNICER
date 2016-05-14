# ----------------------------------------------------------------------
# Import stuff
from astropy.io import fits
from astropy.coordinates import SkyCoord

from pnicer import Magnitudes
from pnicer.utils import get_resource_path


""" This file goes through a typical PNICER session and creates a rough extinction map of Orion A from 2MASS data. """


# ----------------------------------------------------------------------
# Find the test files
science_path = get_resource_path(package="pnicer.tests_resources", resource="Orion_A_2mass.fits")
control_path = get_resource_path(package="pnicer.tests_resources", resource="CF_2mass.fits")


# ----------------------------------------------------------------------
# Define feature names and extinction vector
feature_names = ["Jmag", "Hmag", "Kmag"]
error_names = ["e_Jmag", "e_Hmag", "e_Kmag"]
feature_extinction = [2.5, 1.55, 1.0]


# ----------------------------------------------------------------------
# Load data
with fits.open(science_path) as science, fits.open(control_path) as control:

    sci, con = science[1].data, control[1].data

    # Coordinates
    sci_coo = SkyCoord(l=sci["GLON"], b=sci["GLAT"], frame="galactic", equinox="J2000", unit="deg")
    con_coo = SkyCoord(l=con["GLON"], b=con["GLAT"], frame="galactic", equinox="J2000", unit="deg")

    # Photometry
    sci_phot, con_phot = [sci[n] for n in feature_names], [con[n] for n in feature_names]

    # Errors
    sci_err, con_err = [sci[n] for n in error_names], [con[n] for n in error_names]


# ----------------------------------------------------------------------
# Initialize PNICER
science = Magnitudes(mag=sci_phot, err=sci_err, extvec=feature_extinction, coordinates=sci_coo, names=feature_names)
control = Magnitudes(mag=con_phot, err=con_err, extvec=feature_extinction, coordinates=con_coo, names=feature_names)


# ----------------------------------------------------------------------
# Run PNICER
pnicer = science.pnicer(control=control, add_colors=False)


# ----------------------------------------------------------------------
# Also run NICER
# nicer = science.nicer(control=control)

# ----------------------------------------------------------------------
# Create the extinction maps without any crazy setup
pnicer_emap = pnicer.build_map(bandwidth=3 / 60, metric="gaussian", frame="galactic",
                               sampling=2, nicest=False, use_fwhm=False)

# nicer_emap = pnicer.build_map(bandwidth=2 / 60, metric="gaussian", frame="galactic",
#                               sampling=2, nicest=False, use_fwhm=False)


# ----------------------------------------------------------------------
# Plot the PNICER extinction map
pnicer_emap.plot_map(figsize=10)

""" If no errors pop up, then the basic PNICER package works """
print("{0:<80s}".format("PNICER routines terminated successfully! Happy extinciton mapping :) "))
