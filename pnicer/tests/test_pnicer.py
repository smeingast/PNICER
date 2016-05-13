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
features_names = ["Jmag", "Hmag", "Kmag"]
errors_names = ["e_Jmag", "e_Hmag", "e_Kmag"]
features_extinction = [2.5, 1.55, 1.0]


# ----------------------------------------------------------------------
# Load data
with fits.open(science_path) as science, fits.open(control_path) as control:

    sci, con = science[1].data, control[1].data

    # Coordinates
    science_coo = SkyCoord(l=sci["GLON"], b=sci["GLAT"], frame="galactic", equinox="J2000", unit="deg")
    control_coo = SkyCoord(l=con["GLON"], b=con["GLAT"], frame="galactic", equinox="J2000", unit="deg")

    # Photometry
    science_phot, control_phot = [sci[n] for n in features_names], [sci[n] for n in features_names]

    # Errors
    science_err, control_err = [sci[n] for n in errors_names], [con[n] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data with PNICER
science = Magnitudes(mag=science_phot, err=science_err, extvec=features_extinction, coordinates=science_coo,
                     names=features_names)
control = Magnitudes(mag=control_phot, err=control_err, extvec=features_extinction, coordinates=control_coo,
                     names=features_names)


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
# Plot the PNICER map
pnicer_emap.plot_map(figsize=10)

"""
If no errors pop up, then the basic PNICER package works
"""

print("{0:<80s}".format("PNICER routines terminated successfully! Happy extinciton mapping :) "))
