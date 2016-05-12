# ----------------------------------------------------------------------
# Import stuff
import os
from astropy.io import fits
from pnicer import Magnitudes
from pnicer.utils import get_resource_path

""" This file goes through a typical PNICER session and creates a rough extinction map of Orion A from 2MASS data. """


# ----------------------------------------------------------------------
# Find the test files
test_resources_path = os.path.join(os.path.dirname(__file__), "..", "tests_resources/")
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

    science_dummy = science[1].data
    control_dummy = control[1].data

    # Coordinates
    science_glon, control_glon = science_dummy["GLON"], control_dummy["GLON"]
    science_glat, control_glat = science_dummy["GLAT"], control_dummy["GLAT"]

    # Photometry
    science_data = [science_dummy[n] for n in features_names]
    control_data = [control_dummy[n] for n in features_names]

    # Errors
    science_error = [science_dummy[n] for n in errors_names]
    control_error = [control_dummy[n] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data with PNICER
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)


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
