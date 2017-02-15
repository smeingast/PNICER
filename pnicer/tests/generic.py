def orion():

    """ This method goes through a typical PNICER session and creates an extinction map of Orion A from 2MASS data. """

    # Import
    from astropy.io import fits
    from astropy.coordinates import SkyCoord

    from pnicer import ApparentMagnitudes
    from pnicer.utils.auxiliary import get_resource_path

    # Find the test files
    science_path = get_resource_path(package="pnicer.tests_resources", resource="Orion_A_2mass.fits")
    control_path = get_resource_path(package="pnicer.tests_resources", resource="CF_2mass.fits")

    # Define feature names and extinction vector
    feature_names = ["Jmag", "Hmag", "Kmag"]
    error_names = ["e_Jmag", "e_Hmag", "e_Kmag"]
    feature_extinction = [2.5, 1.55, 1.0]

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

    # Initialize pnicer
    science = ApparentMagnitudes(magnitudes=sci_phot, errors=sci_err, extvec=feature_extinction, coordinates=sci_coo,
                                 names=feature_names)
    control = ApparentMagnitudes(magnitudes=con_phot, errors=con_err, extvec=feature_extinction, coordinates=con_coo,
                                 names=feature_names)

    # Test PNICER
    science.pnicer(control=control)
    pnicer = science.mag2color().pnicer(control=control.mag2color(), max_components=3)

    # Test NICER
    science.nicer(control=control)

    # Discretize extinction distributions from PNICER
    ext_pnicer = pnicer.discretize()

    # Make extinction map
    pnicer_emap = ext_pnicer.build_map(bandwidth=5 / 60, metric="gaussian", nicest=False, use_fwhm=True)

    # Plot the PNICER extinction map
    pnicer_emap.plot_map(figsize=10)

    """ If no errors pop up, then the basic PNICER package works """
    print("{0:<80s}".format("PNICER routines terminated successfully! Happy extinction mapping :) "))
