# -----------------------------------------------------------------------------
# Import packages
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from pnicer.utils.algebra import centroid_sphere


# -----------------------------------------------------------------------------
def data2header(lon, lat, frame, proj_code="TAN", pixsize=1/3600, enlarge=1.05, **kwargs):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude). The world coordinate system can be
    chosen between galactic and equatorial; all WCS projections are supported. Very useful for creating a quick WCS
    to plot data.

    Parameters
    ----------
    lon : list, np.array
        Input list or array of longitude coordinates in degrees.
    lat : list, np.array
        Input list or array of latitude coordinates in degrees.
    frame : str, optional
        World coordinate system frame of input data ('icrs' or 'galactic').
    proj_code : str, optional
        Projection code. (e.g. 'TAN', 'AIT', 'CAR', etc). Default is 'TAN'
    pixsize : int, float, optional
        Pixel size of generated header in degrees. Not so important for plots, but still required.
    enlarge : float, optional
        Optional enlargement factor for calculated field size. Default is 1.05. Set to 1 if no enlargement is wanted.
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    crval1, crval2 = centroid_sphere(lon=lon, lat=lat, units="degree")

    # Projection code
    if frame.lower() == "icrs":
        ctype1 = "RA{:->6}".format(proj_code)
        ctype2 = "DEC{:->5}".format(proj_code)
        frame = "equ"
    elif frame.lower() == "galactic":
        ctype1 = "GLON{:->4}".format(proj_code)
        ctype2 = "GLAT{:->4}".format(proj_code)
        frame = "gal"
    else:
        raise ValueError("Projection system {0:s} not supported".format(frame))

    # Build additional string
    additional = ""
    for key, value in kwargs.items():
        additional += ("{0: <8}= {1}\n".format(key.upper(), value))

    # Create preliminary header without size information
    header = fits.Header.fromstring("NAXIS   = 2" + "\n"
                                    "CTYPE1  = '" + ctype1 + "'\n"
                                    "CTYPE2  = '" + ctype2 + "'\n"
                                    "CRVAL1  = " + str(crval1) + "\n"
                                    "CRVAL2  = " + str(crval2) + "\n"
                                    "CUNIT1  = 'deg'" + "\n"
                                    "CUNIT2  = 'deg'" + "\n"
                                    "CDELT1  = -" + str(pixsize) + "\n"
                                    "CDELT2  = " + str(pixsize) + "\n"
                                    "COORDSYS= '" + frame + "'" + "\n" +
                                    additional,
                                    sep="\n")

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).wcs_world2pix(lon, lat, 1)
    naxis1 = (np.ceil((x.max()) - np.floor(x.min())) * enlarge).astype(int)
    naxis2 = (np.ceil((y.max()) - np.floor(y.min())) * enlarge).astype(int)

    # Calculate pixel shift relative to centroid (caused by anisotropic distribution of sources)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    # Add size to header
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    header["CRPIX1"], header["CRPIX2"] = naxis1 / 2 - xdelta, naxis2 / 2 - ydelta

    # Return Header
    return header


# -----------------------------------------------------------------------------
def data2grid(lon, lat, frame, proj_code="TAN", pixsize=5. / 60, return_skycoord=False, **kwargs):
    """
    Method to build a WCS grid with a valid projection given a pixel scale.

    Parameters
    ----------
    lon : list, np.array
        Input list or array of longitude coordinates in degrees.
    lat : list, np.array
        Input list or array of latitude coordinates in degrees.
    frame : str, optional
        World coordinate system frame of input data ('icrs' or 'galactic').
    proj_code : str, optional
        Any WCS projection code (e.g. CAR, TAN, etc.). Default is 'TAN'.
    pixsize : int, float, optional
        Pixel size of grid. Default is 10 arcminutes.
    return_skycoord : bool, optional
        Whether to return the grid coordinates as a SkyCoord object. Default is False
    kwargs
        Additional projection parameters if required (e.g. pv2_1=-30, pv2_2=0 for a given COE projection)

    Returns
    -------
    tuple
        Tuple containing the header and the world coordinate grids (lon and lat)

    """

    # Create header from data
    header = data2header(lon=lon, lat=lat, frame=frame, proj_code=proj_code, pixsize=pixsize, **kwargs)

    # Get WCS
    mywcs = wcs.WCS(header=header)

    # Create image coordinate grid
    image_grid = np.meshgrid(np.arange(0, header["NAXIS1"], 1), np.arange(0, header["NAXIS2"], 1))

    # Convert to world coordinates and get WCS grid for this projection
    world_grid = mywcs.wcs_pix2world(image_grid[0], image_grid[1], 0)

    # Convert to SkyCoord instance if set
    if return_skycoord:
        world_grid = SkyCoord(*world_grid, frame=frame, unit="degree")

    # Return header and grid
    return header, world_grid
