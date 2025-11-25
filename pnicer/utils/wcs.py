# -----------------------------------------------------------------------------
# Import packages
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from pnicer.utils.algebra import centroid_sphere
from astropy.coordinates import Galactic, ICRS, BaseCoordinateFrame


def make_wcs_image_header(
    ctype1, ctype2, crval1, crval2, pixsize, frame, rotation=0, allsky=False, **kwargs
) -> fits.Header:
    # Build additional string
    additional = ""
    for key, value in kwargs.items():
        additional += "{0: <8}= {1}\n".format(key.upper(), value)

    if allsky:
        crval1 = crval2 = 0.0

    if isinstance(frame, BaseCoordinateFrame):
        frame = frame.name

    # Compute CD matrix
    cdelt1 = -pixsize
    cdelt2 = pixsize
    cd1_1 = cdelt1 * np.cos(np.deg2rad(rotation))
    cd1_2 = -cdelt2 * np.sin(np.deg2rad(rotation))
    cd2_1 = cdelt1 * np.sin(np.deg2rad(rotation))
    cd2_2 = cdelt2 * np.cos(np.deg2rad(rotation))

    # Create preliminary header without size information
    header = fits.Header.fromstring(
        f"NAXIS   = 2\n"
        f"CTYPE1  = '{ctype1}'\n"
        f"CTYPE2  = '{ctype2}'\n"
        f"CRVAL1  = {crval1:0.7f}\n"
        f"CRVAL2  = {crval2:0.7f}\n"
        f"CUNIT1  = 'deg'\n"
        f"CUNIT2  = 'deg'\n"
        f"CD1_1   = {cd1_1:0.7f}\n"
        f"CD1_2   = {cd1_2:0.7f}\n"
        f"CD2_1   = {cd2_1:0.7f}\n"
        f"CD2_2   = {cd2_2:0.7f}\n"
        f"COORDSYS= '{frame}'\n"
        f"{additional}",
        sep="\n",
    )

    return header


def skycoord2header(
    skycoord,
    proj_code="TAN",
    pixsize=1 / 3600,
    rotation=0,
    enlarge=1.05,
    silent=True,
    **kwargs,
):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude).
    The world coordinate system can be chosen between galactic and equatorial;
    all WCS projections are supported. Very useful for creating a quick WCS
    to plot data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance containg the coordinates.
    proj_code : str, optional
        Projection code. (e.g. 'TAN', 'AIT', 'CAR', etc). Default is 'TAN'
    pixsize : int, float, optional
        Pixel size of grid. Default is 1 arcsec.
    rotation : int, float, optional
        Rotation angle of the projection in degrees. Default is 0.
    enlarge : float, optional
        Optional enlargement factor for calculated field size. Default is 1.05.
        Set to 1 if no enlargement is wanted.
    silent : bool, optional
        If False, print some messages when applicable.
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    skycoord_centroid = centroid_sphere(skycoord=skycoord)

    # Determine if allsky should be forced
    sep = skycoord.separation(skycoord_centroid)
    allsky = True if np.max(sep.degree) > 100 else False

    # Issue warning
    if silent is False:
        if allsky:
            print("Warning. Using allsky projection!")

    # Override projection with allsky data
    if allsky:
        if proj_code not in ["AIT", "MOL", "CAR"]:
            proj_code = "AIT"

    # Projection code
    if isinstance(skycoord.frame, ICRS):
        ctype1 = "RA{:->6}".format(proj_code)
        ctype2 = "DEC{:->5}".format(proj_code)
    elif isinstance(skycoord.frame, Galactic):
        ctype1 = "GLON{:->4}".format(proj_code)
        ctype2 = "GLAT{:->4}".format(proj_code)
    else:
        raise ValueError("Frame {0:s} not supported".format(skycoord.frame))

    # Build basic header with WCS information
    header = make_wcs_image_header(
        ctype1=ctype1,
        ctype2=ctype2,
        crval1=skycoord_centroid.spherical.lon.deg,
        crval2=skycoord_centroid.spherical.lat.deg,
        pixsize=pixsize,
        frame=skycoord_centroid.frame,
        rotation=rotation,
        allsky=allsky,
        **kwargs,
    )

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).wcs_world2pix(
        skycoord.spherical.lon, skycoord.spherical.lat, 1
    )

    naxis1 = (np.ceil((x.max()) - np.floor(x.min())) * enlarge).astype(int)
    naxis2 = (np.ceil((y.max()) - np.floor(y.min())) * enlarge).astype(int)

    # Calculate pixel shift relative to centroid
    # (caused by anisotropic distribution of sources)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    # Add size to header
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    if allsky:
        header["CRPIX1"], header["CRPIX2"] = naxis1 / 2, naxis2 / 2
    else:
        header["CRPIX1"], header["CRPIX2"] = naxis1 / 2 - xdelta, naxis2 / 2 - ydelta

    # Add allky keyword
    header["ALLSKY"] = allsky

    # Return Header
    return header


def data2grid(
    skycoord, proj_code="TAN", pixsize=5.0 / 60, return_skycoord=False, **kwargs
):
    """
    Method to build a WCS grid with a valid projection given a pixel scale.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance containg the coordinates.
    proj_code : str, optional
        Any WCS projection code (e.g. CAR, TAN, etc.). Default is 'TAN'.
    pixsize : int, float, optional
        Pixel size of grid. Default is 10 arcminutes.
    return_skycoord : bool, optional
        Whether to return the grid coordinates as a SkyCoord object. Default is False
    kwargs
        Additional projection parameters if required (e.g. pv2_1=-30, pv2_2=0 for a
        given COE projection)

    Returns
    -------
    tuple
        Tuple containing the header and the world coordinate grids (lon and lat)

    """

    # Create header from data
    header = skycoord2header(
        skycoord=skycoord, proj_code=proj_code, pixsize=pixsize, **kwargs
    )

    # Get WCS
    mywcs = wcs.WCS(header=header)

    # Create image coordinate grid
    image_grid = np.meshgrid(
        np.arange(0, header["NAXIS1"], 1), np.arange(0, header["NAXIS2"], 1)
    )

    # Convert to world coordinates and get WCS grid for this projection
    world_grid = mywcs.wcs_pix2world(image_grid[0], image_grid[1], 0)

    # Convert to SkyCoord instance if set
    if return_skycoord:
        world_grid = SkyCoord(*world_grid, frame=skycoord.frame, unit="degree")

    # Return header and grid
    return header, world_grid
