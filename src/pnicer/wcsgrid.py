"""WCS scaffolding for map construction and small spherical-geometry helpers."""

from __future__ import annotations

import numpy as np
from astropy import wcs
from astropy.coordinates import ICRS, Galactic, SkyCoord
from astropy.io import fits

__all__ = ["centroid_sphere", "data2grid", "distance_sky", "skycoord2header"]


def distance_sky(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle (haversine) distance between positions, all in degrees."""
    l1, l2 = np.radians(lon1), np.radians(lon2)
    b1, b2 = np.radians(lat1), np.radians(lat2)
    dis = 2.0 * np.arcsin(
        np.sqrt(
            np.sin((b1 - b2) / 2.0) ** 2
            + np.cos(b1) * np.cos(b2) * np.sin((l1 - l2) / 2.0) ** 2
        )
    )
    return np.degrees(dis)


def centroid_sphere(skycoord: SkyCoord) -> SkyCoord:
    """Centroid of coordinates on the unit sphere."""
    good = np.isfinite(skycoord.spherical.lon) & np.isfinite(skycoord.spherical.lat)
    xyz = skycoord[good].cartesian.xyz.value
    mean = xyz.mean(axis=1)
    mean /= np.sqrt(np.sum(mean**2))
    lon = np.arctan2(mean[1], mean[0])
    if lon < 0:
        lon += 2.0 * np.pi
    lat = np.arcsin(mean[2])
    return SkyCoord(lon, lat, frame=skycoord.frame, unit="rad")


def skycoord2header(
    skycoord: SkyCoord,
    proj_code: str = "TAN",
    pixsize: float = 1 / 3600,
    rotation: float = 0.0,
    enlarge: float = 1.05,
    **kwargs,
) -> fits.Header:
    """Build a FITS WCS header enclosing the given coordinates.

    Parameters
    ----------
    skycoord : SkyCoord
        Coordinates to enclose.
    proj_code : str
        WCS projection code (e.g. "TAN", "AIT", "CAR").
    pixsize : float
        Pixel size in degrees.
    rotation : float
        Rotation angle of the projection in degrees.
    enlarge : float
        Field-size enlargement factor.
    **kwargs
        Additional header cards (e.g. ``pv2_1=-30``).
    """
    centroid = centroid_sphere(skycoord)
    separation = skycoord.separation(centroid)
    allsky = bool(np.nanmax(separation.degree) > 100)

    if allsky and proj_code not in ("AIT", "MOL", "CAR"):
        proj_code = "AIT"

    if isinstance(skycoord.frame, ICRS):
        ctype1, ctype2 = f"RA{proj_code:->6}", f"DEC{proj_code:->5}"
    elif isinstance(skycoord.frame, Galactic):
        ctype1, ctype2 = f"GLON{proj_code:->4}", f"GLAT{proj_code:->4}"
    else:
        raise ValueError(f"Frame {skycoord.frame.name} not supported")

    crval1 = 0.0 if allsky else centroid.spherical.lon.deg
    crval2 = 0.0 if allsky else centroid.spherical.lat.deg
    rot = np.deg2rad(rotation)

    header = fits.Header()
    header["NAXIS"] = 2
    header["CTYPE1"], header["CTYPE2"] = ctype1, ctype2
    header["CRVAL1"], header["CRVAL2"] = round(crval1, 7), round(crval2, 7)
    header["CUNIT1"] = header["CUNIT2"] = "deg"
    # Sign convention matching the legacy implementation (longitude
    # increases leftward); rotation only enters off-diagonal terms.
    header["CD1_1"] = round(-pixsize * np.cos(rot), 7)
    header["CD1_2"] = round(-pixsize * np.sin(rot), 7)
    header["CD2_1"] = round(-pixsize * np.sin(rot), 7)
    header["CD2_2"] = round(pixsize * np.cos(rot), 7)
    header["COORDSYS"] = skycoord.frame.name
    for key, value in kwargs.items():
        header[key.upper()] = value

    # Determine the extent of the data in this projection (finite
    # coordinates only — NaN positions are common in real catalogs)
    x, y = wcs.WCS(header).wcs_world2pix(
        skycoord.spherical.lon, skycoord.spherical.lat, 1
    )
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("No finite coordinates to build a WCS grid from")
    x, y = x[finite], y[finite]
    naxis1 = max(int(np.ceil(x.max() - np.floor(x.min())) * enlarge), 1)
    naxis2 = max(int(np.ceil(y.max() - np.floor(y.min())) * enlarge), 1)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    if allsky:
        header["CRPIX1"], header["CRPIX2"] = naxis1 / 2, naxis2 / 2
    else:
        header["CRPIX1"] = naxis1 / 2 - xdelta
        header["CRPIX2"] = naxis2 / 2 - ydelta
    return header


def data2grid(
    skycoord: SkyCoord,
    proj_code: str = "TAN",
    pixsize: float = 5.0 / 60,
    **kwargs,
) -> tuple[fits.Header, np.ndarray, np.ndarray]:
    """Build a WCS pixel grid covering the given coordinates.

    Returns
    -------
    header : fits.Header
    lon, lat : ndarray, shape (NAXIS2, NAXIS1)
        World coordinates of the pixel centers, in degrees.
    """
    header = skycoord2header(
        skycoord=skycoord, proj_code=proj_code, pixsize=pixsize, **kwargs
    )
    grid_wcs = wcs.WCS(header=header)
    xv, yv = np.meshgrid(np.arange(header["NAXIS1"]), np.arange(header["NAXIS2"]))
    lon, lat = grid_wcs.wcs_pix2world(xv, yv, 0)
    return header, lon, lat
