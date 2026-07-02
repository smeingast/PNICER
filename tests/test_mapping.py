import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from pnicer import ExtinctionCatalog, ExtinctionMap
from pnicer.wcsgrid import distance_sky

STD2FWHM = 2 * np.sqrt(2 * np.log(2))


def _small_catalog(rng, n=400, center=(210.0, -19.0), spread=0.3):
    lon = center[0] + rng.uniform(-spread, spread, n)
    lat = center[1] + rng.uniform(-spread, spread, n)
    coords = SkyCoord(l=lon, b=lat, frame="galactic", unit="deg")
    ext = rng.normal(0.5, 0.2, n)
    var = rng.uniform(0.01, 0.05, n)
    return ExtinctionCatalog(
        extinction=ext,
        variance=var,
        coordinates=coords,
        extinction_vector=np.array([2.5, 1.55, 1.0]),
    )


def _brute_force_pixel(catalog, plon, plat, bandwidth, nicest=False, alpha=1 / 3):
    """Independent reference computation for one gaussian-metric pixel."""
    r_mask = 3 * bandwidth
    lon = catalog.coordinates.spherical.lon.degree
    lat = catalog.coordinates.spherical.lat.degree
    dis = distance_sky(plon, plat, lon, lat)
    sel = dis <= r_mask
    grid = np.arange(-100, 100, 0.01)
    norm = np.trapezoid(np.exp(-0.5 * (grid / bandwidth) ** 2), grid)
    ws = np.exp(-0.5 * (dis[sel] / bandwidth) ** 2) / norm
    e, v = catalog.extinction[sel], catalog.variance[sel]
    wt = ws / v
    ext0 = np.nansum(wt * e) / np.nansum(wt)
    clip = np.abs(e - ext0) > 3 * np.nanstd(e)
    e, v, ws, wt = (np.where(clip, np.nan, x) for x in (e, v, ws, wt))
    if nicest:
        beta = np.log(10) * alpha * 2.5
        boost = 10 ** (alpha * 2.5 * e)
        ws, wt = ws * boost, wt * boost
        cor = np.nansum(wt * v) / np.nansum(wt)
        var = np.nansum(wt**2 * np.exp(2 * beta * e) * (1 + beta * e) ** 2 / v)
        var /= np.nansum(wt * np.exp(beta * e) / v) ** 2
        ext = np.nansum(wt * e) / np.nansum(wt) - cor
    else:
        ext = np.nansum(wt * e) / np.nansum(wt)
        var = np.nansum(wt**2 * v) / np.nansum(wt) ** 2
    return ext, var, np.sum(np.isfinite(wt * e)), np.nansum(ws)


class TestMapMath:
    @pytest.mark.parametrize("nicest", [False, True])
    def test_pixels_match_brute_force(self, rng, nicest):
        catalog = _small_catalog(rng)
        bw = 6 / 60
        emap = catalog.build_map(bandwidth=bw, metric="gaussian", nicest=nicest)
        from astropy.wcs import WCS

        wcs = WCS(emap.map_header)
        ny, nx = emap.shape
        for iy, ix in [(ny // 2, nx // 2), (ny // 3, 2 * nx // 3), (2, 2)]:
            plon, plat = wcs.wcs_pix2world([[ix, iy]], 0)[0]
            ext, var, num, rho = _brute_force_pixel(
                catalog, plon, plat, bw, nicest=nicest
            )
            if np.isnan(ext):
                assert np.isnan(emap.map_ext[iy, ix])
                continue
            assert np.isclose(emap.map_ext[iy, ix], ext, atol=1e-10)
            assert np.isclose(emap.map_var[iy, ix], var, atol=1e-12)
            assert emap.map_num[iy, ix] == num
            assert np.isclose(emap.map_rho[iy, ix], rho, atol=1e-10)

    def test_fwhm_scaling(self, rng):
        catalog = _small_catalog(rng)
        fwhm_map = catalog.build_map(
            bandwidth=5 / 60, metric="gaussian", use_fwhm=True
        )
        assert np.isclose(
            fwhm_map.prime_header["BWIDTH"], (5 / 60) / STD2FWHM
        )

    def test_average_and_median_metrics(self, rng):
        catalog = _small_catalog(rng)
        for metric in ("average", "median"):
            emap = catalog.build_map(bandwidth=6 / 60, metric=metric)
            finite = np.isfinite(emap.map_ext)
            assert finite.any()
            # Both are location estimates of a 0.5-mean field
            assert abs(np.nanmedian(emap.map_ext) - 0.5) < 0.1

    def test_quantity_bandwidth(self, rng):
        from astropy import units as u

        catalog = _small_catalog(rng)
        a = catalog.build_map(bandwidth=6 / 60, metric="gaussian")
        b = catalog.build_map(bandwidth=(6 / 60) * u.deg, metric="gaussian")
        np.testing.assert_array_equal(a.map_ext, b.map_ext)

    def test_requires_coordinates(self, rng):
        catalog = ExtinctionCatalog(
            extinction=np.array([0.5]), variance=np.array([0.01])
        )
        with pytest.raises(ValueError, match="coordinates"):
            catalog.build_map(bandwidth=0.1)


class TestFitsRoundTrip:
    def test_save_and_load(self, rng, tmp_path):
        emap = _small_catalog(rng).build_map(bandwidth=6 / 60)
        path = str(tmp_path / "map.fits")
        emap.save(path)
        loaded = ExtinctionMap.from_fits(path)
        np.testing.assert_array_equal(loaded.map_ext, emap.map_ext)
        np.testing.assert_array_equal(loaded.map_var, emap.map_var)
        np.testing.assert_array_equal(loaded.map_num, emap.map_num)
        assert loaded.prime_header["METRIC"] == "gaussian"
        assert loaded.prime_header["NICEST"] is False


class TestLegacyRegression:
    """Maps against the frozen legacy baselines.

    Differences are expected and bounded: legacy capped neighbors with a
    k-NN heuristic (its own TODO flagged sources beyond the cap); the new
    code uses an exact radius query. Central values agree to numerical
    precision; the tail reflects legacy's missing neighbors in dense pixels.
    """

    def test_nicer_map(self, orion, control, baseline_dir):
        from astropy.io import fits

        emap = orion.nicer(control).build_map(
            bandwidth=5 / 60, metric="gaussian", use_fwhm=True
        )
        with fits.open(baseline_dir / "map_nicer_gauss.fits") as hdul:
            legacy_ext = hdul[1].data
        assert emap.shape == legacy_ext.shape
        both = np.isfinite(emap.map_ext) & np.isfinite(legacy_ext)
        diff = np.abs(emap.map_ext[both] - legacy_ext[both])
        assert np.median(diff) < 1e-4
        assert np.percentile(diff, 95) < 0.02
        assert np.mean(diff < 0.05) > 0.99
