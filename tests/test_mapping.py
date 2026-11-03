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
        beta = np.log(10) * alpha * 1.0  # default nicest_k = 1.0
        boost = 10 ** (alpha * 1.0 * e)
        ws, wt = ws * boost, wt * boost
        cor = beta * np.nansum(wt * v) / np.nansum(wt)
        ext = np.nansum(wt * e) / np.nansum(wt) - cor
        # Propagated error: dA/de_n = p_n [1 + beta (e_n - A0) - beta^2 (v_n - vbar)]
        p = wt / np.nansum(wt)
        a0, vbar = np.nansum(p * e), np.nansum(p * v)
        grad = p * (1 + beta * (e - a0) - beta**2 * (v - vbar))
        var = np.nansum(grad**2 * v)
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
        fwhm_map = catalog.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True)
        assert np.isclose(fwhm_map.prime_header["BWIDTH"], (5 / 60) / STD2FWHM)

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

    def test_numpy_integer_sampling(self, rng):
        catalog = _small_catalog(rng, n=50)
        emap = catalog.build_map(bandwidth=6 / 60, sampling=np.int64(2))
        assert emap.shape[0] > 0

    def test_single_source_catalog(self, rng):
        """Degenerate extent must not crash (empty-bincount dtype, NAXIS>=1)."""
        catalog = ExtinctionCatalog(
            extinction=np.array([0.5]),
            variance=np.array([0.01]),
            coordinates=SkyCoord(l=[210.0], b=[-19.0], frame="galactic", unit="deg"),
        )
        emap = catalog.build_map(bandwidth=6 / 60, metric="gaussian")
        assert np.isfinite(emap.map_ext).sum() >= 1

    def test_nan_coordinates_tolerated(self, rng):
        catalog = _small_catalog(rng, n=100)
        lon = catalog.coordinates.spherical.lon.degree.copy()
        lat = catalog.coordinates.spherical.lat.degree.copy()
        lon[0] = np.nan
        lat[0] = np.nan
        bad_coord = ExtinctionCatalog(
            extinction=catalog.extinction,
            variance=catalog.variance,
            coordinates=SkyCoord(l=lon, b=lat, frame="galactic", unit="deg"),
            extinction_vector=catalog.extinction_vector,
        )
        emap = bad_coord.build_map(bandwidth=6 / 60, metric="gaussian")
        assert np.isfinite(emap.map_ext).any()


class TestNicestVariance:
    """The NICEST map variance is the propagated error of the estimator."""

    def test_alpha_zero_equals_nicer(self, rng):
        catalog = _small_catalog(rng)
        plain = catalog.build_map(bandwidth=6 / 60, nicest=False)
        nicest = catalog.build_map(bandwidth=6 / 60, nicest=True, alpha=0.0)
        np.testing.assert_allclose(nicest.map_ext, plain.map_ext, rtol=1e-12)
        np.testing.assert_allclose(nicest.map_var, plain.map_var, rtol=1e-12)

    def test_matches_finite_differences(self, rng):
        """Numerical gradient of the map value w.r.t. each input extinction."""
        n = 40
        coords = SkyCoord(
            l=210 + rng.normal(0, 0.003, n),
            b=-19 + rng.normal(0, 0.003, n),
            frame="galactic",
            unit="deg",
        )
        ext = rng.normal(1.5, 0.2, n)
        var = rng.uniform(0.01, 0.2, n)

        def build(e):
            return ExtinctionCatalog(e, var, coords).build_map(
                bandwidth=0.02, nicest=True, alpha=0.33
            )

        emap = build(ext)
        pix = np.unravel_index(np.argmax(emap.map_num), emap.shape)
        assert emap.map_num[pix] == n
        h = 1e-6
        grad = np.array(
            [
                (
                    build(ext + h * np.eye(n)[i]).map_ext[pix]
                    - build(ext - h * np.eye(n)[i]).map_ext[pix]
                )
                / (2 * h)
                for i in range(n)
            ]
        )
        assert np.isclose(emap.map_var[pix], np.sum(grad**2 * var), rtol=1e-6)

    def test_monte_carlo(self):
        """Scatter of maps from resampled inputs matches the variance the maps
        report (evaluated at the noisy inputs, as users obtain it)."""
        rng = np.random.default_rng(42)
        n = 3000
        coords = SkyCoord(
            l=210 + rng.uniform(-0.3, 0.3, n),
            b=-19 + rng.uniform(-0.3, 0.3, n),
            frame="galactic",
            unit="deg",
        )
        var = rng.uniform(0.01, 0.09, n)
        maps = [
            ExtinctionCatalog(1 + rng.normal(0, np.sqrt(var)), var, coords).build_map(
                bandwidth=0.1, nicest=True, alpha=0.33
            )
            for _ in range(150)
        ]
        scatter = np.nanvar([m.map_ext for m in maps], axis=0)
        reported = np.nanmean([m.map_var for m in maps], axis=0)
        assert 0.8 < np.nanmedian(scatter / reported) < 1.2


class TestDensityPlane:
    @pytest.mark.parametrize("nicest", [False, True])
    def test_sources_without_estimate_excluded(self, rng, nicest):
        """A NaN extinction with finite variance must behave like a missing
        source: no contribution to rho, the weight sum, or the counts."""
        catalog = _small_catalog(rng)
        ext = catalog.extinction.copy()
        ext[::3] = np.nan  # variance stays finite on purpose
        with_nan = ExtinctionCatalog(ext, catalog.variance, catalog.coordinates)
        var = np.where(np.isfinite(ext), catalog.variance, np.nan)
        both_nan = ExtinctionCatalog(ext, var, catalog.coordinates)
        kw = {"bandwidth": 6 / 60, "nicest": nicest}
        a, b = with_nan.build_map(**kw), both_nan.build_map(**kw)
        np.testing.assert_allclose(a.map_ext, b.map_ext, rtol=1e-12)
        np.testing.assert_allclose(a.map_var, b.map_var, rtol=1e-12)
        np.testing.assert_allclose(a.map_rho, b.map_rho, rtol=1e-12)
        np.testing.assert_array_equal(a.map_num, b.map_num)


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

    def test_nicest_map_from_legacy_catalog(self, orion, baseline_dir):
        """NICEST correction against the legacy map, using the frozen legacy
        per-source estimates as input so the correction math is isolated
        (this is the test that catches a missing beta factor)."""
        from astropy.io import fits

        base = np.load(baseline_dir / "pnicer_color_science_run1.npz")
        catalog = ExtinctionCatalog(
            extinction=base["extinction"].astype(np.float64),
            variance=base["variance"].astype(np.float64),
            coordinates=orion.coordinates,
            extinction_vector=orion.extinction_vector,
        )
        # Legacy derived k implicitly from the color reddening vector
        # (max([0.95, 0.55]) for the 2017 setup); pass it explicitly
        emap = catalog.build_map(
            bandwidth=5 / 60,
            metric="gaussian",
            use_fwhm=True,
            nicest=True,
            nicest_k=0.95,
        )
        with fits.open(baseline_dir / "map_pnicer_gauss_nicest.fits") as hdul:
            legacy_ext = hdul[1].data
        both = np.isfinite(emap.map_ext) & np.isfinite(legacy_ext)
        diff = np.abs(emap.map_ext[both] - legacy_ext[both])
        assert np.median(diff) < 1e-3
        assert np.percentile(diff, 95) < 0.02
