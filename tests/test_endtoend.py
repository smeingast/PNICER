"""End-to-end pipeline tests on the bundled Orion A data."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def orion_model(control):
    return control.fit_intrinsic_colors(n_components=5, random_state=0)


class TestOrionPipeline:
    def test_model_persistence(self, orion_model, tmp_path):
        from pnicer import IntrinsicColorModel

        path = str(tmp_path / "model.npz")
        orion_model.save(path)
        loaded = IntrinsicColorModel.load(path)
        np.testing.assert_allclose(loaded.weights, orion_model.weights)
        np.testing.assert_allclose(loaded.means, orion_model.means)
        assert loaded.supports_adaptive == orion_model.supports_adaptive

    def test_completeness_matches_2mass(self, orion_model):
        """Fitted 50% limits agree with the known 2MASS depths."""
        m50 = [b.m50 for b in orion_model.completeness.bands]
        assert 16.4 < m50[0] < 17.4  # J
        assert 15.8 < m50[1] < 16.8  # H
        assert 15.2 < m50[2] < 16.2  # Ks

    def test_control_field_self_test(self, orion_model, control):
        """Scientific invariant: extinction-free field peaks at A ~ 0,
        sharper than NICER (Meingast+ 2017, Fig. 4)."""
        post = orion_model.posterior(control)
        cat = post.discretize()
        assert abs(np.nanmedian(cat.extinction)) < 0.03
        nicer = control.nicer(control)
        both = np.isfinite(cat.extinction) & np.isfinite(nicer.extinction)
        assert np.nanstd(cat.extinction[both]) <= np.nanstd(nicer.extinction[both])

    def test_science_field(self, orion_model, orion):
        post = orion_model.posterior(orion)
        cat = post.discretize()
        n_finite = np.isfinite(cat.extinction).sum()
        assert n_finite / orion.n_sources > 0.9
        # Orion A has substantial extinction
        assert 0.15 < np.nanmean(cat.extinction) < 0.5

    def test_agrees_with_legacy_pnicer(self, orion_model, orion, baseline_dir):
        """Object-to-object agreement with legacy PNICER (L18 Fig. 13
        analogue): strongly correlated, no systematic offset."""
        base = np.load(baseline_dir / "pnicer_color_science_run1.npz")
        cat = orion_model.posterior(orion).discretize()
        both = np.isfinite(cat.extinction) & np.isfinite(base["extinction"])
        assert both.sum() > 80000
        corr = np.corrcoef(cat.extinction[both], base["extinction"][both])[0, 1]
        assert corr > 0.97
        offset = np.median(cat.extinction[both] - base["extinction"][both])
        assert abs(offset) < 0.02

    def test_adaptive_runs_on_orion(self, orion_model, orion):
        post = orion_model.posterior(orion, adaptive=True)
        cat = post.discretize()
        assert np.isfinite(cat.extinction).sum() > 80000

    def test_map_pipeline(self, orion_model, orion, tmp_path):
        cat = orion_model.posterior(orion).discretize()
        emap = cat.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True)
        assert np.nanmax(emap.map_ext) > 1.0  # the cloud is there
        assert abs(np.nanmedian(emap.map_ext)) < 0.4
        emap.save(str(tmp_path / "orion.fits"))


class TestPerformance:
    def test_million_source_inference(self, orion_model, rng):
        """De-reddening 10^6 sources stays within seconds (2017 ethos)."""
        import time

        from pnicer import Photometry

        n = 1_000_000
        mags = {
            "J": rng.normal(15.0, 1.0, n),
            "H": rng.normal(14.4, 1.0, n),
            "Ks": rng.normal(14.1, 1.0, n),
        }
        errs = {b: np.full(n, 0.05) for b in ("J", "H", "Ks")}
        phot = Photometry(
            magnitudes=mags,
            errors=errs,
            extinction={"J": 2.5, "H": 1.55, "Ks": 1.0},
        )
        start = time.perf_counter()
        post = orion_model.posterior(phot)
        elapsed = time.perf_counter() - start
        assert np.isfinite(post.mean()).all()
        assert elapsed < 30.0
