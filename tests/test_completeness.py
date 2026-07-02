import numpy as np
from scipy.special import erfc

from pnicer.completeness import CompletenessModel, _fit_band


def _draw_counts(rng, n, alpha, m50, width, m_min=8.0, m_max=20.0):
    """Rejection-sample magnitudes from N(m) ~ 10^(alpha m) * c(m)."""
    def density(m):
        return 10.0 ** (alpha * m) * 0.5 * erfc(
            (m - m50) / (np.sqrt(2) * width)
        )

    grid = np.linspace(m_min, m_max, 1000)
    peak = density(grid).max()
    out = []
    while len(out) < n:
        m = rng.uniform(m_min, m_max, size=4 * n)
        keep = rng.uniform(0, peak, size=4 * n) < density(m)
        out.extend(m[keep][: n - len(out)])
    return np.array(out)


class TestCompletenessFit:
    def test_recovers_injected_parameters(self, rng):
        mags = _draw_counts(rng, 30000, alpha=0.33, m50=16.5, width=0.4)
        fit = _fit_band(mags, bin_width=0.25)
        assert abs(fit.m50 - 16.5) < 0.1
        assert abs(fit.width - 0.4) < 0.15
        assert abs(fit.alpha - 0.33) < 0.03

    def test_model_fit_multi_band(self, rng):
        mags = np.column_stack(
            [
                _draw_counts(rng, 8000, 0.3, 16.8, 0.3),
                _draw_counts(rng, 8000, 0.33, 16.2, 0.35),
            ]
        )
        model = CompletenessModel.fit(mags, ("J", "H"))
        assert abs(model.bands[0].m50 - 16.8) < 0.15
        assert abs(model.bands[1].m50 - 16.2) < 0.15


class TestSurvival:
    def test_zero_extinction_is_unity(self):
        model = CompletenessModel.from_parameters(
            ("J", "Ks"), m50=np.array([17.0, 16.0]), width=np.array([0.3, 0.3])
        )
        mags = np.array([[14.0, 13.5], [16.5, 15.8]])
        observed = np.ones_like(mags, dtype=bool)
        s = model.survival(
            mags, observed, np.array([2.5, 1.0]), np.array([0.0])
        )
        np.testing.assert_allclose(s[:, 0], 1.0)

    def test_monotone_decreasing(self):
        model = CompletenessModel.from_parameters(
            ("J", "Ks"), m50=np.array([17.0, 16.0]), width=np.array([0.3, 0.3])
        )
        mags = np.array([[15.5, 15.0]])
        observed = np.ones_like(mags, dtype=bool)
        a_grid = np.linspace(0, 3, 13)
        s = model.survival(mags, observed, np.array([2.5, 1.0]), a_grid)[0]
        assert np.all(np.diff(s) <= 1e-12)
        assert s[-1] < 0.01  # deep extinction kills faint sources

    def test_bright_sources_survive(self):
        model = CompletenessModel.from_parameters(
            ("J", "Ks"), m50=np.array([17.0, 16.0]), width=np.array([0.3, 0.3])
        )
        mags = np.array([[10.0, 9.5]])
        observed = np.ones_like(mags, dtype=bool)
        s = model.survival(
            mags, observed, np.array([2.5, 1.0]), np.array([2.0])
        )
        assert s[0, 0] > 0.99

    def test_missing_bands_ignored(self):
        model = CompletenessModel.from_parameters(
            ("J", "Ks"), m50=np.array([17.0, 16.0]), width=np.array([0.3, 0.3])
        )
        # J is at the detection limit but unobserved -> must not contribute
        mags = np.array([[19.0, 12.0]])
        observed = np.array([[False, True]])
        s = model.survival(
            mags, observed, np.array([2.5, 1.0]), np.array([1.0])
        )
        assert s[0, 0] > 0.99
