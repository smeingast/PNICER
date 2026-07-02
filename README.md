<h1 align="center">PNICER</h1>

<p align="center"><i>Extinction estimates and maps from photometric catalogs.</i></p>

<p align="center">
  <a href="https://github.com/smeingast/PNICER/actions/workflows/ci.yml"><img src="https://github.com/smeingast/PNICER/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License"></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract"><img src="https://img.shields.io/badge/A%26A-2017%2C%20601%2C%20A137-orange" alt="Paper"></a>
</p>

![Orion A extinction map](assets/orion.png)

Interstellar dust dims and reddens the light of every star behind it. PNICER measures this extinction from photometric catalogs: each source receives a probability density for its line-of-sight extinction, and the individual measurements are subsequently combined into smooth extinction maps. The only calibration input is an extinction-free control field; the method neither requires a prior on the column-density distribution nor makes assumptions on the composition of the background population.

> [!IMPORTANT]
> This repository is linked from the published paper [Meingast, Lombardi & Alves (2017), A&A 601, A137](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract), for which PNICER was originally developed. If you are looking for the software exactly as described and used in that paper, install the [v1.0 release](https://github.com/smeingast/PNICER/releases/tag/v1.0), which preserves the original implementation unchanged. The version on this branch (2.0) is an updated package that follows the same philosophy, namely probabilistic, purely data-driven extinction measurements calibrated on a control field, but replaces the internals and the API and improves on the original in accuracy, speed, and reliability (see [Method](#method) and [Verification](#verification)).

## Key features

- Each source receives a full posterior probability density for its extinction in the form of an analytic Gaussian mixture, which can be collapsed to a point estimate with an associated uncertainty where required.
- The intrinsic color distribution of the control field is fitted with extreme deconvolution (Bovy et al. 2011), which removes the photometric errors of the control field itself from the calibration. In our validation tests, the reported uncertainties are accurate to on the order of 0.1%.
- Extinction alters the observable background population, as intrinsically faint galaxies drop out of a magnitude-limited catalog long before stars do. The adaptive correction following Lombardi (2018) models this selection effect; in simulations with known ground truth, the remaining bias amounts to less than 0.02 mag at A_K = 2 mag, where the classic estimators deviate by on the order of 0.1 mag.
- Missing photometry is treated exactly by means of projection matrices, and two observed passbands are sufficient for an estimate.
- The NICER technique (Lombardi & Alves 2001) and the NICEST correction for extinction maps (Lombardi 2009) are provided within the same interface.
- The implementation relies exclusively on numpy and scipy, processes on the order of one million sources per second on a laptop, and is fully reproducible when supplied with a random seed. In the absence of any multiprocessing, PNICER also runs on Windows.

## Quick start

```python
from pnicer import Photometry

bands = {"J": ("Jmag", "e_Jmag"), "H": ("Hmag", "e_Hmag"), "Ks": ("Kmag", "e_Kmag")}
extinction = {"J": 2.5, "H": 1.55, "Ks": 1.0}   # A_band / A_Ks

science = Photometry.from_fits("orion.fits", bands=bands, extinction=extinction,
                               lon="GLON", lat="GLAT", frame="galactic")
control = Photometry.from_fits("control.fits", bands=bands, extinction=extinction,
                               lon="GLON", lat="GLAT", frame="galactic")

# Fit the intrinsic color model once, then reuse it
model = control.fit_intrinsic_colors(random_state=0)

# Extinction posteriors for every source, with the adaptive population correction
posterior = science.pnicer(model, adaptive=True)

# Point estimates, then a smoothed map written to FITS
catalog = posterior.discretize()
emap = catalog.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True)
emap.save("orion_ak.fits")
emap.plot()

# The classic NICER estimator, same interface
nicer = science.nicer(control)
```

The extinction map displayed at the top of this page is built from the bundled 2MASS demonstration data:

```python
from pnicer.demo import orion
orion()
```

## Advanced usage

Beyond the basic pipeline, the example below demonstrates model selection and persistence, direct access to the posterior densities, outlier identification through the model evidence, manually supplied completeness functions for the adaptive correction, and the available map variants.

```python
import numpy as np
from pnicer import IntrinsicColorModel
from pnicer.completeness import CompletenessModel
from pnicer.demo import load_control, load_orion

science, control = load_orion(), load_control()

# Select the number of mixture components via the Bayesian information
# criterion instead of the fixed default, then store the model for reuse
model = control.fit_intrinsic_colors(n_components="bic", max_components=6,
                                     random_state=0)
model.save("intrinsic_colors.npz")
model = IntrinsicColorModel.load("intrinsic_colors.npz")

# The posteriors are analytic Gaussian mixtures; evaluate the densities on
# an arbitrary extinction grid, for instance for plotting or resampling
posterior = science.pnicer(model)
a_grid = np.linspace(-1, 3, 401)
pdfs = posterior.pdf(a_grid)                 # shape (n_sources, 401)

# The log-evidence identifies sources that are poorly described by the
# control field, such as young stellar objects. Evidence values are only
# comparable among sources sharing one missingness pattern.
complete = posterior.pattern_dim == 2
threshold = np.nanpercentile(posterior.log_evidence[complete], 1)
outliers = complete & (posterior.log_evidence < threshold)

# When the survey completeness is known, supply it directly instead of
# fitting it to the control-field number counts
completeness = CompletenessModel.from_parameters(
    band_names=("J", "H", "Ks"),
    m50=np.array([16.9, 16.2, 15.7]),        # 50% completeness limits (mag)
    width=np.array([0.3, 0.4, 0.55]),
)
model = control.fit_intrinsic_colors(random_state=0, completeness=completeness)
posterior = science.pnicer(model, adaptive=True)

# Maps support several smoothing metrics as well as the NICEST correction
# for unresolved substructure and foreground contamination
catalog = posterior.discretize()
nicest = catalog.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True,
                           nicest=True, alpha=1 / 3)
median = catalog.build_map(bandwidth=5 / 60, metric="median")

# All products round-trip through FITS; catalogs export to astropy tables
nicest.save("orion_ak_nicest.fits")
table = catalog.to_table()
```

## Installation

```bash
pip install git+https://github.com/smeingast/PNICER.git
```

Python 3.11 or newer is required; numpy, scipy, astropy, and scikit-learn are installed automatically. Plotting requires the optional extra: `pip install "pnicer[plot] @ git+https://github.com/smeingast/PNICER.git"`.

## Method

Extinction measurements from photometry rest on the color excess: dust reddens the observed colors of a background source along a known reddening vector, and the displacement from the intrinsic colors measures the extinction. The intrinsic colors of an individual source are unknown, but their distribution can be measured in a nearby extinction-free control field. PNICER turns this into a fully probabilistic estimate in two steps.

In the first step, the intrinsic color distribution of the control field is modeled as a Gaussian mixture in the full color space. The fit uses extreme deconvolution ([Bovy et al. 2011](https://ui.adsabs.harvard.edu/abs/2011AnApS...5.1657B/abstract)), an expectation-maximization algorithm that treats every source with its individual error covariance, so the resulting mixture describes the error-free distribution rather than the observed, noise-broadened one. Sources with incomplete photometry still contribute through per-pattern projection matrices. The number of mixture components is either fixed (five by default, following Lombardi 2018) or selected with the Bayesian information criterion.

In the second step, each science source receives its extinction posterior in closed form, following [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract): given the observed colors, their error covariance, and the reddening vector, every mixture component contributes one Gaussian in extinction, with analytically computed mean, width, and amplitude. No fitting, sampling, or gridding takes place per source, which is why the inference runs at about one million sources per second. The accompanying model evidence measures how well a source is described by the control field and can be used to identify objects with peculiar colors. The classic NICER estimator corresponds exactly to the single-component limit of this formalism, a property the test suite verifies to machine precision.

The optional adaptive correction addresses a selection effect that the two steps above ignore: extinction pushes intrinsically faint sources beyond the detection limit, so the population observed behind a cloud differs from the control field, with galaxies disappearing first. PNICER fits completeness functions to the control-field number counts (or accepts them as survey parameters), derives extinction-dependent mixture weights from the survival probability of each control-field source, and evaluates the resulting posterior exactly by Gauss-Hermite quadrature. This last point deviates deliberately from the iterative scheme proposed in Lombardi (2018, Sect. 2.6), which in our injection tests develops a bias of up to 0.08 mag for weakly constrained sources; the published scheme remains available as an option.

Extinction maps follow the well-established NICER approach: the discretized estimates are smoothed onto a WCS grid with a spatial kernel and inverse-variance weighting, with Gaussian, Epanechnikov, triangular, uniform, average, and median metrics available. The NICEST correction ([Lombardi 2009](https://ui.adsabs.harvard.edu/abs/2009A%26A...493..735L/abstract)) counteracts the bias from unresolved substructure and foreground contamination. Map products carry extinction, variance, source count, and source density planes.

Missing measurements are encoded as NaN throughout, and a source requires at least one observed color for an estimate. Direct color-space input without magnitudes is supported through `pnicer.Colors`, whereas the adaptive correction requires band magnitudes, as the completeness functions are defined in magnitude space.

## Verification

The test suite (75 tests) scrutinizes the mathematics rather than only the plumbing:

| Claim | Checked against |
| --- | --- |
| Posterior math is exact | Brute-force numerical integration, random configs & missing-band patterns |
| Deconvolution is correct | Bovy's reference C implementation (`extreme_deconvolution`) |
| NICER is NICER | Legacy v1.0 outputs (machine precision) *and* the K=1 limit of the Bayesian machinery (~2e-15) |
| Maps add up | Independent per-pixel brute force, incl. the NICEST correction, plus frozen legacy maps |
| Adaptive correction works | Ground-truth injection tests: bias ≤ 0.02 mag at A_K = 1-2 |

To run the suite from a clone: `pip install -e ".[dev]" && pytest`.

## Legacy

The original implementation, as described in and linked from the 2017 publication, remains available as the [v1.0 release](https://github.com/smeingast/PNICER/releases/tag/v1.0). Results obtained with it remain reproducible with the tagged version:

```bash
pip install git+https://github.com/smeingast/PNICER.git@v1.0
```

## Citation

If you make use of PNICER in your research, please cite [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract), and for the version 2.0 methodology also [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract).

<details>
<summary>BibTeX entries</summary>

```bibtex
@ARTICLE{2017A&A...601A.137M,
       author = {{Meingast}, Stefan and {Lombardi}, Marco and {Alves}, Jo{\~a}o},
        title = "{Estimating extinction using unsupervised machine learning}",
      journal = {\aap},
         year = 2017,
        month = may,
       volume = {601},
          eid = {A137},
        pages = {A137},
          doi = {10.1051/0004-6361/201630032}
}

@ARTICLE{2018A&A...615A.174L,
       author = {{Lombardi}, Marco},
        title = "{Optimal extinction measurements. I. Single-object extinction inference}",
      journal = {\aap},
         year = 2018,
        month = aug,
       volume = {615},
          eid = {A174},
        pages = {A174},
          doi = {10.1051/0004-6361/201832769}
}
```

</details>
