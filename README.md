<h1 align="center">PNICER</h1>

<p align="center"><i>Extinction estimates and maps from photometric catalogs.</i></p>

<p align="center">
  <a href="https://github.com/smeingast/PNICER/actions/workflows/ci.yml"><img src="https://github.com/smeingast/PNICER/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License"></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract"><img src="https://img.shields.io/badge/A%26A-2017%2C%20601%2C%20A137-orange" alt="Paper"></a>
</p>

![Orion A extinction map](assets/orion.png)

Interstellar dust dims and reddens the light of every star behind it. PNICER measures this extinction from ordinary photometric catalogs: each star gets a probability density for its line-of-sight extinction, and the per-star estimates combine into smooth extinction maps. The only calibration input is a control field, a nearby patch of sky with little to no dust. The method puts no prior on the cloud structure and makes no assumption about which types of sources are in the catalog.

## Highlights

- Each source gets a full posterior for its extinction, an analytic Gaussian mixture rather than a single value with an error bar. Collapse it to a point estimate when you need one.
- The intrinsic color distribution of the control field is fitted with extreme deconvolution (Bovy et al. 2011), which removes the control field's own photometric errors instead of folding them into your results. In our tests the reported uncertainties are accurate to about 0.1%.
- When a cloud sits in front of the background population, faint galaxies drop out of the catalog long before stars do. This shift biases classic estimators. The adaptive correction from Lombardi (2018) models it, and in simulations with known ground truth the bias stays below 0.02 mag at A_K = 2, where NICER is off by about 0.1 mag.
- Missing photometry is handled exactly through projection matrices. Two observed bands are enough for an estimate.
- NICER (Lombardi & Alves 2001) and the NICEST map correction (Lombardi 2009) are included in the same interface.
- The implementation is plain numpy and scipy: about a million sources per second on a laptop, reproducible with a seed, and free of multiprocessing, so it also runs on Windows.

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

# Extinction posteriors for every star, with the adaptive population correction
posterior = science.pnicer(model, adaptive=True)

# Point estimates, then a smoothed map written to FITS
catalog = posterior.discretize()
emap = catalog.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True)
emap.save("orion_ak.fits")
emap.plot()

# The classic NICER estimator, same interface
nicer = science.nicer(control)
```

The map at the top of this page comes from the bundled 2MASS demo data:

```python
from pnicer.demo import orion
orion()
```

## Installation

```bash
pip install git+https://github.com/smeingast/PNICER.git
```

Python 3.11 or newer; numpy, scipy, astropy, and scikit-learn are installed automatically. Plotting needs the optional extra: `pip install "pnicer[plot] @ git+https://github.com/smeingast/PNICER.git"`.

## Under the hood

Version 2.0 is a from-scratch rewrite. It keeps the PNICER idea from [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract), purely data-driven extinction PDFs calibrated on a control field, and replaces the original numerical machinery with the closed-form Bayesian formalism of [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract): a Gaussian mixture model of the intrinsic colors, deconvolved from measurement errors, yields each source's extinction posterior analytically.

Missing measurements are NaN throughout, and a source needs at least one observed color. Direct color-space input without magnitudes works via `pnicer.Colors`. The adaptive correction needs band magnitudes, since completeness lives in magnitude space.

## Verification

The test suite (75 tests) checks the math, not just the plumbing:

| Claim | Checked against |
| --- | --- |
| Posterior math is exact | Brute-force numerical integration, random configs & missing-band patterns |
| Deconvolution is correct | Bovy's reference C implementation (`extreme_deconvolution`) |
| NICER is NICER | Legacy v1.0 outputs (machine precision) *and* the K=1 limit of the Bayesian machinery (~2e-15) |
| Maps add up | Independent per-pixel brute force, incl. the NICEST correction, plus frozen legacy maps |
| Adaptive correction works | Ground-truth injection tests: bias ≤ 0.02 mag at A_K = 1-2 |

To run it from a clone: `pip install -e ".[dev]" && pytest`.

## Legacy

The original implementation, as used since the 2017 paper, remains available as the [v1.0 release](https://github.com/smeingast/PNICER/releases/tag/v1.0):

```bash
pip install git+https://github.com/smeingast/PNICER.git@v1.0
```

## Citation

If PNICER helps your research, please cite [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract), and for the 2.0 methodology also [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract).

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
