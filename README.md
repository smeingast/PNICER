<h1 align="center">PNICER</h1>

<p align="center"><i>Interstellar extinction from photometry — probabilistically, fast, and without priors.</i></p>

<p align="center">
  <a href="https://github.com/smeingast/PNICER/actions/workflows/ci.yml"><img src="https://github.com/smeingast/PNICER/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License"></a>
  <a href="https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract"><img src="https://img.shields.io/badge/A%26A-2017%2C%20601%2C%20A137-orange" alt="Paper"></a>
</p>

![Orion A extinction map](assets/orion.png)

Somewhere between you and every star sits interstellar dust, quietly reddening and dimming starlight. PNICER measures how much: it turns photometric catalogs into per-star extinction **probability densities** and smooth extinction **maps** — using nothing but an extinction-free control field. No column-density priors, no assumed source types, no fuss.

## ✨ Highlights

- 🎲 **Full posteriors, not just numbers** — every source gets an analytic extinction PDF (a 1-D Gaussian mixture); collapse it to a point estimate whenever you're ready.
- 🧹 **Deconvolved, not smeared** — the intrinsic color distribution is fitted with extreme deconvolution (Bovy et al. 2011), so the control field's own photometric errors don't inflate yours. Uncertainties come out calibrated to ~0.1%.
- 🌫️ **Sees through the population shift** — behind dense clouds, faint galaxies vanish from your sample before stars do. The adaptive population correction (Lombardi 2018) removes the resulting bias: ≲ 0.02 mag at A_K = 2 where classic estimators drift by ~0.1 mag.
- 🕳️ **Missing bands? Handled exactly** — NaN photometry is projected out of the problem, not patched over. Two observed bands suffice.
- 🏛️ **The classics, included** — NICER (Lombardi & Alves 2001) in the same interface, NICEST (Lombardi 2009) map correction built in.
- ⚡ **Fast and boring in the best way** — pure numpy/scipy vectorization, ~10⁶ sources/second on a laptop, fully deterministic with a seed, no multiprocessing, runs on Linux/macOS/Windows.

## 🚀 Quick start

```python
from pnicer import Photometry

bands = {"J": ("Jmag", "e_Jmag"), "H": ("Hmag", "e_Hmag"), "Ks": ("Kmag", "e_Kmag")}
extinction = {"J": 2.5, "H": 1.55, "Ks": 1.0}   # A_band / A_Ks

science = Photometry.from_fits("orion.fits", bands=bands, extinction=extinction,
                               lon="GLON", lat="GLAT", frame="galactic")
control = Photometry.from_fits("control.fits", bands=bands, extinction=extinction,
                               lon="GLON", lat="GLAT", frame="galactic")

# Fit the intrinsic color model once, reuse it forever
model = control.fit_intrinsic_colors(random_state=0)

# Extinction posteriors for every star (with the adaptive population correction)
posterior = science.pnicer(model, adaptive=True)

# Point estimates → smooth map → FITS
catalog = posterior.discretize()
emap = catalog.build_map(bandwidth=5 / 60, metric="gaussian", use_fwhm=True)
emap.save("orion_ak.fits")
emap.plot()

# Prefer the classic? Same interface:
nicer = science.nicer(control)
```

Want to see it run right now? The map at the top of this page comes bundled:

```python
from pnicer.demo import orion
orion()
```

## 📦 Installation

```bash
pip install git+https://github.com/smeingast/PNICER.git
```

Python ≥ 3.11; numpy, scipy, astropy, and scikit-learn come along automatically. For plotting, grab the extra: `pip install "pnicer[plot] @ git+https://github.com/smeingast/PNICER.git"`.

## 🔬 Under the hood

Version 2.0 is a from-scratch rewrite. It keeps the PNICER idea from [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract) — extinction PDFs from a control field, purely data-driven — and replaces the original numerical machinery with the closed-form Bayesian formalism of [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract): a Gaussian mixture model of the intrinsic colors, deconvolved from measurement errors, yields each source's extinction posterior analytically. Good ideas from both papers, one clean implementation.

Practical notes: missing measurements are NaN throughout; sources need at least one observed color. Direct color-space input (no magnitudes) works via `pnicer.Colors`; the adaptive correction needs band magnitudes, since completeness lives in magnitude space.

## ✅ Trust, but verify

We take "verified" seriously — the test suite (75 tests) checks the math, not just the plumbing:

| Claim | Checked against |
| --- | --- |
| Posterior math is exact | Brute-force numerical integration, random configs & missing-band patterns |
| Deconvolution is correct | Bovy's reference C implementation (`extreme_deconvolution`) |
| NICER is NICER | Legacy v1.0 outputs (machine precision) *and* the K=1 limit of the Bayesian machinery (~2e-15) |
| Maps add up | Independent per-pixel brute force, incl. the NICEST correction, plus frozen legacy maps |
| Adaptive correction works | Ground-truth injection tests: bias ≤ 0.02 mag at A_K = 1–2 |

Run it yourself from a clone: `pip install -e ".[dev]" && pytest`.

## 🕰️ Legacy

The original implementation, as used since the 2017 paper, lives on as the [v1.0 release](https://github.com/smeingast/PNICER/releases/tag/v1.0):

```bash
pip install git+https://github.com/smeingast/PNICER.git@v1.0
```

## 📖 Citation

If PNICER helps your research, please cite [Meingast, Lombardi & Alves (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...601A.137M/abstract) — and for the 2.0 methodology also [Lombardi (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...615A.174L/abstract).

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
