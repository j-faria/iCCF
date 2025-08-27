---
layout: default
title: What is a CCF?
has_children: false
nav_order: 2
---

Hey ChatGPT, what is the cross correlation function used  
in the radial velocity method of exoplanet detection?
{: .text-right }

<br/>
Great question! Let’s break it down clearly.

---

### Context: Radial Velocity Method

In the **radial velocity (RV) method** of exoplanet detection, astronomers look at how a star’s spectrum shifts back and forth due to the gravitational pull of an orbiting planet. The star wobbles slightly, which causes **Doppler shifts** in its spectral lines. Measuring these tiny shifts tells us the star’s velocity toward or away from us.

---

### Cross-Correlation Function (CCF)

The **cross-correlation function (CCF)** is a mathematical tool used to measure these small shifts. Here’s how it works:

1. **Template spectrum**: A high-resolution reference spectrum (either synthetic or from a similar star) is chosen.

2. **Observed spectrum**: The actual stellar spectrum is recorded by the spectrograph.

3. **Cross-correlation**: The observed spectrum is compared against the template by sliding one relative to the other in velocity space (shifting in wavelength by small Doppler velocities).

4. **CCF value**: At each trial velocity shift $$v$$, a correlation coefficient is computed:

   $$ CCF(v) = \sum_{\lambda} O(\lambda) \, T(\lambda_v) $$

   where

   * $$O(\lambda)$$ = observed spectrum,
   * $$T(\lambda_v)$$ = template shifted by velocity $$v$$.

   (In practice, both spectra are usually continuum-normalized, and the sum is weighted to emphasize absorption lines.)

5. **Best-fit velocity**: The velocity that maximizes the cross-correlation function is taken as the star’s **radial velocity**.

---

### Interpretation

* The **CCF curve** as a function of velocity looks like an inverted Gaussian (because absorption lines produce a dip in correlation).
* The **minimum of the Gaussian fit** to the CCF gives the radial velocity.
* By tracking how this radial velocity changes over time, astronomers infer the presence of an exoplanet and its orbital parameters.

---

✅ **In summary:**
The cross-correlation function in the radial velocity method is essentially a sliding comparison between the observed stellar spectrum and a template spectrum, measuring how well they match at each Doppler shift. The velocity shift that maximizes the correlation corresponds to the star’s radial velocity at that observation.


<br>

Thanks ChatGPT!
{: .text-right }


<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"></script>
