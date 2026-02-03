---
title: Line profile indicators
---

A picture is sometimes worth a thousand words, and this plot from [Santerne et
al (2015)](https://doi.org/10.1093/mnras/stv1080){:target="_blank"} illustrates
quite well some of the commonly used line-profile indicators.


![](assets/images/santerne2015.jpeg){width=70%}


Currently, the following line profile indicators are implemented in **iCCF**


### FWHM

The FWHM is simply the full width at half maximum of the Gaussian fit to the
CCF. It is measured in the same units as the radial velocity, tipically km/s.


### BIS (bisector span)

The *bisector* is made up of the mid-points of horizontal line segments bounded
by the sides of the line profile [[Toner & Gray
1988](https://ui.adsabs.harvard.edu/abs/1988ApJ...334.1008T/abstract){:target="_blank"}].

### Vspan

Suggested by [Boisse et al
2011](https://doi.org/10.1051/0004-6361/201014354){:target="_blank"}, obtained
from the RV difference measured by fitting two Gaussian functions to the top and
to the bottom parts of the line profile. The limits between the top and the
bottom of the line profile are defined as the $\pm 1 \sigma$ limit from the
measured RV.

### $\Delta V$ (bi-Gaussian fit)

This indicator was discussed in [Nardetto et al
2006](https://doi.org/10.1051/0004-6361:20054333){:target="_blank"} and
[Figueira et al
2013](https://www.aanda.org/articles/aa/abs/2013/09/aa20779-12/aa20779-12.html){:target="_blank"}.
It is the RV difference between a Gaussian and a bi-Gaussian fit to the CCF. The
bi-Gaussian is an asymmetric Gaussian with two different widths around the
center.


### Vasy 😢

```py
raise NotImplementedError
```

### Wspan

[Santerne et al (2015)](https://doi.org/10.1093/mnras/stv1080){:target="_blank"}
defined another indicator of line asymmetry, the Wspan, obtained from fitting
two Gaussian functions to the blue and red wings of the line profile.


<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"></script>
