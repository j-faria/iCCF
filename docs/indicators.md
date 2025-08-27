---
layout: default
title: Line profile indicators
has_children: false
nav_order: 3
---

A picture is sometimes worth a thousand words, and this plot from [Santerne et
al (2015)](https://doi.org/10.1093/mnras/stv1080){:target="_blank"} illustrates
quite well some of the commonly used line-profile indicators.

![](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/451/3/10.1093/mnras/stv1080/2/m_stv1080figa1.jpeg?Expires=1758699556&Signature=CcJJnXmxYtfsgk1qG2kz~VG6WEZFNvlQmbmPLNtTsAgv2FrpH-SqT-2c00AeOFmWnJc0ZNMzOd99LbCm4CgYXuHxlQR~xP9WeJo7JlmoKzi4DHu9rDcrQS2dZA7Ebm56--XIVXi5FmGEjOSJ3LipMvFFJB2VWOVEBOhJSfvfkpc2KHgrgM3JKXV6tK9ZV4V4eXRTFiZ2fN3xGOfise4pyzbqQRztYckCZfcIHKW9KDz7amOlPVse-IM48Rhh-vn2tUo1foKfFfzbnZNzc3NIf7HWzvfKxmSknRp2I4Sm5QJa532NqLTn4btvfVne7Hi~tYJUh0aXpKajZEylZf~gHw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


Currently, the following line profile indicators are implemented in **iCCF**

<!-- <details open markdown="block"> -->
  <!-- <summary></summary> -->
  {: .text-delta }
1. TOC
{:toc}
<!-- </details> -->


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
bottom of the line profile are defined as the $$\pm 1 \sigma$$ limit from the
measured RV.

### BiGauss 😢

```py
raise NotImplementedError
```

### Vasy 😢

```py
raise NotImplementedError
```

### Wspan

[Santerne et al (2015)](https://doi.org/10.1093/mnras/stv1080){:target="_blank"}
defined another indicator of line asymmetry, the Wspan, obtained from fitting
two Gaussian functions to the blue and red wings of the line profile.


<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"></script>
