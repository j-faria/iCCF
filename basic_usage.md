---
layout: default
title: Basic Usage
has_children: false
nav_order: 3
---

Most often, you will want to create an `Indicators` object from a CCF file:

```python
import iCCF

i = iCCF.from_file('r.ESPRE.YYYY-MM-DDTHH:MM:SS.SSS_CCF_A.fits')
```

Then, the `i` object stores several CCF indicators as attributes.  
The radial velocity and uncertainty are stored in the `RV` and `RVerror`
attributes

```python
i.RV
i.RVerror
```