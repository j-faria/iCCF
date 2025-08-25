---
layout: default
title: Scripts
has_children: false
nav_order: 4
---

**iCCF** provides a few command-line scripts to work with CCFs

- `iccf-make-ccf`

This script is very meta, because it creates CCFs themselves! Provided an S2D
file with a stellar spectra, it will correlate it with a CCF mask to calculate
the CCF. It uses code from the `iCCF.meta` and `iCCF.mask` modules.

```
usage: iccf-make-ccf [-h] [-o OUTPUT] [-m MASK] [-rv RV] [--rv-range RV_RANGE] 
                          [--keep-prefix] [--ncores NCORES] files [files ...]

This script takes a list of S2D fits files and calculates the CCF for 
a given RV array and a given mask. If these are not provided, it uses
the same as specified in the S2D file.

positional arguments:
  files                 S2D files

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file name
  -m MASK, --mask MASK  Mask (G2, G9, K6, M2, ...). A file called `INST_[mask].fits` should exist.
  -rv RV, --rv RV       RV array, in the form start:end:step [km/s]
  --rv-range RV_RANGE   Full RV range where to calculate CCF [km/s]
  --keep-prefix         Keep any prefix of the S2D file on the output file
  --ncores NCORES       Number of cores to distribute calculation; default is all available (20)
```