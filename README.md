<p align="center">
  <img width = "450" src="https://github.com/j-faria/iCCF/blob/master/iCCF/logo/logo.png?raw=true"/>
  <!-- <br> -->
  <!-- Line profile asymmetry indicators -->
</p>


An implementation of common line profile indicators
measured from the cross-correlation function (CCF).

[![Build Status](https://travis-ci.org/j-faria/iCCF.svg?branch=master)](https://travis-ci.org/j-faria/iCCF)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/iCCF.svg)](https://pypi.org/project/iCCF/)
[![Funding](https://img.shields.io/badge/funding-FCT-darkgreen.svg)](https://www.fct.pt/)
[![PT](https://img.shields.io/badge/made%20in-🇵🇹-white.svg)](https://opensource.org/licenses/MIT)

#### To install 

Directly from PyPI
```
pip install iCCF
```

Or from source
```
git clone https://github.com/j-faria/iCCF.git
cd iCCF
python setup.py install 
```

You can append `--user` to both commands (`pip` and `python ...`) in case you don't have root access.


All the indicators are based on the works of others.
Please cite these works if you use **iCCF**.

  - [Queloz et al 2001](https://doi.org/10.1051/0004-6361:20011308)
  - [Boisse et al 2011](https://doi.org/10.1051/0004-6361/201014354)
  - [Nardetto et al 2006](https://doi.org/10.1051/0004-6361:20054333)
  - [Figueira et al 2013](https://www.aanda.org/articles/aa/abs/2013/09/aa20779-12/aa20779-12.html)
  - [Santerne et al 2015](https://doi.org/10.1093/mnras/stv1080)
  - [Lanza et al 2018](https://doi.org/10.1051/0004-6361/201731010)
  

#### See also

- Similar codes were developed by Figueira et al. A&A 557, A93 (2013)  
  with a Python package available [here](https://bitbucket.org/pedrofigueira/line-profile-indicators/src/master/)
  (described in Appendix A of Santos et al. A&A 566, A35 (2014))

- A similar package (in IDL) was developed by Lanza et al. A&A 616, A155 (2018).  
  It is available [here](https://www.ict.inaf.it/gitlab/antonino.lanza/HARPSN_spectral_line_profile_indicators).
