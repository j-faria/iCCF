import numpy as np
from astropy.io import fits

def getRV(fitsfile, keyword=None):
    hdul = fits.open(fitsfile)
    if keyword is not None:
        return hdul[0].header[keyword]
    
    # need to look for it
    fail = ValueError(f'Could not find any RV keyword in header of "{fitsfile}"')
    
    try: return hdul[0].header['HIERARCH ESO QC CCF RV']
    except KeyError: pass
    
    try: return hdul[0].header['HIERARCH ESO DRS CCF RVC']
    except KeyError: pass

    raise fail

def getRVarray(fitsfile, keywords=None):
    hdul = fits.open(fitsfile)

    try:
        sta, ste, n = (hdul[0].header['HIERARCH ESO RV START'],
                       hdul[0].header['HIERARCH ESO RV STEP'],
                       hdul[1].header['NAXIS1'])
    except KeyError:
        try:
            sta, ste, n = (hdul[0].header['CRVAL1'],
                           hdul[0].header['CDELT1'],
                           hdul[0].header['NAXIS1'])
        except KeyError:
            raise KeyError
    finally:
        return sta + ste * np.arange(n)


def getBJD(fitsfile, keyword=None, mjd=True):
    """
    Try to extract the bjd from `fitsfile`. If `keyword` is given, this function
    returns that keyword from the header of the fits file. Otherwise, it will
    try to search for the following typical keywords that store the bjd:
    - HIERARCH ESO QC BJD
    - HIERARCH ESO DRS BJD
    - MJD-OBS
    If `mjd` is True, the function returns *modified* julian day.
    """
    hdul = fits.open(fitsfile)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    fail = ValueError(f'Could not find any BJD keyword in header of "{fitsfile}"')

    if mjd:
        sub = 24e5 + 0.5
    else:
        sub = 24e5

    try: return hdul[0].header['HIERARCH ESO QC BJD'] - sub
    except KeyError: pass

    try: return hdul[0].header['HIERARCH ESO DRS BJD'] - sub
    except KeyError: pass

    try: return hdul[0].header['MJD-OBS'] 
    except KeyError: pass

    raise fail