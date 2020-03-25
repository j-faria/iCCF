import numpy as np
from astropy.io import fits

from .ssh_files import ssh_fits_open

def _get_hdul(fitsfile, **kwargs):
    if fitsfile.startswith('ssh:'):
        hdul = ssh_fits_open(fitsfile[4:], **kwargs)
    else:
        hdul = fits.open(fitsfile)#, lazy_load_hdus=False)
    return hdul


def getRV(fitsfile, hdul=None, keyword=None, return_hdul=False, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    fail = ValueError(f'Could not find any RV keyword in header of "{fitsfile}"')

    if return_hdul:
        ret = lambda a: (a, hdul)
    else:
        ret = lambda a: a

    try:
        return ret(hdul[0].header['HIERARCH ESO QC CCF RV'])
    except KeyError:
        pass

    try:
        return ret(hdul[0].header['HIERARCH ESO DRS CCF RVC'])
    except KeyError:
        pass

    raise fail


def getRVarray(fitsfile, hdul=None, keywords=None, return_hdul=False, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

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
        if return_hdul:
            return sta + ste * np.arange(n), hdul
        else:
            return sta + ste * np.arange(n)


def getBJD(fitsfile, hdul=None, keyword=None, mjd=True, return_hdul=False,
           **kwargs):
    """
    Try to extract the bjd from `fitsfile`. If `keyword` is given, this function
    returns that keyword from the header of the fits file. Otherwise, it will
    try to search for the following typical keywords that store the bjd:
    - HIERARCH ESO QC BJD
    - HIERARCH ESO DRS BJD
    - MJD-OBS
    If `mjd` is True, the function returns *modified* julian day.
    """
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    fail = ValueError(f'Could not find any BJD keyword in header of "{fitsfile}"')

    if mjd:
        sub = 24e5 + 0.5
    else:
        sub = 24e5

    if return_hdul:
        ret = lambda a: (a, hdul)
    else:
        ret = lambda a: a

    try:
        return ret(hdul[0].header['HIERARCH ESO QC BJD'] - sub)
    except KeyError:
        pass

    try:
        return ret(hdul[0].header['HIERARCH ESO DRS BJD'] - sub)
    except KeyError:
        pass

    try:
        return ret(hdul[0].header['MJD-OBS'])
    except KeyError:
        pass

    raise fail


def getFWHM(fitsfile, hdul=None, keyword=None, return_hdul=False, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    fail = ValueError(
        f'Could not find any FWHM keyword in header of "{fitsfile}"')

    if return_hdul:
        ret = lambda a: (a, hdul)
    else:
        ret = lambda a: a

    try:
        return ret(hdul[0].header['HIERARCH ESO QC CCF FWHM'])
    except KeyError:
        pass

    raise fail