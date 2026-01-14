from textwrap import dedent
import numpy as np
from .utils import _get_hdul

__all__ = ['getRV', 'getRVerror', 'getRVarray',
           'getBJD', 'getFWHM', 'getCONTRAST', 'getMASK', 'getINSTRUMENT']


def _check_hdul(hdul, fitsfile, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)
    return hdul

def _try_keywords(hdul, *keywords, exception=None):
    for kw in keywords:
        try:
            return hdul[0].header[kw]
        except KeyError:
            pass
    return None


class getKW:
    __call_doc__ = """
    Args:
        fitsfile (str):
            The name of the fits file
        hdul (astropy.fits.HDUList, optional):
            If provided, ignore `fitsfile` and use this HDU list directly
        keyword (str, optional):
            Get this keyword from the header instead of looking for RV keywords
        return_hdul: (bool, optional)
            Whether to return the HDU list read from the file
    """
    def __init__(self, name, kws):
        self.name = name
        self.kws = kws
    def __call__(self, fitsfile, hdul=None, keyword=None, return_hdul=False, **kwargs):
        hdul = _check_hdul(hdul, fitsfile, **kwargs)
        if keyword is not None:
            return hdul[0].header[keyword]
        val = _try_keywords(hdul, *self.kws)
        if val is not None:
            if return_hdul:
                return val, hdul
            else:
                return val
        obj = fitsfile or hdul.filename()
        fail = ValueError(f'Could not find any {self.name} keyword in header of "{obj}"')
        raise fail


getOBJECT = getKW('object', ['OBJECT', 'HIERARCH TNG OBS TARG NAME'])


getRV = getKW('RV', ['HIERARCH ESO QC CCF RV', 'HIERARCH ESO DRS CCF RVC',
                     'HIERARCH TNG QC CCF RV'])
getRV.__doc__ = """
Try to find the radial velocity in the header of `fitsfile`. If `keyword`
is not provided, search for the following keywords
    - HIERARCH ESO QC CCF RV
    - HIERARCH ESO DRS CCF RVC
""" + dedent(getKW.__call_doc__)


getRVerror = getKW('RV error', ['HIERARCH ESO QC CCF RV ERROR', 'HIERARCH ESO DRS CCF NOISE'])
getRVerror.__doc__ = """
Try to find the radial velocity uncertainty in the header of `fitsfile`. If
`keyword` is not provided, search for the following keywords
    - HIERARCH ESO QC CCF RV ERROR
    - HIERARCH ESO DRS CCF NOISE
""" + dedent(getKW.__call_doc__)


getBIS = getKW('BIS', ['HIERARCH ESO QC CCF BIS SPAN'])


def getRVarray(fitsfile, hdul=None, return_hdul=False, **kwargs):
    """
    Try to find the radial velocity array from the header of `fitsfile`. This 
    function will look for the keywords
    - HIERARCH ESO RV START, HIERARCH ESO RV STEP, NAXIS1
    - CRVAL1, CDELT1, NAXIS1

    Arguments
    ---------
    fitsfile : str:
        The name of the fits file
    hdul : astropy.fits.HDUList (optional)
        If provided, ignore `fitsfile` and use this HDU list directly
    return_hdul: bool (optional, default False)
        Whether to return the HDU list read from the file
    """
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    sta = getKW('RV start', ['HIERARCH ESO RV START', 'CRVAL1', 'HIERARCH TNG RV START'])(None, hdul=hdul)
    ste = getKW('RV step', ['HIERARCH ESO RV STEP', 'CDELT1', 'HIERARCH TNG RV STEP'])(None, hdul=hdul)
    try:
        n = hdul[0].header['NAXIS1']
    except KeyError:
        n = hdul[1].header['NAXIS1']

    if return_hdul:
        return sta + ste * np.arange(n), hdul
    else:
        return sta + ste * np.arange(n)


def getBJD(fitsfile, hdul=None, keyword=None, mjd=True, return_hdul=False,
           **kwargs):
    """
    Try to find the BJD in the header of `fitsfile`. If `keyword` is not 
    provided, search for the following keywords
    - HIERARCH ESO QC BJD
    - HIERARCH ESO DRS BJD
    - MJD-OBS
    
    If `mjd` is True, the function returns the *modified* julian day.

    Arguments
    ---------
    fitsfile : str:
        The name of the fits file
    hdul : astropy.fits.HDUList (optional)
        If provided, ignore `fitsfile` and use this HDU list directly
    keyword : str (optional)
        Get this keyword from the header instead of looking for RV keywords
    return_hdul: bool (optional, default False)
        Whether to return the HDU list read from the file
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


getFWHM = getKW('FWHM', ['HIERARCH ESO QC CCF FWHM', 'HIERARCH ESO DRS CCF FWHM'])
getFWHM.__doc__ = """
Try to find the FWHM in the header of `fitsfile`. If `keyword` is not 
provided, search for the following keywords
    - HIERARCH ESO QC CCF FWHM
    - HIERARCH ESO DRS CCF FWHM
""" + dedent(getKW.__call_doc__)


getCONTRAST = getKW('CONTRAST', ['HIERARCH ESO QC CCF CONTRAST'])
getCONTRAST.__doc__ = """
Try to find the contrsat in the header of `fitsfile`. If `keyword` is not 
provided, search for the following keywords
    - HIERARCH ESO QC CCF CONTRAST
""" + dedent(getKW.__call_doc__)


def getMASK(fitsfile, hdul=None, keyword=None, return_hdul=False, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    kws = ['HIERARCH ESO QC CCF MASK', 'HIERARCH ESO DRS CCF MASK']
    val = _try_keywords(hdul, *kws)

    if val is not None:
        if return_hdul:
            return val, hdul
        else:
            return val

    obj = fitsfile or hdul
    fail = ValueError(f'Could not find any MASK keyword in header of "{obj}"')
    raise fail


def getINSTRUMENT(fitsfile, hdul=None, keyword=None, return_hdul=False, **kwargs):
    if hdul is None:
        hdul = _get_hdul(fitsfile, **kwargs)

    if keyword is not None:
        return hdul[0].header[keyword]

    # need to look for it
    kws = ['INSTRUME', 'HIERARCH ESO INS MODE']
    val = _try_keywords(hdul, *kws)

    if val is not None:
        if return_hdul:
            return val, hdul
        else:
            return val

    obj = fitsfile or hdul
    fail = ValueError(f'Could not find any instrument keyword in header of "{obj}"')
    raise fail


get_SCIRED_CHECK = getKW('SCIRED CHECK', ['HIERARCH ESO DRS CCF CHECK'])

