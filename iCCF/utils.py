import os
import sys
import re
import warnings
import importlib.resources as resources
from contextlib import contextmanager
from copy import copy
import numpy as np
from astropy.io import fits

from .ssh_files import ssh_fits_open

_c = 299792.458


def no_stack_warning(message):
    old_show = copy(warnings.showwarning)
    def new_show(message, category, filename, lineno, file, line):
        print(category.__name__ + ':', message)
    warnings.showwarning = new_show
    warnings.warn(message)
    warnings.showwarning = old_show


def get_ncores():
    try:
        ncores = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        ncores = multiprocessing.cpu_count()
    return ncores


def float_exponent(f):
    return int(np.floor(np.log10(abs(f)))) if f != 0 else 0

def float_mantissa(f):
    return f / 10**float_exponent(f)


def one_line_warning(message, category, filename, lineno, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'

@contextmanager
def disable_exception_traceback():
    if hasattr(sys, 'tracebacklimit'):
        old_tb_limit = sys.tracebacklimit
        sys.tracebacklimit = 0
        yield
        sys.tracebacklimit = old_tb_limit
    else:
        sys.tracebacklimit = 0
        yield
        delattr(sys, 'tracebacklimit')


# from https://stackoverflow.com/a/30141358/1352183
def running_mean(x, N=2):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


# from https://stackoverflow.com/a/16090640
def natsort(s):
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)
    ]


def doppler_shift_wave(wave, rv, f=1.0):
    """ 
    Doppler shift the wavelength array `wave` by the radial velocity `rv` [km/s].
    Note: positive values for `rv` indicate a red-shift. Negative values 
    indicate a blue-shift.

    Parameters
    ----------
    wave : array or float
        Original wavelength to be shifted
    rv : float
        Radial velocity [in km/s]
    """
    return wave * f * (1 + rv / _c)


def numerical_gradient(rv, ccf):
    """
    Return the gradient of the CCF.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.

    Notes
    -----
    The gradient is computed using the np.gradient routine, which uses second
    order accurate central differences in the interior points and either first
    or second order accurate one-sides (forward or backwards) differences at
    the boundaries. The gradient has the same shape as the input array.
    """
    return np.gradient(ccf, rv)


def find_myself():
    """ Return the path to iCCF's source. """
    thisdir = os.path.dirname(os.path.realpath(__file__))
    topdir = os.path.dirname(thisdir)
    return topdir


def load_example_data():
    """ Load the example CCF stored in iCCF/example_data """
    file = os.path.join('example_data', 'CCF1.npy')
    try:
        
        path = os.path.join(resources.files('iCCF'), file)
        data = np.load(path)
    except AttributeError:
        path = os.path.join(find_myself(), 'iCCF/example_data/CCF1.npy')
        data = np.load(path)
    return data

def find_data_file(file):
    """ Find a file from iCCF/data """
    try:
        path = resources.files('iCCF') / 'data' / file
        if path.exists():
            return path
    except AttributeError:
        path = os.path.join(find_myself(), 'iCCF', 'data', file)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(file)


def _get_hdul(fitsfile, **kwargs):
    """ Dispatch opening of fits files to fits.open in normal cases, or to
    `ssh_fits_open` for filenames starting with "ssh:"
    """
    if fitsfile.startswith('ssh:'):
        hdul = ssh_fits_open(fitsfile[4:], **kwargs)
    else:
        hdul = fits.open(fitsfile, lazy_load_hdus=False, memmap=False)
    
    return hdul
