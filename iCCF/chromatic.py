import os
import bisect

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from cached_property import cached_property

from .iCCF import Indicators
from .gaussian import gaussfit, RV
from .keywords import getRV, getRVarray
from .utils import find_myself
# from .utils import get_orders_mean_wavelength


def read_spectral_format():
    here = os.path.dirname(__file__)
    sf_red_file = os.path.join(here, '..', 'data', 'spectral_format_red.dat')
    red = np.loadtxt(sf_red_file)
    sf_blue_file = os.path.join(here, '..', 'data', 'spectral_format_blue.dat')
    blue = np.loadtxt(sf_blue_file)

    col_start_wave = 7
    col_end_wave = 8

    order_wave_range = {}

    for i, order in enumerate(blue[::-1]):
        order_range = [order[col_start_wave], order[col_end_wave]]
        order_wave_range[i] = order_range

    for i, order in enumerate(red[::-1], start=i+1):
        order_range = [order[col_start_wave], order[col_end_wave]]
        order_wave_range[i] = order_range

    return order_wave_range


class chromaticRV():
    def __init__(self, indicators):
        """
        indicators : Indicators or list
            Instance or list of instances of iCCF.Indicators
        """
        self.order_wave_range = read_spectral_format()
        self.wave_starts = [v[0] for v in self.order_wave_range.values()]
        self.wave_ends = [v[1] for v in self.order_wave_range.values()]

        self.blue_wave_limits = (440, 570)
        self.mid_wave_limits = (570, 690)
        self.red_wave_limits = (730, 790)

        self._slice_policy = 0  # by default use both slices
        
        self.blue_orders = self._find_orders(self.blue_wave_limits)
        self.mid_orders = self._find_orders(self.mid_wave_limits)
        self.red_orders = self._find_orders(self.red_wave_limits)
        
        self._blueRV = None
        self._midRV = None
        self._redRV = None

        self.n = len(indicators)
        if self.n == 1:
            indicators = [indicators,]
        self.I = self.indicators = indicators
        # store all but the last CCF for each of the Indicators instances
        self.ccfs = [i.HDU[1].data[:-1] for i in self.I]

    def __repr__(self):
        bands = ', '.join(map(repr, self.bands))
        nb = len(self.bands)
        return f'chromaticRV({self.n} CCFs; {nb} bands: {bands} nm)'

    @property
    def slice_policy(self):
        """ How to deal with the two order slices.
        0: use both slices by adding the corresponding CCFs
        1: use only the first slice
        2: use only the second slice
        Default is 0.
        """
        return self._slice_policy
    
    @slice_policy.setter
    def slice_policy(self, val):
        self._slice_policy = val
        self.blue_orders = self._find_orders(self.blue_wave_limits)
        self.mid_orders = self._find_orders(self.mid_wave_limits)
        self.red_orders = self._find_orders(self.red_wave_limits)
        self._blueRV = None
        self._midRV = None
        self._redRV = None


    def _find_orders(self, wave_limits):
        order_start = bisect.bisect(self.wave_starts, wave_limits[0])
        order_end = bisect.bisect(self.wave_ends, wave_limits[1])
        
        order_start = order_start * 2
        order_end = order_end * 2 + 1

        if self.slice_policy == 0: # using both order slices
            step = 1
            return slice(order_start, order_end+1, step)

        elif self.slice_policy == 1:  # using first slice only
            step = 2
            return slice(order_start, order_end+1, step)

        elif self.slice_policy == 2:  # using second slice only
            step = 2
            return slice(order_start+1, order_end+1, step)


    def get_rv(self, orders):
        """ Get radial velocity for specific orders

        orders : int, slice, tuple, array
            The CCFs of these orders will be summed to calculate the RV.
            If int, only the CCF at that index will be used.
            If slice, orders in the slice's range will be used.
            If tuple, should have length 2 or 3 and correspond to minimum index,
            maximum index and possibly step, for which orders to use
            If array, should contain indices of orders to use
        """
        if isinstance(orders, int):
            orders = slice(orders, orders + 1)
        elif isinstance(orders, tuple):
            orders = slice(*orders)

        rv = []
        for i in self.I:
            ccf = i.HDU[1].data[orders].sum(axis=0)
            rv.append(RV(i.rv, ccf))
        return np.array(rv)


    @property
    def bands(self):
        """ Wavelength limits of blue, mid, and red bands """
        b = self.blue_wave_limits, self.mid_wave_limits, self.red_wave_limits
        return b

    @cached_property
    def time(self):
        """ BJD of observations """
        return np.fromiter((i.bjd for i in self.I), dtype=np.float,
                           count=self.n)

    @property
    def blueRV(self):
        if self._blueRV is None:
            self._blueRV = self.get_rv(self.blue_orders)
        return self._blueRV

    @property
    def midRV(self):
        if self._midRV is None:
            self._midRV = self.get_rv(self.mid_orders)
        return self._midRV

    @property
    def redRV(self):
        if self._redRV is None:
            self._redRV = self.get_rv(self.red_orders)
        return self._redRV

    def plot_ccfs(self, orders=None):
        if orders is None:
            orders = slice(None, None)
        elif isinstance(orders, int):
            orders = slice(orders, orders + 1)
        elif isinstance(orders, tuple):
            orders = slice(*orders)
        
        for i in self.I:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(i.ccf[orders].T)




def each_order_rv(rv, ccfs, exclude_last=True):
    """
    Calculate RVs for each spectral order by fitting Gaussians to individual CCFs
    
    Parameters
    ----------
    rv : array
        Radial velocities where each CCF is defined
    ccfs : array
        The CCFs for each spectral order (order o, radial velocity rv)
    exclude_last : bool
        Whether to exclude the last index of ccfs (usually the sum of all other 
        CCFs) from the calculation.
    
    Returns
    -------
    rvs : array
        The center of a Gaussian fit to each order's CCF
    """
    last = -1 if exclude_last else None

    gen = (gaussfit(rv, ccf)[1] for ccf in ccfs[:last])
    rvs = np.fromiter(gen, dtype=float)

    return rvs


def rv_color(rv, ccfs, blue=slice(0,80), red=slice(80,-1), avoid_blue=0, gap=0):
    """
    Calculate the RV color by combining blue and red CCFs

    Parameters
    ----------
    rv : array
        Radial velocities where each CCF is defined
    ccfs : array
        The CCFs for each spectral order (order o, radial velocity rv)
    blue : slice
        A slice object with the start and stop indices of the blue orders. The
        default (0:80) is for ESPRESSO. For HARPS, use ...
    red : slice
        A slice object with the start and stop indices of the red orders. The
        default (80:-1) is for ESPRESSO. For HARPS, use ...
    avoid_blue : int
        How many orders to skip in the bluest part of the spectrum. This will 
        be added to the beginning of the `blue` slice
    gap : int or tuple
        If an integer, the number of orders to remove from the "middle" for both
        blue and red parts. If a tuple, the number of orders to remove from the 
        blue and red, respectively
    """
    if isinstance(gap, tuple):
        gap_blue, gap_red = gap
    elif isinstance(gap, int):
        gap_blue = gap_red = gap
    else:
        raise ValueError(f"`gap` should be int or tuple, got {gap}")

    blue = slice(blue.start + avoid_blue, blue.stop - gap_blue)
    red = slice(red.start + gap_red, red.stop)

    ccf_blue = ccfs[blue, :].sum(axis=0)
    ccf_red = ccfs[red, :].sum(axis=0)

    rv_blue = gaussfit(rv, ccf_blue)[1]
    rv_red = gaussfit(rv, ccf_red)[1]
    print(rv_blue, rv_red)



def chromatic_index(rv, ccfs, wave, rvpipe=None):
    """ 
    Calculate the chromatic index, as described in Zechmeister et al. (2018).

    Parameters
    ----------
    rv : array
        Radial velocities where each CCF is defined   
    """
    if isinstance(wave, str): # assume it's a filename
        wave = get_wave(wave)
    elif isinstance(wave, np.ndarray):
        pass
    else:
        raise ValueError('`wave` should be filename or array with wavelengths')

    mean_wave = get_orders_mean_wavelength(wave, log=True)
    rvs = each_order_rv(rv, ccfs)

    ind = ~np.isnan(rvs)
    p = np.polyfit(np.log(mean_wave[ind]), rvs[ind], 1)

    if rvpipe is None:
        rvpipe = gaussfit(rv, ccfs[-1])[1]

    beta = p[0]
    lv = np.exp(abs((p[1] - rvpipe)/p[0]))
    return beta, lv



def chromatic_index_from_files(s2dfile, ccffile):
    """ 
    Calculate the chromatic index, as described in Zechmeister et al. (2018).

    Parameters
    ----------
    s2dfile : str
        Filename of the S2D fits file
    ccffile : str
        Filename of the CCF fits file
    """
    wave = get_wave(s2dfile)
    mean_wave = get_orders_mean_wavelength(wave, log=True)

    rvpipe = getRV(ccffile)
    rv = getRVarray(ccffile)

    ccfs = fits.open(ccffile)[1].data
    rvs = each_order_rv(rv, ccfs)

    return chromatic_index(rv, ccfs, wave, rvpipe)
