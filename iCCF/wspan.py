import numpy as np
import matplotlib.pyplot as plt

from .gaussian import gaussfit


def wspan(rv, ccf):
    """ 
    Calculate the Wspan, defined in Santerne et al. MNRAS, 451, 3 (2015).
    This is a measure of the CCF asymmetry based on two Gaussian fits to the
    blue and red wings of the line profile.

    Parameters
    ----------
    rv : array_like
        The radial velocity values where the CCF is defined.
    ccf : array_like
        The CCF values.
    """
    # fit a Gaussian to the profile
    c = gaussfit(rv, ccf)
    x0, sig = c[1], c[2]
    xx = rv - x0

    # fit the blue wing of the CCF
    mask_blue = (rv <= x0) | (rv >= x0 + 3*sig)
    c_blue = gaussfit(rv[mask_blue], ccf[mask_blue], c)

    # fit the red wing of the CCF
    mask_red = (rv <= x0 - 3*sig) | (rv >= x0)
    c_red = gaussfit(rv[mask_red], ccf[mask_red], c)

    # Vspan is the velocity difference
    return c_blue[1] - c_red[1]


def Wspan_plot(rv, ccf, rvunits='km/s'):
    """ Make a pretty plot of the line profile and the Wspan """
    pass