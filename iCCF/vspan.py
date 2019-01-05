import numpy as np
import matplotlib.pyplot as plt

from .gaussian import gaussfit


def vspan(x, y, limits=(1, 3)):
    """ 
    Calculate the Vspan, defined in Boisse et al. A&A 528, A4 (2011).
    This is a measure of the CCF asymmetry based on two Gaussian fits which 
    consider exclusively the upper and lower part of the CCF.

    Parameters
    ----------
    x : array_like
        The x (velocity) values where the CCF is defined.
    y : array_like
        The CCF values.
    limits : tuple, optional
        For limits = (l1, l2), the upper part of the CCF is defined 
        in the range [-∞ : -l1*σ][+l1*σ : +∞] and the lower part is defined 
        in the range [-∞ : -l2*σ][-l1*σ : +l1*σ][+l2*σ : +∞],
        where σ is the width of a Gaussian fit to the full profile.
        Defaults to (1, 3), as used by Boisse+ (2011).
    """
    # fit a Gaussian to the profile
    c = gaussfit(x, y, [y.mean() - y.max(), x[y.argmin()], 1, y.mean()])
    x0, sig = c[1], c[2]
    xx = x - x0

    # fit the upper part of the CCF
    mask_upper = (xx < -limits[0] * sig) | (xx > limits[0] * sig)
    c_upper = gaussfit(x[mask_upper], y[mask_upper], c)

    # fit the lower part of the CCF
    mask_lower = (xx < -limits[1] * sig) | (xx > limits[1] * sig)
    mask_lower = mask_lower | ((-limits[0] * sig < xx) &
                               (xx < limits[0] * sig))
    c_lower = gaussfit(x[mask_lower], y[mask_lower], c)

    # Vspan is the velocity difference
    return c_upper[1] - c_lower[1]


def Vspan_plot(x, y, rvunits='km/s'):
    """ Make a pretty plot of the line profile and the Vspan """
    pass