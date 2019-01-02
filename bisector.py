import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline

from gaussian import gaussfit


def bisector(x, y, center='min'):
    """ 
    Return a function which evaluates the bisector of a Gaussian.

    Parameters
    ----------
    x : array_like
        The x values where the Gaussian is defined.
    y : array_like
        The y values of the Gaussian.
    center : string
        If 'min', the bisector starts at the minimum of `y`. 
        If 'rv', the bisector stats at the center of a Gaussian fit to `x`,`y`.
    """
    spl = InterpolatedUnivariateSpline(x, y, k=1)  # linear interpolation

    if center.lower() == 'min':
        left, center, right = x.min(), x[y.argmin()], x.max()
    elif center.lower() == 'rv':
        c = gaussfit(x, y, [y.mean() - y.max(), x[y.argmin()], 1, y.mean()])
        left, center, right = x.min(), c[1], x.max()

    # invert the CCF interpolator, on the left side
    invfleft = lambda y: brentq(lambda rv: spl(rv) - y, left, center)

    # invert the CCF interpolator, on the right side
    invfright = lambda y: brentq(lambda rv: spl(rv) - y, center, right)

    # the bisector
    bis = lambda y: 0.5 * (invfleft(y) + invfright(y))
    bis = np.vectorize(bis)
    return bis


def BIS(x, y, down=(10, 40), up=(60, 90), full_output=False):
    """
    Calculate the Bisector Inverse Slope (BIS) of a line profile given by x,y.
    The BIS is defined as the difference between the velocity of the top and 
    bottom sections of the line profile, where
    top = average mid-point of the flux levels within `up`% 
    bottom = average mid-point of the flux levels within `down`% 
    of the line full depth. See Figueira et al. A&A 557, A93 (2013).
    """
    # check the limits
    assert 0 < down[0] < 100 and 0 < down[1] < 100 and down[0] < down[1]
    assert 0 < up[0] < 100 and 0 < up[1] < 100 and up[0] < up[1]

    # fit a Gaussian to the profile
    c = gaussfit(x, y, [y.mean() - y.max(), x[y.argmin()], 1, y.mean()])

    # the absolute bottom, and range of the (fitted) profile
    bot, ran = c[3] + c[0], abs(c[0])

    # the absolute flux limits for the given limits
    bottom_limit1 = bot + ran * (down[0] / 100)
    bottom_limit2 = bot + ran * (down[1] / 100)
    top_limit1 = bot + ran * (up[0] / 100)
    top_limit2 = bot + ran * (up[1] / 100)

    # the mid-point flux levels at the bottom and at the top
    fl1 = 0.5 * (bottom_limit1 + bottom_limit2)
    fl2 = 0.5 * (top_limit1 + top_limit2)

    # the BIS value
    bisf = bisector(x, y)
    bis = bisf(fl2) - bisf(fl1)
    if full_output:
        return bis, c, bot, ran, \
               (bottom_limit1, bottom_limit2), (top_limit1, top_limit2), \
               fl1, fl2, bisf
    else:
        return bis


def BISplus(x, y):
    """ The BIS+ is the BIS with down=(10%, 20%) and up=(80%, 90%)."""
    return BIS(x, y, down=(10, 20), up=(80, 90))


def BISminus(x, y):
    """ The BIS- is the BIS with down=(30%, 40%) and up=(60%, 70%)."""
    return BIS(x, y, down=(30, 40), up=(60, 70))


def BIS_plot(x, y, down=(10, 40), up=(60, 90), rvunits='km/s'):
    """ Make a pretty plot of the line profile and the BIS """
    # calculate 
    out = BIS(x, y, down, up, full_output=True)
    # unpack the output
    bis, c, bot, ran, \
    (bottom_limit1, bottom_limit2), (top_limit1, top_limit2), \
    fl1, fl2, bisf = out

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, 'o-', ms=3, alpha=0.2)
    
    mask_top = (top_limit1 < y) & (y < top_limit2)
    ax.plot(x[mask_top], y[mask_top], 'og', ms=4, alpha=0.6)

    yy = np.linspace(top_limit1, top_limit2, 100)
    ax.plot(bisf(yy), yy, 'k')
    ax.plot(bisf(fl2), fl2, 'or', ms=4)

    mask_bot = (bottom_limit1 < y) & (y < bottom_limit2)
    ax.plot(x[mask_bot], y[mask_bot], 'og', ms=4, alpha=0.6)

    yy = np.linspace(bottom_limit1, bottom_limit2, 100)
    ax.plot(bisf(yy), yy, 'k')
    ax.plot(bisf(fl1), fl1, 'or', ms=4)

    ax.set(xlabel=f'RV [{rvunits}]', ylabel='Flux')
    ax.legend(['observed CCF', 'BIS regions', 'bisector', 'mid-points'])
    ax.set_title(f'BIS {down};{up}  {bis:7.4f} {rvunits}')
    
    fig.tight_layout()
    plt.show()