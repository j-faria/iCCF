import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline
from functools import partial
from PyAstronomy.pyasl import intep
from multipledispatch import dispatch

from .gaussian import gaussfit

def bisector(x, y, center='min', k=2):
    """ 
    Return a function which evaluates the bisector of a CCF profile.

    Parameters
    ----------
    x : array_like
        The x (velocity) values where the CCF is defined.
    y : array_like
        The y (CCF) values.
    center : string
        If 'min', the bisector starts at the minimum of `y`. 
        If 'rv', the bisector stats at the center of a Gaussian fit to `x`,`y`.
    k : int, str
        If int, interpolation order (k=2 - quadratic is the default).
        If k='intep', use the INTEP routine from Hill, PDAO 16, 67 (1982) as it
        is implemented in PyAstronomy.
    """
    if isinstance(k, int):
        # k-th order spline interpolation
        spl = InterpolatedUnivariateSpline(x, y, k=k)
    elif k == 'intep':
        spl = partial(intep, x, y)
    else:
        raise ValueError('In `bisector`, k should be int or "intep".')

    if center.lower() == 'min':
        # maybe the ccf is upside down?
        if y[x.size // 2] > y[0]: # yap
            center = x[y.argmax()]
        else:
            center = x[y.argmin()]
        left, right = x.min(), x.max()

    elif center.lower() == 'rv':
        c = gaussfit(x, y, [y.mean() - y.max(), x[y.argmin()], 1, y.mean()])
        left, center, right = x.min(), c[1], x.max()

    # invert the ccf interpolator, on the left side
    invfleft = lambda yy: brentq(lambda rv: spl(rv) - yy, left, center)

    # invert the ccf interpolator, on the right side
    invfright = lambda yy: brentq(lambda rv: spl(rv) - yy, center, right)

    # the bisector
    bis = lambda yyy: 0.5 * (invfleft(yyy) + invfright(yyy))
    bis = np.vectorize(bis)
    return bis


def BIS(x, y, down=(10, 40), up=(60, 90), full_output=False):
    """
    Calculate the Bisector Inverse Slope (BIS) of a line profile given by x,y.
    The BIS is defined as the difference between the average bisector in the 
    top and bottom sections of the line profile, where
    top = flux levels within `up`% 
    bottom = flux levels within `down`% 
    of the line full depth. See Figueira+ A&A 557, A93 (2013).

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    down, up : tuples, optional
        Tuples defining the lower and upper regions of the profile used for the
        BIS calculation. Defaults are the original values from Queloz+(2001).
        Values should be given in percentage.
    full_output : boolean, optional
        Return extra intermediate values. 
    """
    # check the limits
    assert 0 < down[0] < 100 and 0 < down[1] < 100 and down[0] < down[1]
    assert 0 < up[0] < 100 and 0 < up[1] < 100 and up[0] < up[1]

    # fit a Gaussian to the profile
    c = gaussfit(x, y, [y.mean() - y.max(), x[y.argmin()], 1, y.mean()])

    # the absolute bottom and range of the (fitted) profile
    bot, ran = c[3] + c[0], abs(c[0])

    # a uniform range of depths
    depth = np.linspace(c[3], bot, 100)

    # the absolute flux limits for the given limits
    bottom_limit1 = bot + ran * (down[0] / 100)
    bottom_limit2 = bot + ran * (down[1] / 100)
    top_limit1 = bot + ran * (up[0] / 100)
    top_limit2 = bot + ran * (up[1] / 100)

    # the mid-point flux levels at the bottom and at the top
    fl1 = 0.5 * (bottom_limit1 + bottom_limit2)
    fl2 = 0.5 * (top_limit1 + top_limit2)

    # the bisector "function"
    bisf = bisector(x, y)
    # bisectors at the top and bottom regions of the profile
    RV_top = bisf(depth[(top_limit1 < depth) & (depth < top_limit2)])
    RV_bot = bisf(depth[(bottom_limit1 < depth) & (depth < bottom_limit2)])
    # difference of the average bisectors
    BIS = RV_top.mean() - RV_bot.mean()
    
    if full_output:
        return BIS, c, bot, ran, \
               (bottom_limit1, bottom_limit2), (top_limit1, top_limit2), \
               fl1, fl2, bisf
    else:
        return BIS


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




def normalize_ccf(rv, ccf):
    p = gaussfit(rv, ccf, [ccf.mean() - ccf.max(), rv[ccf.argmin()], 1, ccf.mean()])
    a, _, _, off = p
    nccf = -off/a*(1.-ccf/off)
    return nccf



def Lovis_interpolate(rv, ccf):
    # err = np.ones(len(ccf), 'd')
    p = gaussfit(rv, ccf, [ccf.mean() - ccf.max(), rv[ccf.argmin()], 1, ccf.mean()])
    k, v0, sigma, c = p
    norm_ccf = -c/k*(1.-ccf/c)
    nstep = 100
    margin = 5
    depth = np.arange(nstep-2*margin+1)/nstep + float(margin)/nstep

    p = np.zeros([len(ccf),3],'d')
    bis_b = np.zeros(len(depth),'d')
    bis_r = np.zeros(len(depth),'d')

    for i in range(len(ccf)-1):
        if (max(norm_ccf[i],norm_ccf[i+1]) >= depth[0]) & (min(norm_ccf[i],norm_ccf[i+1]) <= depth[-1]):
            v = (rv[i]+rv[i+1])/2.
            dccfdRV = -(v-v0)/sigma**2 * np.exp(-(v-v0)**2/2/sigma**2)
            d2ccfdRV2 = ((v-v0)**2/sigma**2-1)/sigma**2 * np.exp(-(v-v0)**2/2/sigma**2)
            d2RVdccf2 = -d2ccfdRV2/dccfdRV**3
            p[i,2] = d2RVdccf2/2
            p[i,1] = (rv[i+1]-rv[i]-p[i,2]*(norm_ccf[i+1]**2-norm_ccf[i]**2))/(norm_ccf[i+1]-norm_ccf[i])
            p[i,0] = rv[i] - p[i,1]*norm_ccf[i] - p[i,2]*norm_ccf[i]**2

    for j in range(len(depth)):
        i_b = norm_ccf.argmax()
        while (norm_ccf[i_b] > depth[j]) & (i_b > 1): 
            i_b = i_b-1
        i_r = norm_ccf.argmax()
        while (norm_ccf[i_r+1] > depth[j]) & (i_r < len(ccf)-2): 
            i_r = i_r+1
        bis_b[j] = p[i_b,0] + p[i_b,1]*depth[j] + p[i_b,2]*depth[j]**2
        bis_r[j] = p[i_r,0] + p[i_r,1]*depth[j] + p[i_r,2]*depth[j]**2
    
    return bis_b, bis_r, (1 + k*depth/c)*c


def BIS_HARPS(rv, ccf, down=(10, 40), up=(60, 90)):
    """ 
    BIS calculation as performed by the HARPS pipeline.
    This implementation uses a custom quadratic interpolation, devised by
    Christophe Lovis, where the quadratic term is set by a Gaussian fit to 
    the profile. 

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    down, up : tuples
        Tuples defining the lower and upper regions of the profile used for the
        BIS calculation. Defaults are the original values from Queloz+(2001).
        Values should be given in percentage.
    """

    # check the limits
    assert 0 < down[0] < 100 and 0 < down[1] < 100 and down[0] < down[1]
    assert 0 < up[0] < 100 and 0 < up[1] < 100 and up[0] < up[1]
    down = tuple(d/100 for d in down)
    up = tuple(u/100 for u in up)

    # err = np.ones(len(ccf), 'd')
    p = gaussfit(rv, ccf, [ccf.mean() - ccf.max(), rv[ccf.argmin()], 1, ccf.mean()])
    k, v0, sigma, c = p
    norm_ccf = -c/k*(1.-ccf/c)
    nstep = 100
    margin = 5
    depth = np.arange(nstep-2*margin+1)/nstep + float(margin)/nstep

    p = np.zeros([len(ccf),3],'d')
    bis_b = np.zeros(len(depth),'d')
    bis_r = np.zeros(len(depth),'d')

    for i in range(len(ccf)-1):
        if (max(norm_ccf[i],norm_ccf[i+1]) >= depth[0]) & (min(norm_ccf[i],norm_ccf[i+1]) <= depth[-1]):
            v = (rv[i]+rv[i+1])/2.
            dccfdRV = -(v-v0)/sigma**2 * np.exp(-(v-v0)**2/2/sigma**2)
            d2ccfdRV2 = ((v-v0)**2/sigma**2-1)/sigma**2 * np.exp(-(v-v0)**2/2/sigma**2)
            d2RVdccf2 = -d2ccfdRV2/dccfdRV**3
            p[i,2] = d2RVdccf2/2
            p[i,1] = (rv[i+1]-rv[i]-p[i,2]*(norm_ccf[i+1]**2-norm_ccf[i]**2))/(norm_ccf[i+1]-norm_ccf[i])
            p[i,0] = rv[i] - p[i,1]*norm_ccf[i] - p[i,2]*norm_ccf[i]**2

    for j in range(len(depth)):
        i_b = norm_ccf.argmax()
        while (norm_ccf[i_b] > depth[j]) & (i_b > 1): 
            i_b = i_b-1
        i_r = norm_ccf.argmax()
        while (norm_ccf[i_r+1] > depth[j]) & (i_r < len(ccf)-2): 
            i_r = i_r+1
        bis_b[j] = p[i_b,0] + p[i_b,1]*depth[j] + p[i_b,2]*depth[j]**2
        bis_r[j] = p[i_r,0] + p[i_r,1]*depth[j] + p[i_r,2]*depth[j]**2

    bis = (bis_b+bis_r)/2.
   
    for i in range(len(bis)):
        if not np.isfinite(bis[i]): 
            bis = np.zeros(len(depth),'d')
   
    qq = np.greater_equal(depth, down[0]) * np.less_equal(depth, down[1])
    RV_top = np.mean(np.compress(qq, bis))
    qq = np.greater_equal(depth, up[0]) * np.less_equal(depth, up[1])
    RV_bottom = np.mean(np.compress(qq, bis))
    span = RV_top - RV_bottom
   
    return span #(1 + k*depth/c)*c, bis, span, v0, depth
