import numpy as np 
from numpy import exp, log, sqrt, inf
from scipy import optimize


def gauss(x, p):
    """ A Gaussian function with parameters p = [A, x0, σ, offset]. """
    return p[0] * exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]


def gauss_partial_deriv(x, p):
    """ Partial derivatives of a Gaussian with respect to each parameter """
    A, x0, sig, offset = p
    g = gauss(x, p)
    dgdA = gauss(x, [1.0, x0, sig, 0.0])
    dgdx0 = dgdA * ((x-x0)/sig**2)
    dgdsig = dgdA * ((x-x0)**2/sig**3)
    dgdoffset = np.ones_like(x)
    return np.c_[dgdA, dgdx0, dgdsig, dgdoffset]


def gaussfit(x, y, p0=None):
    """ 
    Fit a Gaussian function to `x`,`y` using least-squares, with initial guess
    `p0` = [A, x0, σ, offset]. If p0 is not provided, the function tries an
    educated guess, which might lead to bad results.
    """
    f = lambda p, x, y: gauss(x, p) - y
    # f = lambda x, A, x0, sig, offset: gauss(x, [A, x0, sig, offset])
    
    if p0 is None:
        p0 = []
        p0.append(y.mean() - y.max())  # guess the amplitude
        # guess the center, but maybe the ccf is upside down?
        if y[x.size // 2] > y[0]:  # seems like it
            p0.append(x[y.argmax()])
        else:
            p0.append(x[y.argmin()])
        p0.append(1)  # guess the sigma
        p0.append(y.mean())  # guess the offset
    
    result = optimize.leastsq(f, p0, args=(x, y))

    result[0][2] = abs(result[0][2]) # positive sigma
    return result[0]
    # bounds = ([-inf, -inf, 0, -inf], [inf, inf, inf, inf])
    # return optimize.curve_fit(f, x, y, p0=p0, bounds=bounds)[0]


def sig2fwhm(sig):
    """ Convert standard deviation to full width at half maximum. """
    return 2 * sqrt(2 * log(2)) * sig


def fwhm2sig(fwhm):
    """ Convert full width at half maximum to standard deviation. """
    return fwhm / (2 * sqrt(2 * log(2)))


def RV(rv, ccf):
    """
    Calculate the radial velocity as the center of a Gaussian fit the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    """
    _, rv, _, _ = gaussfit(rv, ccf)
    return rv


def FWHM(rv, ccf):
    """
    Calculate the full width at half maximum (FWHM) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    """
    _, _, sig, _ = gaussfit(rv, ccf)
    return sig2fwhm(sig)


def contrast(rv, ccf):
    """
    Calculate the contrast (depth, measured in percentage) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    """
    A, _, _, continuum = gaussfit(rv, ccf)
    return abs(100 * A / continuum)
