from numpy import exp, log, sqrt
from scipy import optimize

def gauss(x, p):
    """ A Gaussian function with parameters p = [A, x0, σ, offset]. """
    return p[0] * exp(-(x-p[1])**2 / (2*p[2]**2)) + p[3]

def gaussfit(x, y, p0):
    """ 
    Fit a Gaussian function to `x`,`y` using least-squares, with initial guess
    `p0` = [A, x0, σ, offset]. 
    """
    f = lambda p, x, y: gauss(x, p) - y
    return optimize.leastsq(f, p0, args=(x, y))[0]

def sig2fwhm(sig):
    """ Convert standard deviation to full width at half maximum. """
    return 2 * sqrt(2*log(2)) * sig

def fwhm2sig(fwhm):
    """ Convert full width at half maximum to standard deviation. """
    return fwhm / (2 * sqrt(2*log(2)))