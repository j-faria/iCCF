from numpy import exp, log, sqrt
from scipy import optimize

twolntwo = 2 * log(2)

def bigauss(x, p):
    """ A bi-Gaussian function with parameters p = [A, x0, fwhm, asy, offset]. """
    return (x < p[1]).choose([
        p[0] * exp(-twolntwo * (x - p[1])**2 / (p[2] * (1 + p[3])**2)) + p[4],
        p[0] * exp(-twolntwo * (x - p[1])**2 / (p[2] * (1 - p[3])**2)) + p[4]
    ])


def bigaussfit(x, y, p0):
    """ 
    Fit a bi-Gaussian function to `x`,`y` using least-squares, with initial 
    guess `p0` = [A, x0, fwhm, asy, offset]. 
    """
    f = lambda p, x, y: bigauss(x, p) - y
    return optimize.leastsq(f, p0, args=(x, y))[0]
