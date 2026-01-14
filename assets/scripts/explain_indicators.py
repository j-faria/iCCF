import numpy as np
from matplotlib import pyplot as plt

from iCCF.gaussian import gauss
from iCCF.bigaussian import bigauss

def plot_ccf(ax=None):
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    
    p = [-1, 0.5, 1, 0.03, 0]
    rv = np.linspace(-10, 10, 72)
    ccf = bigauss(rv, p) + np.random.normal(0, 3e-3, len(rv))
    ax.plot(rv, ccf, 'ko')
    ax.axis('off')
    return ax, rv, ccf

def DeltaV():
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    plot_ccf(axs[0])


if __name__ == '__main__':
    DeltaV()