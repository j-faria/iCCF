import numpy as np

# from https://stackoverflow.com/a/30141358/1352183
def running_mean(x, N=2):
    cumsum = np.cumsum(np.insert(x, 0, 0))  
    return (cumsum[N:] - cumsum[:-N]) / N 