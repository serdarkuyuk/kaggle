import numpy as np
import matplotlib.pyplot as plt

def weighted_median(x=None):
    if x is None:
        x = np.random.randint(low=1, high=100, size=10)
    # x = np.array([17, 91, 35, 73, 51])
    inv_x = 1.0 / x
    w = inv_x / sum(inv_x)

    idxs = np.argsort(w)
    sorted_w = w[idxs]
    ##%%
    sorted_w_cumsum = np.cumsum(sorted_w)
    idx = np.where(sorted_w_cumsum > 0.5)[0][0]
    #print(idx)
    ##%%
    #plt.plot(sorted_w_cumsum, 'o');
    #plt.show()
    #print('sorted_w_cumsum: ', sorted_w_cumsum)

    pos = idxs[idx]
    #print(np.sort(x))

    return x[pos]
    #print(np.median(x))

x= np.array([17, 91, 35, 73, 51])
weighted_median(x)