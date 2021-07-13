import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np


def vis_dataset_2d(data, labels=None):
    for l in np.unique(labels):
        idx = np.where(labels == l)[0]
        x, y = data[idx,0], data[idx,1]
        lbl = '{}'.format(l)
        c = cm.rainbow(l / 4.0)
        plt.scatter(x, y, color=c, label=l)
    plt.legend()
    plt.show()