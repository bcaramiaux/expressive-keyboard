import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import load


def confmat():
    filename = 'clf_instructions_uist'
    conf = pickle.load(open('{}.pkl'.format(filename), 'rb'))
    conf['confmat'] /= np.tile(
        np.sum(conf['confmat'], axis=1).reshape(-1, 1),
        [1, conf['confmat'].shape[1]])
    print(conf['confmat'])
    config = load.get_config()
    labels = [l.replace(" ", "\n") for l in config['conditions']]
    plt.imshow(conf['confmat'])
    plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    plt.colorbar()
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')
    plt.show()


def mapping():
    n_components = 3
    filename = 'clf_instructions_dyn_pca_'
    conf = pickle.load(open('{}.pkl'.format(filename), 'rb'))
    print(
        np.cumsum(conf['features']['variances']),
        conf['features']['mapping'][:, :3])
    best_dims = []
    best_corrs = []
    for d in range(n_components):
        correls = []
        for k in range(conf['features']['data'].shape[1]):
            correls.append(np.abs(np.corrcoef(conf['features']['data'][:,k],conf['features']['projection'][:,d])[0,1]))
        print(d,correls)
        best_dims.append(np.argmax(correls))
        best_corrs.append(correls[np.argmax(correls)])
    print('Total variance explained = {}'.format(np.cumsum(conf['features']['variances'])[n_components-1]), best_dims, best_corrs)

def clfacc_on_gesture():
    pass


if __name__ == "__main__":
    confmat()