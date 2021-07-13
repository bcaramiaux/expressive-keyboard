# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np

from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Input, Dense, Lambda, Layer, Dropout, LSTM
from keras import backend as K
from keras import metrics


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils


def lstm(data):
    raw_data = data
    mins = [np.min(d) for d in data]
    maxs = [np.max(d) for d in data]
    gmin = np.min(mins)
    gmax = np.max(maxs)
    n_vocab = int(gmax - gmin)

    # # load ascii text and covert to lowercase
    # filename = "wonderland.txt"
    # raw_text = open(filename, 'rb').read()
    # raw_text = raw_text.lower()
    # # create mapping of unique chars to integers
    # chars = sorted(list(set(raw_text)))
    # sample_to_int = dict((c, i) for i, c in enumerate(np.arange(n_vocab)))
    # print(sample_to_int)
    # # summarize the loaded data
    # n_chars = len(raw_text)
    # n_vocab = len(chars)
    # print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    # # prepare the dataset of input to output pairs encoded as integers
    seq_length = 10
    dataX = []
    dataY = []
    for di, d in enumerate(data):
        for i in range(0, len(d) - seq_length, 1):
            seq_in = d[i:i + seq_length]
            seq_out = d[i + seq_length]
            dataX.append([int(char) for char in seq_in])
            dataY.append(int(seq_out))
    # n_patterns = len(dataX)
    # print("Total Patterns: ", n_patterns)
    # # reshape X to be [samples, time steps, features]
    # X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # # normalize
    # X = X / float(n_vocab)
    # # one hot encode the output variable
    # y = np_utils.to_categorical(dataY)
    # # define the LSTM model
    # model = Sequential()
    # model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    # model.add(Dropout(0.2))
    # model.add(Dense(y.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # # define the checkpoint
    # filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    # # fit the model
    # # model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

def vae(data):
    
    batch_size = 256
    original_dim = data.shape[1]
    intermediate_dim = 64
    latent_dim = 3
    epsilon_std = 1.0

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    # sampling function from latent encoding
    def sampling(args):
        z_mean, z_log_sigma = args
        # epsilon = K.random_normal(shape=(batch_size, latent_dim),
        #                           mean=0., stddev=epsilon_std)
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    # norm
    data = (data + np.min(data, axis=0)) / np.max(data + np.min(data, axis=0), axis=0)

    idx = np.arange(data.shape[0])
    # np.random.shuffle(idx)
    idx_train = idx[:int(0.8*len(idx))]
    idx_test = idx[int(0.8*len(idx)):]

    x_train = data[idx_train,:]
    x_test = data[idx_test,:]

    vae.fit(x_train, x_train,
            epochs=5,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test))

    x_t = encoder.predict(data) #, batch_size=batch_size)

    # # this is the size of our encoded representations
    # encoding_dim = 3
    # # this is our input placeholder
    # input_data = Input(shape=(data.shape[1],))
    # # "encoded" is the encoded representation of the input
    # encoded = Dense(encoding_dim, activation='relu')(input_data)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(data.shape[1], activation='sigmoid')(encoded)
    # # this model maps an input to its reconstruction
    # autoencoder = Model(input_data, decoded)
    # # separate encoder
    # encoder = Model(input_data, encoded)
    # # separate decoder
    # # create a placeholder for an encoded (32-dimensional) input
    # encoded_input = Input(shape=(encoding_dim,))
    # # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # # norm
    # data = (data + np.min(data, axis=0)) / np.max(data + np.min(data, axis=0), axis=0)

    # idx = np.arange(data.shape[0])
    # np.random.shuffle(idx)
    # idx_train = idx[:int(0.8*len(idx))]
    # idx_test = idx[int(0.8*len(idx)):]

    # x_train = data[idx_train,:]
    # x_test = data[idx_test,:]

    # autoencoder.fit(x_train, x_train,
    #                 epochs=5,
    #                 batch_size=256,
    #                 shuffle=True,
    #                 validation_data=(x_test, x_test))

    # x_t = encoder.predict(data)
    return x_t


def vae_(data):
    batch_size = 100
    original_dim = data.shape[1]
    latent_dim = data.shape[1]
    intermediate_dim = 64
    epochs = 5
    epsilon_std = 1.0

    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(
                x, x_decoded_mean)
            kl_loss = -0.5 * K.sum(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)

    # train the VAE on MNIST digits
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # data= data / np.max(data, axis=0)

    # norm
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx_train = idx[:int(0.8*len(idx))]
    idx_test = idx[int(0.8*len(idx)):]

    x_train = data[idx_train,:]
    x_test = data[idx_test,:]

    # norm
    # x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    # x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

    # print(x_train.shape)
    # print(x_train[10,:])
    # plt.plot(x_train[10,:])
    # plt.show()

    vae.fit(
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    x_t = encoder.predict(data)
    # # display a 2D plot of the digit classes in the latent space
    # x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    # plt.colorbar()
    # plt.show()

    # # build a digit generator that can sample from the learned distribution
    # decoder_input = Input(shape=(latent_dim, ))
    # _h_decoded = decoder_h(decoder_input)
    # _x_decoded_mean = decoder_mean(_h_decoded)
    # generator = Model(decoder_input, _x_decoded_mean)

    # # display a 2D manifold of the digits
    # n = 15  # figure with 15x15 digits
    # digit_size = 28
    # figure = np.zeros((digit_size * n, digit_size * n))
    # # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    # grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    # grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    # for i, yi in enumerate(grid_x):
    #     for j, xi in enumerate(grid_y):
    #         z_sample = np.array([[xi, yi]])
    #         x_decoded = generator.predict(z_sample)
    #         digit = x_decoded[0].reshape(digit_size, digit_size)
    #         figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(
    #             j + 1) * digit_size] = digit

    # plt.figure(figsize=(10, 10))
    # plt.imshow(figure, cmap='Greys_r')
    # plt.show()
    return x_t


def unsupervised_features(data, method='pca', args={}):
    feats = {}
    feats['data'] = data
    if method == 'pca':
        pca = PCA(**args)
        feats['projection'] = pca.fit_transform(data)
        feats['mapping'] = pca.components_
        feats['variances'] = pca.explained_variance_ratio_
        print(data.shape, pca.components_.shape, feats['projection'].shape)
    elif method == 'isomap':
        pca = Isomap(**args)
        feats['projection'] = pca.fit_transform(data)
    elif method == 'tsne':
        pca = TSNE(**args)
        feats['projection'] = pca.fit_transform(data)
    elif method == 'vae':
        feats['projection'] = vae(data)
    return feats


def supervised_features(data, labels, method='pca', args={}):
    if method == 'lda':
        lda = LinearDiscriminantAnalysis(
            solver="svd", store_covariance=True, **args)
        y_pred = lda.fit(data, labels).predict(data)
        x_t = lda.transform(data)
    return x_t


def clf_cross_validation(X, Y, clf='knn', args={}):
    print('\033[95m' + 'Classification cross-vals' + '\033[0m')
    print('data size:', X.shape)
    clfs = {
            'knn': neighbors.KNeighborsClassifier, 
            'svm': svm.SVC
            }
    scores = []
    confmat = np.zeros((len(np.unique(Y)), len(np.unique(Y))))
    n_splits = 20
    kf = StratifiedShuffleSplit(n_splits=n_splits)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X, Y)):
        utils.progress_bar(fold_i + 1, n_splits)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        cur_clf = clfs[clf](**args)
        
        cur_clf.fit(X_train, Y_train)
        y_pred = cur_clf.predict(X_test)
        score = len(np.where(Y_test == y_pred)[0]) / len(Y_test)
        confmat_fold = confusion_matrix(Y_test, y_pred)
        scores.append(score)
        confmat = confmat + confmat_fold
    utils.close_progress_bar()
    return scores, confmat


