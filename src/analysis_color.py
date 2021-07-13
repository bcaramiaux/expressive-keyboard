import os
import csv
import pickle
import numpy as np
from scipy.signal import butter, lfilter, savgol_filter, savgol_coeffs, filtfilt
import matplotlib.pylab as plt
from random import shuffle
import datetime
import json

import load
import visualization
from scipy import signal
from processing import resample, savitzky_golay_filter, adhoc_features
from learning import unsupervised_features, supervised_features, clf_cross_validation

from sklearn import svm
from sklearn import neighbors

import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras import regularizers
import random


import warnings
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")


def classification_unsupervised_features(data, labels):
    
    # test parameters
    parameters = {
                    'feature_type': 'kinematics', #'uist'
                    'granularity': 'exp', #'gesture', 'sample', 'sequence', 'exp'
                    'dimension_reduction': 'pca', #'None', 'pca'
                    'dimension_reduction_args': {'n_components': 3},
                    'classifier': 'svm'
                 }

    # build features
    feature_type = parameters['feature_type']
    labels_int = []
    classes_map = []

    task = 'reg' # 'reg', 'clf'


    config = {'n_units': [2048],
              'dropout': 0.5,
              'n_epochs': 50,
              'feats': [0,-1]}


    if task == 'clf':

        for l in labels:
            lab = int(l[0]/255 + 2*l[1]/255 + 4*l[2]/255)
            if lab not in classes_map:
                classes_map.append(lab)
            labels_int.append(classes_map.index(lab))
    else:
        labels_int = labels

    features_, labels_ = adhoc_features(data, labels_int, feat_type=feature_type, granularity=parameters['granularity'], out_fn=feature_type+'_color')

    features_tmp_ = []
    labels_tmp_ = []
    for fi, f in enumerate(features_):
        if config['feats'][1] == -1:
            features_tmp = features_[fi][config['feats'][0]:]
        else:
            features_tmp = features_[fi][config['feats'][0]:config['feats'][1]]
        # plt.plot(features_tmp[:,:2])
        # plt.show()
        # print(np.std(features_tmp, axis=0).shape)
        # if len(np.where(np.isnan(features_tmp) == True)[0]):
        #     print(features_tmp)
        # features_tmp = np.subtract(features_tmp, np.mean(features_tmp, axis=0))
        # if len(np.where(np.mean(features_tmp, axis=0) == True)[0]):
        #     print(features_tmp)
        #features_tmp__ = features_tmp
        # stds = np.std(features_tmp, axis=0)
        # if len(np.where(stds == 0.0)[0]) == 0:
        #     print(fi, np.std(features_[fi], axis=0))

        #features_tmp = np.divide(features_tmp, np.std(features_tmp, axis=0))
        #if len(np.where(np.isnan(features_tmp) == True)[0]):
            # print(np.isnan(features_tmp))
            #print(np.std(features_tmp, axis=0))
            #print(features_tmp__)
            
        # plt.subplot(1,2,1)
        # plt.plot(features_[fi])
        # plt.subplot(1,2,2)
        # plt.plot(features_tmp)
        # plt.show()
        # if len(np.std(features_tmp, axis=0)) == 0:
        features_tmp_.append(features_tmp)
        labels_tmp_.append(labels_[fi] / 255.0)
    features_ = np.array(features_tmp_)
    print(features_.shape)
    labels_ = np.array(labels_tmp_)

    no_classes = len(np.unique(labels_))

    
    for n_epochs in [500]:

        for activ in ['sigmoid', 'linear']:

            for n_units in [2048, 1024, 256]:

                for dropout in [0.1, 0.5]:

                    scores = []

                    for loop in range(5):

                        print(n_epochs, n_units, activ, dropout, loop+1)

                        config['n_units'] = [n_units]
                        config['n_epochs'] = n_epochs
                        config['dropout'] = dropout
                        config['activation'] = activ


                        train_idx = np.random.choice(len(features_), int(2*len(features_)/3))
                        test_idx = [i for i in np.arange(len(features_)) if i not in train_idx]

                        if task == 'clf':
                            training_X = features_[train_idx]
                            #training_X = training_X.reshape(-1, training_X.shape[1] * training_X.shape[2])
                            training_y = labels_[train_idx]
                            training_y = keras.utils.to_categorical(training_y, num_classes=no_classes)

                            testing_X = features_[test_idx]
                            #testing_X = testing_X.reshape(-1, testing_X.shape[1] * testing_X.shape[2])
                            testing_y = labels_[test_idx]
                            testing_y = keras.utils.to_categorical(testing_y, num_classes=no_classes)

                        elif task == 'reg':
                            training_X = features_[train_idx]
                            training_y = labels_[train_idx]

                            testing_X = features_[test_idx]
                            testing_y = labels_[test_idx] #/ 255.0

                        l_in = Input(shape=(training_X.shape[1],))

                        for ni, n in enumerate(config['n_units']):
                            if ni == 0:
                                l_rnn = Dense(n, 
                                              activation='relu', 
                                              kernel_regularizer=regularizers.l2(0.00001),
                                              activity_regularizer=regularizers.l2(0.00001))(l_in)
                            else:
                                l_rnn = Dense(n, 
                                              activation='relu', 
                                              kernel_regularizer=regularizers.l2(0.00001),
                                              activity_regularizer=regularizers.l2(0.00001))(l_rnn)
                            l_rnn = Dropout(config['dropout'])(l_rnn)


                        if task == 'clf':
                            l_o = Dense(no_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.00001),
                                    activity_regularizer=regularizers.l2(0.00001))(l_rnn)
                            model = Model(inputs=l_in, outputs=l_o)
                            model.compile(loss=keras.losses.categorical_crossentropy,
                                          optimizer=keras.optimizers.Adam(),
                                          metrics=['accuracy'])

                        elif task == 'reg':
                            l_o = Dense(3, activation=activ)(l_rnn)
                            model = Model(inputs=l_in, outputs=l_o)
                            model.compile(loss='mean_squared_error', 
                                          optimizer=keras.optimizers.Adam())


                        model.fit(training_X, 
                                  training_y, 
                                  epochs=config['n_epochs'], 
                                  batch_size=64, 
                                  validation_data=(testing_X, testing_y),
                                  verbose=0)

                        scores.append(model.evaluate(testing_X, 
                                               testing_y, 
                                               batch_size=64,
                                               verbose=0))
                        # print(score)
                        pickle.dump(scores, open('model-{}-scores.pkl'.format(json.dumps(config)), 'wb'))
                        model.save('model-{}.h5'.format(json.dumps(config)))


                        if task == 'reg':
                            ypred = model.predict(testing_X)
                            # ypred = np.clip(ypred, 0., 1.)
                            colors = ['r', 'g', 'b']
                            plt.figure(figsize=(12,6))
                            plt.subplot(1,2,1)
                            for k in range(testing_y.shape[1]):
                                plt.plot(testing_y[:,k], color=colors[k])
                            plt.subplot(1,2,2)
                            for k in range(testing_y.shape[1]):
                                plt.plot(ypred[:,k], color=colors[k])
                            plt.savefig('model-{}-plot.pdf'.format(json.dumps(config)), bbox_inches='tight')
                            # plt.show()
                            plt.close()

                        del model 

                    print('  -> mean score:', np.mean(scores), '  | s:', np.std(scores))

  
def analysis_classification_res(folder='outputs'):
    scores = []
    models = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.split('.')[-1] == 'pkl':
                score = pickle.load(open(os.path.join(root, file), 'rb'))
                scores.append(np.mean(score))
                models.append(os.path.join(root, file[:-11]))
    i = np.argmax(scores)
    print(models[i])


if __name__ == "__main__":

    # load all the data and experiment configuration
    data, labels = load.get_color_data()
    # config = load.get_config()
    # classification_unsupervised_features(data, labels)

    analysis_classification_res()

    # load.clf_accuracies(data, labels)

    # classify_instructions_by_words_participants(data, labels, config)
    # labels_ = np.array(labels_)

    # for p in range(1,10):
    #     idx = np.where(labels_[:,2] == 'p{}'.format(p))[0]
    #     X, Y = features_[idx,:], labels_[idx,:]
    #     Y_int = np.array([config['conditions'].index(c)+1 for c in Y[:,0]])
    #     X_t = learn_representation_static(data=X, labels=Y_int, mthd='lda')
    #     visualization.vis_dataset_2d(X_t, Y_int)

    # viz_inter_participant_var(data, labels)
    # dataset_stats(labels)
    # gbl_gesture_variability(data, labels)
    # accuracies = clf_accuracies(data, labels)
    # variability_word_cond(data, labels)
