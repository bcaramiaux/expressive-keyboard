import csv
import pickle
import numpy as np
from scipy.signal import butter, lfilter, savgol_filter, savgol_coeffs, filtfilt
import matplotlib.pylab as plt
from random import shuffle
import datetime

import load
import visualization
from scipy import signal
from processing import resample, savitzky_golay_filter, adhoc_features
from learning import unsupervised_features, supervised_features, clf_cross_validation

from sklearn import svm
from sklearn import neighbors

import warnings
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")


def classification_unsupervised_features(data, labels, config):
    
    # test parameters
    parameters = {
                    'feature_type': 'kinematics', #'uist'
                    'granularity': 'sample', #'gesture'
                    'dimension_reduction': 'pca', #'None', 'pca'
                    'dimension_reduction_args': {'n_components': 3},
                    'classifier': 'svm'
                 }

    # build features
    feature_type = parameters['feature_type']
    features_, labels_ = adhoc_features(data, labels, feat_type=feature_type, granularity=parameters['granularity'])
    
    # unsupervised dimension reduction
    method = parameters['dimension_reduction']
    if method == 'None':
        feats = {}
        if parameters['feature_type'] == 'kinematics':
            feats['projection'] = features_[:,2:]
        else:
            feats['projection'] = features_
        pkl_fn = 'clf_instructions_{}.pkl'.format(feature_type)
    else:
        if parameters['feature_type'] == 'kinematics':
            features_ = features_[:,2:]
        method_args = parameters['dimension_reduction_args']
        feats = unsupervised_features(features_, method=method, args=method_args)
        argument_string = ''
        for k in method_args.keys(): 
            argument_string += '{}'.format(k)+'='+'{}'.format(method_args[k])+'_'
        pkl_fn = 'clf_instructions_{}_{}_{}.pkl'.format(feature_type,method,argument_string)

    print('writing features')
    pickle.dump({'features': feats}, open(pkl_fn, 'wb'))
    pickle.dump(np.transpose(feats['mapping']), open('mapping.pkl', 'wb'))
    print(feats['mapping'].shape)
    
    # classification
    X = feats['projection']
    Y = np.array([config['conditions'].index(c) + 1 for c in labels_[:, 0]])
    scores, confmat = clf_cross_validation(X, Y, clf=parameters['classifier'])
    print('Scores: {:.2f}%'.format(np.mean(scores)*100.0))

    # store test
    pickle.dump({'scores': scores, 'confmat': confmat, 'features': feats}, open(pkl_fn, 'wb'))
    




# def classification_unsupervised_features_by_words_participants(data, labels, config):
#     # build features
#     # @feat_type uist: spd, curvature
#     # @feat_type dyn: pos, vel, acc, jerk
#     features_, labels_ = adhoc_features(
#         data, labels, feat_type='dyn', average=False)

#     # unsupervised dimension reduction
#     # features_ = unsupervised_features(features_, method='pca', args={'n_components': 2})

#     # supervised dimension reduction
#     # Y = np.array([config['conditions'].index(c) + 1 for c in labels_[:, 0]])
#     # features_ = supervised_features(features_, Y, method='lda')

#     # classification
#     # @clf: knn, svm, (to add: lstm)
#     labels_int = np.array([config['conditions'].index(c) + 1 for c in labels_[:, 0]])
#     parts = np.unique(labels_[:,2])
#     for w in config['words']:
#         all_scores = []
#         for p in parts:
#             idx_p = np.where(labels_[:,2] == p)[0]
#             idx_w = np.where(labels_[:,3] == w)[0]
#             idx = list(set(idx_p) & set(idx_w))
#             X = features_[idx,:]
#             Y = labels_int[idx]
#             scores, confmat = clf_cross_validation(X, Y, clf='knn')
#             all_scores.append(scores)
#         all_scores = np.array(all_scores).T
#         plt.errorbar(np.arange(all_scores.shape[1])+1, np.mean(all_scores, axis=0), yerr=np.std(all_scores, axis=0), label=w)
#     lgd = plt.legend(bbox_to_anchor=(1.2,1.0))
#     plt.xlabel('Participants')
#     plt.ylabel('Mean instruction classification')
#     plt.savefig('clf_inst_words_parts.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
#     plt.show()


# def sample_wise_classification(data, labels, config):
#     features_, labels_ = adhoc_features(
#         data, labels, feat_type='dyn', average=False)
#     parts = np.unique(labels_[:,2])
#     labels_int = np.array([config['conditions'].index(c) + 1 for c in labels_[:, 0]])
#     for p in parts:
#         for w in config['words']:
#             idx_p = sorted(np.where(labels_[:,2] == p)[0])
#             idx_w = sorted(np.where(labels_[:,3] == w)[0])
#             idx = sorted(list(set(idx_p) & set(idx_w)))
#             X = features_[idx, 2:]
#             Y = labels_int[idx]
#             clf = svm.SVC()
#             clf.fit(X, Y)
#             y_pred = clf.predict(X)
#             for c in np.unique(Y):
#                 idx_c = sorted(np.where(Y == c)[0])
#                 gestures  = []
#                 gest_tmp = []
#                 predictions = []
#                 pred_tmp = []
#                 gest_tmp.append(X[idx_c[0],:2])
#                 pred_tmp.append(y_pred[idx_c[0]])
#                 for i in range(1,len(idx_c)):
#                     if idx_c[i]-idx_c[i-1]>1:
#                         gestures.append(gest_tmp)
#                         gest_tmp = []
#                         predictions.append(pred_tmp)
#                         pred_tmp = []
#                     gest_tmp.append(X[idx_c[i],:2])
#                     pred_tmp.append(y_pred[idx_c[i]])
#                 gestures.append(gest_tmp)
#                 predictions.append(pred_tmp)
#                 print(c,len(gestures), len(gestures[0]))
#                 plt.plot(predictions[0])
#                 plt.show()
                
#             # print(len(y_pred))
            


if __name__ == "__main__":

    # load all the data and experiment configuration
    data, labels = load.get_data()
    config = load.get_config()
    classification_unsupervised_features(data, labels, config)

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
