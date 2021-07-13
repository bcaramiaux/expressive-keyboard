import numpy as np
import csv
import os
import pickle
from scipy.signal import butter, lfilter, savgol_filter, savgol_coeffs, filtfilt
import matplotlib.pyplot as plt
# from scipy.misc import imresize
# from processing import savitzky_golay_filter

conditions = [
    'exaggeratedly while sitting', 'quickly while sitting',
    'exaggeratedly while walking', 'quickly while walking'
]
participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
words = ['find', 'flag', 'last', 'life', 'over', 'that', 'time', 'with']
days = [1, 2, 3, 4, 5]
word_areas = [160380.0, 26784.0, 142560.0, 115830.0, 213840.0, 89100.0, 196020.0, 106920.0]

# def init_dict():
#     data = {}
#     labels = {}
#     for c in conditions:
#         data[c] = {}
#         labels[c] = {}
#         for pi, p in enumerate(participants):
#             data[c]['p{}'.format(pi+1)] = {}
#             labels[c]['p{}'.format(pi+1)] = {}
#             for w in words:
#                 data[c]['p{}'.format(pi+1)][w] = {}
#                 labels[c]['p{}'.format(pi+1)][w] = {}
#                 for d in days:
#                     data[c]['p{}'.format(pi+1)][w]['d{}'.format(d)] = []
#                     labels[c]['p{}'.format(pi+1)][w]['d{}'.format(d)] = []
#     return data, labels


def pickle_data(data_path='../data/', output_fn='dataset.pkl'):
    data, labels = [], []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if '[gestures]' in f:
                pid = int(f.split('_')[0][1:])
                if pid in participants:
                    did = int(f.split('_')[1].split('[')[0][3:])
                    if did in days:
                        with open(os.path.join(root, f), 'r') as fcsv:
                            reader = csv.reader(fcsv)
                            rows = np.array([row for row in reader])
                            keys = list(rows[0, :])
                            pre_word = 0
                            cur_gesture = []
                            for l in range(1, rows.shape[0]):
                                word_id = int(rows[l, keys.index('WORD_ID')]
                                              .split('-')[1])
                                phrase = rows[l, keys.index('PHRASE')]
                                cur_word = phrase.split(' ')[word_id]
                                cur_cond = rows[l, keys.index('INSTRUCTION')]
                                if pre_word != word_id:
                                    if len(cur_gesture) > 10:
                                        data.append(np.array(cur_gesture))
                                        labels.append([
                                            cur_cond, did, 'p{}'.format(
                                                participants.index(pid) + 1),
                                            rows[l - 1,
                                                 keys.index('WORD')],
                                            rows[l - 1,
                                                 keys.index('OUTPUT')]
                                        ])
                                        pre_word = word_id
                                        cur_gesture = []
                                    else:
                                        print('Check {}, wrong size'.format(
                                            [pid, did, cur_word]))
                                        pre_word = word_id
                                        cur_gesture = []
                                cur_gesture.append([
                                    float(rows[l, keys.index('X')]),
                                    float(rows[l, keys.index('Y')])
                                ])
    pickle.dump({'data': data, 'labels': labels}, open(output_fn, 'wb'))


def get_data(data_fn='dataset.pkl'):
    if not os.path.isfile(data_fn):
        pickle_data(output_fn=data_fn)
    dataset = pickle.load(open(data_fn, 'rb'))
    labels = np.array(dataset['labels'])
    return dataset['data'], labels


def get_color_data(data_fn='dataset_color.pkl'):
    if not os.path.isfile(data_fn):  
        with open('Exp2[gestures].csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            p_word_id = 0
            n_word_id = 0
            data = []
            labels = []
            new_word = []
            p_row = []
            for r_i, row in enumerate(csvReader):
                if r_i > 0:
                    n_word_id = int(row[5])
                    if n_word_id > p_word_id: 
                        if len(new_word):
                            if len(new_word) > 30 and len(new_word) < 750:
                                made = np.array([int(p_row[14]), int(p_row[15]), int(p_row[16])])
                                target = np.array([int(p_row[11]), int(p_row[12]), int(p_row[13])])
                                # print(made, target, np.sqrt(np.sum(np.power(made - target, 2))))
                                if np.sqrt(np.sum(np.power(made - target, 2))) < 50:
                                    data.append(np.array(new_word))
                                    # label = int(p_row[11])/255 + 2*int(p_row[12])/255 + 4*int(p_row[13])/255
                                    # labels.append(int(label))
                                    labels.append([int(p_row[11]), int(p_row[12]), int(p_row[13])])
                                
                            new_word = []
                    new_word.append([int(row[8]), int(row[9])])
                    p_word_id = n_word_id
                    p_row = list(row)
                    # participant.append(row[0])
            pickle.dump({'data': data, 'labels': labels}, open(data_fn, 'wb'))
    dataset = pickle.load(open(data_fn, 'rb'))
    labels = np.array(dataset['labels'])
    
    return dataset['data'], labels


def get_config():
    return {
        'conditions': conditions,
        'participants': participants,
        'words': words,
        'days': days,
        'areas': word_areas
    }


def dataset_stats(labels):
    labels = np.array(labels)
    print('Num. participants = {}'.format(len(np.unique(labels[:, 2]))))
    for p in np.unique(labels[:, 2]):
        idx = np.where(labels[:, 2] == p)[0]
        for c in np.unique(labels[idx, 0]):
            idx2 = np.where(labels[idx, 0] == c)[0]
            print('{}: {}\t num. gestures = {}'.format(p, c, len(idx2)))


def clf_accuracies(data, labels, plot=True):
    conds = np.unique(labels[:, 0])
    parts = np.unique(labels[:, 2])
    accuracies = np.zeros((len(conds), len(parts)))
    for ci, c in enumerate(conds):
        for pi, p in enumerate(parts):
            idx_c = np.where(labels[:, 0] == c)[0]
            idx_p = np.where(labels[:, 2] == p)[0]
            idx = list(set(idx_c) & set(idx_p))
            accuracies[ci, pi] = len(
                np.where(list(labels[idx, 3] == labels[idx, 4]))[0]) / len(idx)
    if plot:
        plt.plot(
            np.arange(1, len(conds) + 1), np.mean(accuracies, axis=1), '--o')
        for n in range(len(conds)):
            plt.plot(
                [n + 1, n + 1], [
                    np.mean(accuracies, axis=1)[n],
                    np.mean(accuracies, axis=1)[n] + np.std(
                        accuracies, axis=1)[n]
                ],
                '-k',
                linewidth=1)
            plt.plot(
                [n + 1, n + 1], [
                    np.mean(accuracies, axis=1)[n],
                    np.mean(accuracies, axis=1)[n] - np.std(
                        accuracies, axis=1)[n]
                ],
                '-k',
                linewidth=1)
        plt.savefig('accuracies.pdf', bbox_inches='tight')
        plt.show()
    
    # print(conds)
    writer = csv.writer(open('nativecf_acc.csv', "wt"), delimiter=',')
    writer.writerow(['subject', 'exag_sit', 'exag_walk', 'quick_sit', 'quick_walk'])
    for p in range(accuracies.shape[1]):
        row = [p+1]
        for c in range(accuracies.shape[0]):
            row.append(accuracies[c,p])
        writer.writerow(row)
    from scipy import stats
    tab_test = np.zeros((accuracies.shape[1], 2))
    for p in range(accuracies.shape[1]):
        for ci,c in enumerate([0,2]):
            tab_test[p,ci] = (accuracies[c,p] + accuracies[c+1,p])/2.0
    print('Exag vs Quick', stats.ttest_ind(tab_test[:,0], tab_test[:,1]))
    plt.plot([1,2], [np.mean(tab_test[:,0]), np.mean(tab_test[:,1])], '-')
    tab_test = np.zeros((accuracies.shape[1], 2))
    for p in range(accuracies.shape[1]):
        for ci,c in enumerate([0,1]):
            tab_test[p,ci] = (accuracies[c,p] + accuracies[c+2,p])/2.0
    print('Sit vs Walk', stats.ttest_ind(tab_test[:,0], tab_test[:,1]))
    plt.plot([1,2], [np.mean(tab_test[:,0]), np.mean(tab_test[:,1])], '-')
    plt.show()


    # ofile.close()
    return accuracies





def variability_word_cond(data, labels):
    config = load.get_config()
    conds = np.unique(labels[:, 0])
    parts = np.unique(labels[:, 2])
    variability = np.zeros((len(conds), len(parts)))
    for w in config['words']:
        for ci, c in enumerate(conds):
            for pi, p in enumerate(parts):
                gestures = {}
                idx_c = np.where(labels[:, 0] == c)[0]
                idx_p = np.where(labels[:, 2] == p)[0]
                idx_w = np.where(labels[:, 3] == w)[0]
                idx = list(set(idx_c) & set(idx_p) & set(idx_w))
                # idx = list(set(idx_c) & set(idx_p))
                for i in idx:
                    res = resample(data[i], 50)
                    for d in range(res.shape[1]):
                        if '{}'.format(d) not in gestures.keys():
                            gestures['{}'.format(d)] = []
                        gestures['{}'.format(d)].append(
                            res[:, d].reshape(1, -1))
                var = 0
                for k in gestures.keys():
                    var += np.mean(np.std(np.squeeze(gestures['0']), axis=0))
                variability[ci, pi] = var
        plt.plot(
            np.arange(1, len(conds) + 1),
            np.mean(variability, axis=1),
            '--o',
            label=w)
        for n in range(len(conds)):
            plt.plot(
                [n + 1, n + 1], [
                    np.mean(variability, axis=1)[n],
                    np.mean(variability, axis=1)[n] + np.std(
                        variability, axis=1)[n]
                ],
                '-k',
                linewidth=1)
            plt.plot(
                [n + 1, n + 1], [
                    np.mean(variability, axis=1)[n],
                    np.mean(variability, axis=1)[n] - np.std(
                        variability, axis=1)[n]
                ],
                '-k',
                linewidth=1)
    plt.legend()
    plt.savefig('variability_inst.pdf', bbox_inches='tight')
    plt.show()


def viz_inter_participant_var(data, labels):
    config = get_config()
    conds = np.unique(labels[:, 0])
    parts = np.unique(labels[:, 2])
    variability = np.zeros((len(conds), len(parts)))
    for w in config['words']:
        for pi, p in enumerate(parts):
            for ci, c in enumerate(conds):
                gestures = {}
                idx_c = np.where(labels[:, 0] == c)[0]
                idx_p = np.where(labels[:, 2] == p)[0]
                idx_w = np.where(labels[:, 3] == w)[0]
                idx = list(set(idx_c) & set(idx_p) & set(idx_w))
                plt.subplot(2, 2, ci + 1)
                for i in idx:
                    # data_smooth = savitzky_golay_filter(data[i], fs=100, derivative=0)
                    data_smooth = data[i]
                    plt.plot(data_smooth[:, 0], data_smooth[:, 1])
                plt.title(c)
            plt.savefig('gestures_p{}_{}.pdf'.format(pi + 1, w))
            # plt.show()
            plt.close()



if __name__ == '__main__':
    pickle_data()
