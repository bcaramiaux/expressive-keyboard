import numpy as np
import os
import sys
import pickle
from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter, savgol_coeffs, filtfilt
import load
import utils


def savitzky_golay_filter(data, fs=240, derivative=1):
    filtered_data = data.copy()
    win_size = 19
    # print(win_size, data.shape[0])
    if data.shape[0] < win_size:
        if (data.shape[0] - 1) % 2 == 0:
            win_size = data.shape[0] - 2
        else:
            win_size = data.shape[0] - 1
    # print(data.shape[0], win_size)
    if (win_size > 1):
        if (len(data.shape) > 1):
            for d in range(data.shape[1]):
                filtered_data[:, d] = savgol_filter(
                    data[:, d], win_size, 3, deriv=derivative) * np.power(
                        fs, derivative)
        else:
            filtered_data = savgol_filter(
                data, win_size, 3, deriv=derivative) * np.power(
                    fs, derivative)
        return filtered_data
    else:
        return []


def adhoc_features(data,
                   labels,
                   feat_type='kinematics',
                   granularity='gesture',
                   out_fn="features"):
    print('\033[95m' + 'Processing features' + '\033[0m')
    output_filename = out_fn + '_' + feat_type
    
    # if granularity == 'gesture':
    #     average = True
    # elif granularity == 'sample':
    #     average = False
        
    if granularity == 'gesture':
        output_filename += '_avg'
    elif  granularity == 'sequence':
        output_filename += '_seq'
    output_filename += '.pkl'

    if os.path.isfile(output_filename):
        print('\033[92m' + "'" + output_filename + "' exists, loading" +
              '\033[0m')
        data_lbls = pickle.load(open(output_filename, 'rb'))
        newdata = data_lbls['data']
        newlabels = data_lbls['labels']
    else:
        print('\033[93m' + "'" + output_filename + "' doesn't exist, creating"
              + '\033[0m')
        newdata = []
        newlabels = []
        for gi, g in enumerate(data):
            utils.progress_bar(gi + 1, len(data))
            tmp_data = np.float32(np.asarray(g)) / 1024.0
            if feat_type == 'kinematics':
                pos = np.float32(np.asarray(g)) / 1024.0
                vel = savitzky_golay_filter(tmp_data, fs=100, derivative=1)
                for v in vel:
                    if v[0] == 0 or v[1] ==0:
                        print(v)
                acc = savitzky_golay_filter(tmp_data, fs=100, derivative=2)
                jer = savitzky_golay_filter(tmp_data, fs=100, derivative=3)
                tmp_data = np.c_[
                    pos, \
                    vel, \
                    acc, \
                    jer, \
                    np.sqrt(np.sum(np.power(vel,2), axis=1)), \
                    np.sqrt(np.sum(np.power(acc,2), axis=1)), \
                    np.sqrt(np.sum(np.power(jer,2), axis=1))
                    ]
                if granularity == 'sample':
                    newdata = np.r_[newdata, tmp_data] if len(newdata) else tmp_data
                    # print(labels)
                    newlabels.extend([labels[gi] for i in range(len(g))])
                elif granularity == 'exp':
                    tmp_data_ = []
                    tmp_data__ = []
                    # print(np.max(tmp_data[:,8]), np.min(tmp_data[:,8]), np.mean(tmp_data[:,8]))
                    for i in range(len(tmp_data)):
                        if tmp_data[i, 8] > 0.0:
                            tmp_data__.append(tmp_data[i,:])
                            if len(tmp_data__) == 3:
                                tmp_data_.append(np.mean(tmp_data__, axis=0))
                                tmp_data__ = []
                    tmp_data_ = np.array(tmp_data_)
                    newdata = np.r_[newdata, tmp_data_] if len(newdata) else tmp_data_
                    newlabels.extend([labels[gi] for i in range(len(tmp_data_))])
                elif granularity == 'gesture':
                    newdata.append(np.mean(tmp_data, axis=0))
                    newlabels.append(labels[gi])
                elif granularity == 'sequence':
                    seq_n = 40
                    hop_n = 5 
                    chunk_n = int((len(tmp_data) - seq_n) / hop_n)
                    for i in range(chunk_n):
                        # if len(np.where(np.isnan(tmp_data[i*hop_n:i*hop_n+seq_n,:]) == True)[0]):
                        #     print(pos, vel)
                        newdata.append(tmp_data[i*hop_n:i*hop_n+seq_n,:])
                        newlabels.append(labels[gi])
            elif feat_type == 'uist':
                velocities = savitzky_golay_filter(tmp_data, fs=100, derivative=1)
                half_len = int(len(velocities)/2)
                speed_ratio = np.mean(np.sum(np.power(velocities[:half_len,:],2),axis=1)) / \
                                np.mean(np.sum(np.power(velocities[half_len:,:],2),axis=1))
                curvature = []
                for i in range(len(tmp_data) - 2):
                    u = tmp_data[i + 1, :] - tmp_data[i, :]
                    v = tmp_data[i + 2, :] - tmp_data[i + 1, :]
                    curvature.append(
                        np.abs(
                            np.arctan2(
                                np.abs(np.cross(u, v)), (u[0] * v[0] +
                                                         u[1] * v[1]))))
                curviness = np.std(curvature)
                config = load.get_config()
                g_box = np.max(tmp_data, axis=0) - np.min(tmp_data, axis=0)
                # print(labels[gi][3], config['areas'][config['words'].index(labels[gi][3])])
                inflation = g_box[0] * g_box[1] / config['areas'][config['words'].index(labels[gi][3])]
                newdata.append(
                    np.array([
                        speed_ratio,
                        curviness,
                        inflation
                    ]))
                newlabels.append(labels[gi])
        utils.close_progress_bar()
        newdata = np.array(newdata)
        newlabels = np.array(newlabels)
        pickle.dump({
            'data': newdata,
            'labels': newlabels
        }, open(output_filename, 'wb'))
    return newdata, newlabels


def resample(data, n):
    data_res = np.array(data)
    length = 0.0
    for i in range(1, len(data_res)):  #(var i = 1; i < points.length; i++)
        length += np.sqrt(
            np.sum(np.power(data_res[i - 1, :] - data_res[i, :], 2)))
    I = length / (n - 1)
    #interval length
    D = 0.0
    newpoints = []
    newpoints.append(data_res[0])
    i = 1
    while (i < len(data_res)):
        # for i in range(1, len(data_res)): #(let i = 1; i < data_res.length; i++)
        d = np.sqrt(np.sum(np.power(data_res[i - 1, :] - data_res[i, :],
                                    2)))  #(data_res[i - 1], data_res[i]);
        # print(i, d, I, data_res[i-1,:], data_res[i,:])
        if ((D + d) >= I):
            q = data_res[i - 1, :] + (
                (I - D) / d) * (data_res[i, :] - data_res[i - 1, :])
            newpoints.append(q)
            data_res = np.insert(
                data_res, i, q, axis=0
            )  #insert 'q' at position i in points s.t. 'q' will be the next i
            # print(((I - D) / d),data_res)
            D = 0.0
        else:
            D += d
        i += 1
    if (len(newpoints) == n - 1):
        newpoints.append(data_res[len(data_res) - 1, :])
    newpoints = np.array(newpoints)
    return newpoints