import torch
import scipy.io
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py


class SignalWindow(Dataset):
    def __init__(self, window_data, window_label) -> None:
        super().__init__()
        self.window_data = window_data
        self.window_label = window_label

    def __len__(self):
        return len(self.window_label)
    
    def __getitem__(self, idx):
        return self.window_data[idx, :, :8], self.window_label[idx]


def h5py_to_window(filename):
    file = h5py.File(filename, 'r')
    emg = file['windowData'][:]
    label = file['windowLabel'][:]
    file.close()
    return emg, label


def split_window_ration(emg, label, ratio, window_overlap=200):
    window_size = 400
    denominator = sum(ratio)

    train_emg, train_label, val_emg, val_label, eval_emg, eval_label = [], [], [], [], [], []

    for i in range(len(label)):
        data_len = len(emg[i])
        train_len = int(data_len * ratio[0] / denominator)
        val_len = int(data_len * ratio[1] / denominator)

        emg_type = np.array(emg[i][:train_len])
        window_count = 0
        for j in range(0, len(emg_type) - window_size, window_size - window_overlap):
            train_emg.append(emg_type[j : j + window_size])
            train_label.append(label[i])
            window_count += 1
        # print('{} train window data found in type {} emg signal.'.format(window_count, label[i]))

        emg_type = np.array(emg[i][train_len : train_len + val_len])
        window_count = 0
        for j in range(0, len(emg_type) - window_size, window_size):
            val_emg.append(emg_type[j : j + window_size])
            val_label.append(label[i])
            window_count += 1
        # print('{} val window data found in type {} emg signal.'.format(window_count, label[i]))

        emg_type = np.array(emg[i][train_len + val_len :])
        window_count = 0
        for j in range(0, len(emg_type) - window_size, window_size):
            eval_emg.append(emg_type[j : j + window_size])
            eval_label.append(label[i])
            window_count += 1
        # print('{} eval window data found in type {} emg signal.'.format(window_count, label[i]))

    train_emg = np.array(train_emg)
    train_label = np.array(train_label)
    val_emg = np.array(val_emg)
    val_label = np.array(val_label)
    eval_emg = np.array(eval_emg)
    eval_label = np.array(eval_label)
    return train_emg, train_label, val_emg, val_label, eval_emg, eval_label


def load_emg_label_from_file(filename, class_type=10):
    emg, label = [], []
    for i in range(class_type):
        emg.append([])

    # iterate each file
    mat_file = scipy.io.loadmat(filename)
    file_emg = mat_file['emg']
    file_label = mat_file['restimulus']


    # store one file data except 'rest' action
    for i in range(len(file_label)):
        label_idx = file_label[i][0]
        if label_idx == 0 or label_idx > class_type:
            continue
        movement_idx = label_idx - 1
        if len(emg[movement_idx]) == 0:
            label.append(label_idx)
        emg[movement_idx].append(file_emg[i].tolist())
    print('{} has read, get {} types movement.'.format(filename, class_type))


    print('emg.length = ', len(emg))
    print('label = \n', label)

    return emg, label