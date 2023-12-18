import clip
import torch
import argparse

from torch.optim import lr_scheduler, Adam
from dataloader import SignalWindow, h5py_to_window, split_window_ration, load_emg_label_from_file
from torch.utils.data import DataLoader
from train import train_one_epoch_signal_text, validate_signal_text, evaluate_signal_text
from utils import setup_seed, save_model_weight
from loss import combine_loss

import torch.nn as nn
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="dataset batch size")
    parser.add_argument("--epochs", type=int, default=60, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--dataset", type=str, default="./dataset/img", help="dataset directory path")

    return parser.parse_args()

def main(args):
    setup_seed()
    
    epochs = args.epochs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    filename = 'dataset/window.h5'
    weight_path = 'res/best.pt'
    best_precision, current_precision = 0, 0

    model_dim = 2 # 数据维数 1为(B,8,400,1)，2为(B,1,400,8)
    classification = True # 是否是分类任务
    model = clip.EMGbuild_model(classification=classification, model_dim=model_dim)
    
    # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.2)
    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_func = combine_loss

    emg, label = h5py_to_window(filename)
    data_len = len(label)
    index = np.random.permutation(data_len)
    emg = emg[index] * 20000
    label = label[index]

    train_len = int(data_len * 0.8)
    val_len = int(data_len * 0.1)
    print('{} window for training, {} window for validation and {} data for test.'.format \
          (train_len, val_len, data_len - train_len - val_len))

    # 数据按照8:1:1分为训练集、验证集和测试集
    train_emg = emg[: train_len]
    train_label = label[: train_len]
    val_emg = emg[train_len : train_len + val_len]
    val_label = label[train_len : train_len + val_len]
    eval_emg = emg[train_len + val_len :]
    eval_label = label[train_len + val_len :]

    # data_filename = 'D:/Download/Datasets/Ninapro/DB2/S1/S1_E1_A1.mat'
    # emg, label = load_emg_label_from_file(data_filename)
    # # [(4,1,1), 200] [(2,1,1), 300]
    # train_emg, train_label, val_emg, val_label, eval_emg, eval_label = split_window_ration(emg, label, (4,1,1), window_overlap=200)
    # train_index = np.random.permutation(len(train_emg))
    # val_index = np.random.permutation(len(val_emg))
    # eval_index = np.random.permutation(len(eval_emg))
    # train_emg, train_label = train_emg[train_index] * 20000, train_label[train_index]
    # val_emg, val_label = val_emg[val_index] * 20000, val_label[val_index]
    # eval_emg, eval_label = eval_emg[eval_index] * 20000, eval_label[eval_index]


    train_loader = DataLoader(
                    SignalWindow(train_emg, train_label),
                    batch_size=args.batch_size,
                    num_workers=0
                    )
    
    val_loader = DataLoader(
                    SignalWindow(val_emg, val_label),
                    batch_size=8,
                    num_workers=0
                    )

    eval_loader = DataLoader(
                    SignalWindow(eval_emg, eval_label),
                    batch_size=8,
                    num_workers=0
                    )

    if classification:
        print('-------- Classification task --------')
    else:
        print('-------- Pair task --------')
    print("{}D signal input.".format(model_dim))
    model.train().half()
    model.to(device)
    print("start training...")
    for epoch in range(epochs):
        train_one_epoch_signal_text(model, epoch, epochs, device, train_loader, loss_func, 
                                    optimizer, scheduler, classification=classification, model_dim=model_dim)
        current_precision = validate_signal_text(model, device, val_loader, loss_func, classification=classification, model_dim=model_dim)

        if current_precision > best_precision:
            best_precision = current_precision
            save_model_weight(model=model, filename=weight_path)

    print('\nCurrent best precision in val set is: %.4f' % (best_precision * 100) + '%')
    model.load_state_dict(torch.load(weight_path))
    evaluate_signal_text(model, device, eval_loader, loss_func, classification=classification, model_dim=model_dim)

if  __name__ == "__main__":
    args = arg_parse()
    main(args)
