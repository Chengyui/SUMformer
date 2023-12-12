# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:15:17 2023

@author: AA
"""

import torch.utils.data as torch_data
import numpy as np
import torch
from torch.utils.data import DataLoader

def time_continues(timestamps):
    fixed_interval = timestamps[1] - timestamps[0]  # calculate the interval
    
    is_continuous = True
    
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i-1]
        if interval != fixed_interval:
            is_continuous = False
            return is_continuous
            break
    return is_continuous

def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        # std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastGrid(torch_data.Dataset):
    def __init__(self, data, time_vec, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1, prefer_shape = None):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        self.data = data
        if not prefer_shape is None:
            if prefer_shape == ['T', 'C']:
                self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1] * self.data.shape[2] * self.data.shape[3]])
        self.df_length = self.data.shape[0]
        self.time_vec = time_vec
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        while True:
            hi = self.x_end_idx[index] #12
            lo = hi - self.window_size #0
            train_data = self.data[lo: hi] #0:12
            target_data = self.data[hi:hi + self.horizon] #12:24
            target_time = self.time_vec[lo: hi + self.horizon]         
            if time_continues(target_time):
                x = torch.from_numpy(train_data).type(torch.float)
                y = torch.from_numpy(target_data).type(torch.float)
                return x, y      
            index += 1
            if index >= len(self.x_end_idx):
                index = 0
        
    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx
    
    
# if __name__ ==  '__main__':
#     data = np.load('taxibj/taxibj.npy')
#     length = data.shape[0]
#     data_train = data[:int(length*0.7)]
#     tvec = np.load('taxibj/taxibj_time.npy', allow_pickle=True)
#     tvec_train = tvec[:int(length*0.7)]
#     train_mean = np.mean(data_train, axis=0)
#     train_std = np.std(data_train, axis=0)
#     train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
#     train_set = ForecastGrid(data_train, tvec_train, window_size=12, horizon=12,
#                             normalize_method='z_score', norm_statistic=train_normalize_statistic)
    
#     train_loader = DataLoader(train_set, batch_size=32, drop_last=False, shuffle=True,
#                                         num_workers=1)
    
#     for i, (inputs, target) in enumerate(train_loader):
#         print(inputs.shape, target.shape)