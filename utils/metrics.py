# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:42:13 2022

@author: AA
"""

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def R2(pred, true):
    test_mean = np.mean((true[~np.isnan(true)]))
    err = pred - true
    s_err = err**2
    m_err = (true - test_mean)**2
    return 1 - np.sum(s_err[~np.isnan(s_err)])/np.sum(m_err[~np.isnan(m_err)])

def SMAPE(a, f):
    return np.mean(2 * np.abs(f-a) / (np.abs(a) + np.abs(f) + 1e-6))


def metric(pred, true, mask = np.array([None])):
    if mask.any() == None:
        mask = np.ones(pred.shape)
    pred = pred[mask > 0]
    true = true[mask > 0]
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    r2 = R2(pred, true)
    smape = SMAPE(pred, true)
    return mae, mse, rmse, rse, corr, r2, smape