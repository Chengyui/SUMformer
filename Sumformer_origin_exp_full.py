import numpy as np
import argparse
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.PeMS_dataset import de_normalized, ForecastGrid
# from model.CNN.Ours_origin import SimVP_Model
from model.sumformer.sumformer import Sumformer
# from model.sumformer.sumformer_vit import Sumformer
from utils.metrics import MAE, RMSE, SMAPE, MSE
# from plot.plot_TS import plot_12
import time

from timm.scheduler.cosine_lr import CosineLRScheduler
import os
import random
from PeakLoss.peakloss import peak_loss

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def plot(MSE_list, path):
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(MSE_list) + 1))
    plt.plot(epochs, MSE_list, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss per Epoch minist MSE:{}'.format(min(MSE_list)))
    plt.grid(True)

    plt.savefig(path + ".png")


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='GPU or CPU')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.2], help='train/val/test ratio', type=float)
parser.add_argument('--batch', type=int, default=16, help='training batch size')
parser.add_argument('--warmup_lr', type=float, default=1e-5, help='warmup_lr')
parser.add_argument('--warmup_epoch', type=int, default=5, help='warmup_epoch')
parser.add_argument('--drop_path', type=float, default=0.1, help='drop_path')
parser.add_argument('--sched', type=str, default='cosine', help='training schedule')
parser.add_argument('--spatio_kernel_enc', type=int, default=3, help='encoder spatial kernel size')
parser.add_argument('--spatio_kernel_dec', type=int, default=3, help='decoder spatial kernel size')
parser.add_argument('--hid_S', type=int, default=32, help='hidden spatial dimension,hidden Variable dim')
parser.add_argument('--hid_T', type=int, default=256, help='hidden temporal dimension')
parser.add_argument('--N_T', type=int, default=128, help='number of forecasting steps')
parser.add_argument('--N_S', type=int, default=2, help='')
parser.add_argument('--In_T', type=int, default=128, help='number of input steps')
parser.add_argument('--Variable', type=int, default=2, help='number of variables')
parser.add_argument('--SpaceH', type=int, default=32, help='number of grid points')
parser.add_argument('--SpaceW', type=int, default=32, help='number of grid points')
parser.add_argument('--Epoch', type=int, default=80, help='number of epoches')
parser.add_argument('--lr', type=float, default=5e-4, help='number of epoches')
parser.add_argument('--pth', type=str, default='pth/SimVP_car_chengdu_full.pth', help='number of epoches')
parser.add_argument('--pretrain_pth', type=str, default='pth/SimVP_car_chengdu_full.pth', help='number of epoches')
parser.add_argument('--no_hid', action='store_true', help='Set this flag to True.')
parser.add_argument('--dilat', action='store_true', help='Set this flag to True.')
parser.add_argument('--test', action='store_true', help='Set this flag to True.')
parser.add_argument('--seg_len',type=int,default=16)
parser.add_argument('--router_factor',type=int,default=256)
parser.add_argument('--d_model',type=int,default=128)
parser.add_argument('--e_layers',type=int,default=4)
parser.add_argument('--n_heads',type=int,default=4)
parser.add_argument('--win_size',type=int,default=2)
parser.add_argument('--dataset',type=str,default='taxibj',help='taxibj,Chengdu,NYC')
parser.add_argument('--peak_loss', action='store_true', help='Set this flag to True.')
parser.add_argument('--layer_scaler',type=float,default=1)
parser.add_argument('--Peak_eval',action='store_true')
parser.add_argument('--accu_step',type=int,default=1,help='accumulative loss steps for saving memory')
parser.add_argument('--layer_type',type=str,default='MD',help='choose the variant type for SUMformer')
parser.add_argument('--layer_depth', default=[2,2,6,2], type=int,nargs='*',help ='The depth for each TVF block')
args = parser.parse_args()

def adjust_learning_rate(optimizer, epoch, learning_rate):

    lr_adjust = {80:learning_rate*0.5,90:learning_rate*0.25}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

if __name__ == '__main__':
    '''
    Read data
    '''
    print(args)
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.dataset =='taxibj':
        data = np.load('datasets/taxibj/taxibj.npy')
        tvec = np.load('datasets/taxibj/taxibj_time.npy', allow_pickle=True)
    elif args.dataset=='Chengdu':
        data = np.load('datasets/Chengdu/Chengdu_phv_TS_standard.npy')
        tvec = np.load('datasets/Chengdu/phv_timestamp.npy', allow_pickle=True)
    elif args.dataset=='NYC':
        data = np.load('datasets/NYC/NYC_2015.npy')
        tvec = np.load('datasets/NYC/NYC_timestamp.npy', allow_pickle=True)
        args.SpaceH,args.SpaceW = 10,20
    index = 0
    for d in data:
        if np.allclose(d, np.zeros_like(d)):
            print(index)
        index += 1
    length = data.shape[0]
    data_train = data[:int(length * args.train_val_ratio[0])]
    tvec_train = tvec[:int(length * args.train_val_ratio[0])]
    train_mean = np.mean(data_train)
    train_std = np.std(data_train)
    train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    train_set = ForecastGrid(data_train, tvec_train, window_size=args.In_T, horizon=args.N_T,
                             normalize_method='z_score', norm_statistic=train_normalize_statistic, interval=1)

    data_test = data[-int(length * args.train_val_ratio[1]):]
    tvec_test = tvec[
                -int(length * args.train_val_ratio[1]):]
    test_set = ForecastGrid(data_test, tvec_test, window_size=args.In_T, horizon=args.N_T,
                            normalize_method='z_score', norm_statistic=train_normalize_statistic)

    data_val = data[int(length * args.train_val_ratio[0]):-int(length * args.train_val_ratio[1])]
    tvec_val = tvec[
                int(length * args.train_val_ratio[0]):-int(length * args.train_val_ratio[1])]
    val_set = ForecastGrid(data_val, tvec_val, window_size=args.In_T, horizon=args.N_T,
                            normalize_method='z_score', norm_statistic=train_normalize_statistic)

    train_loader = DataLoader(train_set, batch_size=args.batch, drop_last=False, shuffle=True,
                              num_workers=1)

    test_loader = DataLoader(test_set, batch_size=args.batch, drop_last=False, shuffle=True,
                             num_workers=1)
    val_loader = DataLoader(val_set, batch_size=args.batch, drop_last=False, shuffle=True,
                             num_workers=1)

    '''
    Define model
    '''
    model = Sumformer(args.SpaceH*args.SpaceW*args.Variable, args.In_T,args.N_T,args.seg_len,\
                        device=args.device,factor=args.router_factor,\
                        e_layers=args.e_layers,d_model=args.d_model,\
                        n_heads=args.n_heads,win_size=args.win_size,layer_scaler=args.layer_scaler,layer_type = args.layer_type,layer_depth = args.layer_depth)

    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criteria = nn.MSELoss()
    peak = peak_loss(48,spatial=True)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.Epoch,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epoch,
        t_in_epochs=True  # update lr by_epoch(True) steps(False)
    )
    MSE_list = []

    '''
    Training and validation
    '''
    train_minibatches = len(train_loader)
    val_minibatches = len(val_loader)
    test_minibatches = len(test_loader)
    begin_time = time.time()

    for epoch in range(args.Epoch):
        print("epoch: {}".format(epoch))
        loss_cum = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.train()
            output = model(inputs,frozen=False)
            loss = criteria(output, target)
            if args.peak_loss:
                p_loss = peak(output,target)
                loss = loss+0.2*p_loss

            loss = loss/args.accu_step
            loss.backward()
            # optimizer.step()
            if (i+1)%args.accu_step==0:
                optimizer.step()
                optimizer.zero_grad()
            loss_cum += loss.item()*args.accu_step
            if i % 100 == 0:
                end_time = time.time()
                print("time comsuming: {}".format(end_time - begin_time))
                print("training epoch:{}:{}%".format(epoch, i / train_minibatches * 100))
        end_time = time.time()
        print("time comsuming: {}".format(end_time - begin_time))
        if optimizer.param_groups[0]['lr']>0.00015 or epoch<5:
            scheduler.step(epoch)
        # adjust_learning_rate(optimizer,epoch,args.lr)
        print("Adam lr epoch:{} lr:{}".format(epoch, optimizer.param_groups[0]['lr']))
        print("MSE loss :{}".format(loss_cum / train_minibatches))
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
        model.eval()
        outputs = []
        vals = []
        with torch.no_grad():
            for i, (inputs, val) in enumerate(val_loader):
                inputs = inputs.to(args.device)
                output = model(inputs)
                outputs.append(output.detach().cpu().numpy())
                vals.append(val.detach().numpy())
                if i % 100 == 0:
                    end_time = time.time()
                    print("time comsuming: {}".format(end_time - begin_time))
                    print("test epoch:{}:{}%".format(epoch, i / val_minibatches * 100))

        mse, mmae = MSE(np.concatenate(outputs), np.concatenate(vals)), MAE(np.concatenate(outputs),
                                                                             np.concatenate(vals))

        outputs = de_normalized(np.concatenate(outputs), 'z_score', train_normalize_statistic)
        outputs[outputs < 0] = 0  # physical limits
        outputs = np.round(outputs)
        vals = de_normalized(np.concatenate(vals), 'z_score', train_normalize_statistic)
        mae, rmse, smape = MAE(outputs, vals), RMSE(outputs, vals), SMAPE(outputs, vals)
        print('episode', epoch, 'mae', mae, 'rmse', rmse, 'smape', smape, 'mse', mse, 'mmae', mmae)

        MSE_list.append(rmse)
        if rmse == min(MSE_list):
            torch.save(model.state_dict(), args.pth)
        plot(MSE_list, args.pth[:-4])
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
    plot(MSE_list, args.pth[:-4])

    '''
    Testing
    '''
    outputs = []
    tests = []
    model = model.load_state_dict(torch.load(args.pth, map_location=args.device))
    model.eval()
    with torch.no_grad():
        for i, (inputs, test) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            output = model(inputs)
            outputs.append(output.detach().cpu().numpy())
            tests.append(test.detach().numpy())

    mse, mmae = MSE(np.concatenate(outputs), np.concatenate(tests)), MAE(np.concatenate(outputs),
                                                                         np.concatenate(tests))

    outputs = de_normalized(np.concatenate(outputs), 'z_score', train_normalize_statistic)
    outputs[outputs < 0] = 0  # physical limits
    outputs = np.round(outputs)
    tests = de_normalized(np.concatenate(tests), 'z_score', train_normalize_statistic)

    mae, rmse, smape = MAE(outputs, tests), RMSE(outputs, tests), SMAPE(outputs, tests)
    print('mae', mae, 'rmse', rmse, 'smape', smape, 'mse', mse, 'mmae', mmae)
