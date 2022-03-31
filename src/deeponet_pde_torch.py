from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import argparse
import os 
import sys 
import logging 
import time

import numpy as np
import torch 
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import tensorflow as tf

# import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

parser = argparse.ArgumentParser(description='DeepONet')

parser.add_argument('--name', type=str, default='debug', help='Experiments name')
parser.add_argument('--save_path', type=str, default='exp', help='Experiments path')
parser.add_argument('--num_train', type=int, default=5000, help='Num of training samples')
parser.add_argument('--num_test', type=int, default=100, help='Num of test samples')
parser.add_argument('--save_data', action='store_true', help='Saving data')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--m', type=int, default=100)

parser.add_argument('--activation', type=str, default='tanh', help='Experiments path')

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s):
        self.u = torch.tensor(u)#.cuda() 
        self.y = torch.tensor(y)#.cuda() 
        self.s = torch.tensor(s)#.cuda()  

        self.data_size = u.shape[0]
    
    def __getitem__(self, index):
        return self.u[index], self.y[index], self.s[index]

    def __len__(self):
        return self.data_size 
        
class DataSampler(data.Dataset):
    def __init__(self, u, y, s, batch_size):
        self.u = torch.tensor(u).cuda() 
        self.y = torch.tensor(y).cuda() 
        self.s = torch.tensor(s).cuda()  
        self.batch_size = batch_size
        self.data_size = u.shape[0]
        self.index = range(self.data_size)
    
    def __getitem__(self, index):
        idx = np.random.choice(self.index, self.batch_size, replace=False)
        return self.u[idx], self.y[idx], self.s[idx]

    def __len__(self):
        return self.data_size 
    
class Sine(torch.nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class DeepONet(nn.Module):
    def __init__(self, trunk_layer, branch_layer, activation, init=True):
        super(DeepONet, self).__init__()
        
        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'softplus':
            self.activation = torch.nn.Softplus
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        elif activation == 'logsigmoid':
            self.activation = torch.nn.LogSigmoid

        layer_trunk = []
        for i in range(len(trunk_layer)-2):
            layer_trunk.append(torch.nn.Linear(trunk_layer[i], trunk_layer[i+1]).double())
            layer_trunk.append(self.activation())
        layer_trunk.append(torch.nn.Linear(trunk_layer[-2], trunk_layer[-1]).double())
        self.layer_trunk = torch.nn.Sequential(*layer_trunk)

        layer_branch = []
        for i in range(len(branch_layer)-2):
            layer_branch.append(torch.nn.Linear(branch_layer[i], branch_layer[i+1]).double())
            layer_branch.append(self.activation())
        layer_branch.append(torch.nn.Linear(branch_layer[-2], branch_layer[-1]).double())
        self.layer_branch = torch.nn.Sequential(*layer_branch)

        if init:
            for name,m in self.layer_trunk.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for name,m in self.layer_branch.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, u, y):
        branch_output = self.layer_branch(u)
        trunk_output = self.layer_trunk(y)
        return torch.sum(branch_output * trunk_output, dim=-1, keepdims=True)

def lt_system(npoints_output):
    """Legendre transform"""
    return LTSystem(npoints_output)


def ode_system(T):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        # k = 1
        # return [s[1], - k * np.sin(s[0]) + u]

    s0 = [0]
    # s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T)


def dr_system(T, npoints_output):
    """Diffusion-reaction"""
    D = 0.01
    k = 0.01
    Nt = 100
    return DRSystem(D, k, T, Nt, npoints_output)


def cvc_system(T, npoints_output):
    """Advection"""
    f = None
    g = None
    Nt = 100
    return CVCSystem(f, g, T, Nt, npoints_output)


def advd_system(T, npoints_output):
    """Advection-diffusion"""
    f = None
    g = None
    Nt = 100
    return ADVDSystem(f, g, T, Nt, npoints_output)


def run(args, problem, system, space, T, m, nn, model, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    if args.save_data:
        (u_train, y_train), s_train = system.gen_operator_data(space, m, num_train)
        (u_test, y_test), s_test = system.gen_operator_data(space, m, num_test, test=True)
        print(u_train.shape, y_train.shape,s_train.shape)
        print(u_test.shape, y_test.shape, s_test.shape)
        np.savez_compressed(os.path.join('data',f"train_{args.num_train}.npz"), u_train=u_train, y_train=y_train, s_train=s_train)
        np.savez_compressed(os.path.join('data',f"test_{args.num_test}.npz"), u_test=u_test, y_test=y_test, s_test=s_test)
        return
    # # debug 
    # import pdb 
    # pdb.set_trace()
    # y_test = y_test.reshape(-1, 100*100)
    # t_min, t_max = X_test[1][:, 1].min(), X_test[1][:, 1].max()
    # x_min, x_max = X_test[1][:, 0].min(), X_test[1][:, 0].max()
    # for i,img in enumerate(y_test):
    #     fig = plt.figure()
    #     ax0 = fig.add_subplot(111)

    #     h0 = ax0.imshow(img.reshape(100,100), interpolation='nearest', cmap='rainbow',
    #                     extent=[t_min,t_max,x_min,x_max],
    #                     origin='lower', aspect='auto')
    #     divider0 = make_axes_locatable(ax0)
    #     cax0 = divider0.append_axes("right", size="5%", pad=0.10)
    #     cbar0 = fig.colorbar(h0, cax=cax0)
    #     cbar0.ax.tick_params(labelsize=15)
    #     ax0.set_title('y_test')

    #     ax0.set_xlabel('t', fontweight='bold', size=15)
    #     ax0.set_ylabel('x', fontweight='bold', size=15)
    #     plt.savefig(os.path.join(args.name, f'y_test_{i}.png'))  

    d = np.load(f"data/train_{args.num_train}.npz")
    u_train, y_train, s_train = d["u_train"], d["y_train"], d["s_train"]
    d = np.load(f"data/test_{args.num_test}.npz")
    u_test, y_test, s_test = d["u_test"], d["y_test"], d["s_test"]

    # X_test_trim = trim_to_65535(X_test)[0]
    # y_test_trim = trim_to_65535(y_test)[0]
    
    train_data = DataGenerator(u_train, y_train, s_train)
    test_data = DataGenerator(u_test, y_test, s_test)
    train_loader = data.DataLoader(train_data, batch_size=10000, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    
    # train_data = DataSampler(u_train, y_train, s_train, batch_size=10000)
    # test_data = DataSampler(u_test, y_test, s_test, batch_size=10000)
    # train_loader = iter(train_data)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr, betas=(0.9, 0.999), eps=1e-8)
    schduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/2000))
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    writer = SummaryWriter(log_dir=args.name) 
    step = 0
    for i in range(epochs):
        model.train()

        end = time.time()
        for j, (u, y, s) in enumerate(train_loader):
            # measure data loading time
            u = u.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            s = s.cuda(non_blocking=True)
            # u,y,s = next(train_loader)
            data_time = time.time() - end

            # compute output
            output = model(u,y)
            loss = criterion(output, s)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            writer.add_scalars('loss', {'train': loss.detach()}, step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            
            if (step+1) % 100 == 0:
                logging.info(f'{[step+1]} data: {batch_time:.3f}({data_time:.3f}) loss: {loss:.4e}')
            
            if (step+1) % 1000 == 0:

                save_checkpoint({
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, args.name)

                model.eval()
                losses = []
                target = []
                pred = []
                with torch.no_grad():
                    for k, (u, y, s) in enumerate(test_loader):
                        # measure data loading time
                        u = u.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)
                        s = s.cuda(non_blocking=True)
                        data_time = time.time() - end

                        # compute output
                        output = model(u,y)
                        loss = criterion(output, s)
                        losses.append(u.shape[0] * loss)
                        target.append(s)
                        pred.append(output)
                        # measure elapsed time
                        batch_time = time.time() - end
                        end = time.time()
                        # if (k+1) % 100 == 0:
                        #     logging.info(f'Test: {[k+1]} data: {batch_time:.3f}({data_time:.3f}) loss: {loss:.4e}')
                    losses = sum(losses) / len(test_data)

                    target = torch.cat(target, dim=0).reshape(-1, 100*100)
                    pred = torch.cat(pred, dim=0).reshape(-1, 100*100)
                    error = torch.linalg.norm(pred-target, dim=-1) / torch.linalg.norm(target, dim=-1)
                    logging.info(f'test loss {losses:.4e}, L2-error {error.mean():.4e}')
                    writer.add_scalars('loss', {'test': losses}, step)
                    writer.add_scalars('error', {'test': error.mean()}, step)
            step += 1
            schduler.step()

def save_checkpoint(state, save, is_best=False, filename='checkpoint.pth.tar'):
    filename = os.path.join(save, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save, 'model_best.pth.tar'))

def main():
    args = parser.parse_args()
    args.name = '{}-{}'.format(args.name,
            time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.name = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.name):
        os.mkdir(args.name)
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    logging.info("args = %s", args)

    # Problems:
    # - "lt": Legendre transform
    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    # - "dr": Diffusion-reaction
    # - "cvc": Advection
    # - "advd": Advection-diffusion
    problem = "dr"
    T = 1
    if problem == "lt":
        npoints_output = 20
        system = lt_system(npoints_output)
    elif problem == "ode":
        system = ode_system(T)
    elif problem == "dr":
        npoints_output = 100
        system = dr_system(T, npoints_output)
    elif problem == "cvc":
        npoints_output = 100
        system = cvc_system(T, npoints_output)
    elif problem == "advd":
        npoints_output = 100
        system = advd_system(T, npoints_output)

    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    space = GRF(1, length_scale=0.2, N=512, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    m = args.m 
    num_train = args.num_train 
    num_test = args.num_test
    lr = args.lr
    epochs = args.epoch 
    # device = torch.device('cuda')
    # Network
    # nn = "opnn"
    # activation = "relu"
    # initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    # dim_x = 1 if problem in ["ode", "lt"] else 2
    # if nn == "opnn":
    #     net = dde.maps.OpNN(
    #         [m, 100, 100],
    #         [dim_x, 100, 100],
    #         activation,
    #         initializer,
    #         use_bias=True,
    #         stacked=False,
    #     )
    # elif nn == "fnn":
    #     net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    # elif nn == "resnet":
    #     net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)
    branch_layer = [m, 50, 50, 50, 50, 50]
    trunk_layer =  [2, 50, 50, 50, 50, 50]
    
    net = DeepONet(trunk_layer, branch_layer, activation=args.activation, init=True).cuda()
    logging.info(net)
    
    run(args, problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)


if __name__ == "__main__":
    main()
