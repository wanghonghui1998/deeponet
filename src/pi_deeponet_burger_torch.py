from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import argparse
import os 
import sys 
import logging 
import time
from scipy import io

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

parser = argparse.ArgumentParser(description='PI-DeepONet')

parser.add_argument('--name', type=str, default='debug', help='Experiments name')
parser.add_argument('--save_path', type=str, default='exp', help='Experiments path')
parser.add_argument('--num_train', type=int, default=5000, help='Num of training samples')
parser.add_argument('--num_test', type=int, default=100, help='Num of test samples')
parser.add_argument('--save_data', action='store_true', help='Saving data')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--m', type=int, default=101)

parser.add_argument('--activation', type=str, default='tanh', help='Experiments path')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, enlarge=1):
        self.u = torch.tensor(u, dtype=torch.float64)#.cuda() 
        self.y = torch.tensor(y, dtype=torch.float64)#.cuda() 
        self.s = torch.tensor(s, dtype=torch.float64)#.cuda()  

        self.data_size = u.shape[0]
        self.enlarge = enlarge
    
    def __getitem__(self, index):
        index = index % self.data_size
        return self.u[index], self.y[index], self.s[index]

    def __len__(self):
        return self.data_size * self.enlarge 
    
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
                    nn.init.xavier_normal_(m.weight)
                    print('init xavier')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for name,m in self.layer_branch.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_normal_(m.weight)
                    print('init xavier')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, u, y):
        branch_output = self.layer_branch(u)
        trunk_output = self.layer_trunk(y)
        return torch.sum(branch_output * trunk_output, dim=-1, keepdims=True)

class PIDeepONet():
    def __init__(self, trunk_layer, branch_layer, activation, init=True):
        # super(PIDeepONet, self).__init__()
        
        self.net = DeepONet(trunk_layer, branch_layer, activation, init=True).cuda()

    def loss_res(self, u_res, t_res, x_res):
        s_pred = self.net(u_res, torch.cat([t_res, x_res], dim=-1))
        s_x = torch.autograd.grad(
                s_pred, x_res,
                grad_outputs=torch.ones_like(s_pred),
                retain_graph=True,
                create_graph=True
            )[0]
        s_t = torch.autograd.grad(
                s_pred, t_res,
                grad_outputs=torch.ones_like(s_pred),
                retain_graph=True,
                create_graph=True
            )[0]
        s_xx = torch.autograd.grad(
                s_x, x_res,
                grad_outputs=torch.ones_like(s_x),
                retain_graph=True,
                create_graph=True
            )[0]
        res = s_t + s_pred * s_x - 0.01 * s_xx
        return torch.mean((res)**2)

    def loss_bcs(self, u_bcs, t_bcs_1, x_bcs_1, t_bcs_2, x_bcs_2):
        s_1 = self.net(u_bcs, torch.cat([t_bcs_1, x_bcs_1], dim=1))
        s_2 = self.net(u_bcs, torch.cat([t_bcs_2, x_bcs_2], dim=1))

        s_x_1 = torch.autograd.grad(
                s_1, x_bcs_1,
                grad_outputs=torch.ones_like(s_1),
                retain_graph=True,
                create_graph=True
            )[0]
        s_x_2 = torch.autograd.grad(
                s_2, x_bcs_2,
                grad_outputs=torch.ones_like(s_2),
                retain_graph=True,
                create_graph=True
            )[0]

        return torch.mean((s_1-s_2)**2) + torch.mean((s_x_1-s_x_2)**2)

    def loss_ics(self, u_ics, y_ics, s_ics):
        pred = self.net(u_ics, y_ics)
        return torch.mean((pred-s_ics)**2)
    # def loss(self, u_res, x_res, t_res, s_res, u_bcs, y_bcs, s_bcs):
    #     loss_res = 



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

def load_data(args):
    # Geneate ics training data corresponding to one input sample
    def generate_one_ics_training_data(u0, m=101, P=101):

        t_0 = np.zeros((P,1))
        x_0 = np.linspace(0, 1, P)[:, None]

        y = np.hstack([t_0, x_0])
        u = np.tile(u0, (P, 1))
        s = u0[:, None]

        return np.hstack([u, y, s])

    # Geneate bcs training data corresponding to one input sample
    def generate_one_bcs_training_data(u0, m=101, P=100):

        # t_bc = random.uniform(key, (P,1))
        t_bc = np.random.uniform(size=(P,1))
        x_bc1 = np.zeros((P, 1))
        x_bc2 = np.ones((P, 1))
    
        y1 = np.hstack([t_bc, x_bc1])  # shape = (P, 2)
        y2 = np.hstack([t_bc, x_bc2])  # shape = (P, 2)

        u = np.tile(u0, (P, 1))
        y =  np.hstack([y1, y2])  # shape = (P, 4)
        s = np.zeros((P, 1))

        return np.hstack([u, y, s])

    # Geneate res training data corresponding to one input sample
    def generate_one_res_training_data(u0, m=101, P=1000):

        # subkeys = random.split(key, 2)
    
        # t_res = random.uniform(subkeys[0], (P,1))
        # x_res = random.uniform(subkeys[1], (P,1))
        t_res = np.random.uniform(size=(P,1))
        x_res = np.random.uniform(size=(P,1))
        u = np.tile(u0, (P, 1))
        y =  np.hstack([t_res, x_res])
        s = np.zeros((P, 1))

        return np.hstack([u, y, s])

    # Geneate test data corresponding to one input sample
    def generate_one_test_data(idx,usol, m=101, P=101):

        u = usol[idx]
        u0 = u[0,:]

        t = np.linspace(0, 1, P)
        x = np.linspace(0, 1, P)
        T, X = np.meshgrid(t, x)

        s = u.T.flatten()[:, None]
        u = np.tile(u0, (P**2, 1))
        y = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])

        return np.hstack([u, y, s])

    # Prepare the training data

    # Load data
    path = 'data/Burger.mat'  # Please use the matlab script to generate data

    data = io.loadmat(path)
    usol = np.array(data['output'])

    N = usol.shape[0]  # number of total input samples, default 2000
    N_train =args.num_train      # number of input samples used for training, default 1000
    print(N, N_train)
    N_test = N - N_train  # number of input samples used for test
    m = args.m            # number of sensors for input samples, default 101
    P_ics_train = 101   # number of locations for evulating the initial condition
    P_bcs_train = 100    # number of locations for evulating the boundary condition
    P_res_train = 2500   # number of locations for evulating the PDE residual
    P_test = 101        # resolution of uniform grid for the test data

    u0_train = usol[:N_train,0,:]   # input samples
    N_x = u0_train.shape[-1]

    # Generate training data for inital condition
    # u_ics_train, y_ics_train, s_ics_train = vmap(generate_one_ics_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_ics_train)
    result = np.vstack([generate_one_ics_training_data(u0_train[i], m, P_ics_train) for i in range(u0_train.shape[0])])
    u_ics_train, y_ics_train, s_ics_train = result[:, :N_x], result[:, N_x:-1], result[:, -1:]

    # Generate training data for boundary condition
    # u_bcs_train, y_bcs_train, s_bcs_train = vmap(generate_one_bcs_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_bcs_train)
    result = np.vstack([generate_one_bcs_training_data(u0_train[i], m, P_bcs_train) for i in range(u0_train.shape[0])])
    u_bcs_train, y_bcs_train, s_bcs_train = result[:, :N_x], result[:, N_x:-1], result[:, -1:]

    # Generate training data for PDE residual
    # u_res_train, y_res_train, s_res_train = vmap(generate_one_res_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_res_train)
    result = np.vstack([generate_one_res_training_data(u0_train[i], m, P_res_train) for i in range(u0_train.shape[0])])
    u_res_train, y_res_train, s_res_train = result[:, :N_x], result[:, N_x:-1], result[:, -1:]

    result = np.vstack([generate_one_test_data(i, usol, m, P_test) for i in range(N_train, N)])
    u_test, y_test, s_test = result[:, :N_x], result[:, N_x:-1], result[:, -1:]
    # import pdb 
    # pdb.set_trace()
    np.savez_compressed(os.path.join('data',f"burger_train_ics_{N_train}.npz"), u_ics_train=u_ics_train, y_ics_train=y_ics_train, s_ics_train=s_ics_train)
    np.savez_compressed(os.path.join('data',f"burger_train_bcs_{N_train}.npz"), u_bcs_train=u_bcs_train, y_bcs_train=y_bcs_train, s_bcs_train=s_bcs_train)
    np.savez_compressed(os.path.join('data',f"burger_train_res_{N_train}.npz"), u_res_train=u_res_train, y_res_train=y_res_train, s_res_train=s_res_train)
    np.savez_compressed(os.path.join('data',f"burger_test_{N_test}.npz"), u_test=u_test, y_test=y_test, s_test=s_test)

def run(args, problem, system, space, T, m, nn, model, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    if args.save_data:
        load_data(args)
        d = np.load(f"data/burger_test_{args.num_test}.npz")
        u_test, y_test, s_test = d["u_test"], d["y_test"], d["s_test"]
        
        # debug 
        s_test = s_test.reshape(-1, 101*101)
        for i,img in enumerate(s_test):
            fig = plt.figure()
            ax0 = fig.add_subplot(111)
            # import pdb 
            # pdb.set_trace()
            h0 = ax0.imshow(img.reshape(101,101).T, interpolation='nearest', cmap='rainbow',
                            extent=[0,1,0,1],
                            origin='lower', aspect='auto')
            divider0 = make_axes_locatable(ax0)
            cax0 = divider0.append_axes("right", size="5%", pad=0.10)
            cbar0 = fig.colorbar(h0, cax=cax0)
            cbar0.ax.tick_params(labelsize=15)
            ax0.set_title('y_test')

            ax0.set_xlabel('x', fontweight='bold', size=15)
            ax0.set_ylabel('t', fontweight='bold', size=15)
            plt.savefig(os.path.join(args.name, f'y_test_{i}.png'))  
        return
   
    d = np.load(f"data/burger_train_ics_{args.num_train}.npz")
    u_train_ics, y_train_ics, s_train_ics= d["u_ics_train"], d["y_ics_train"], d["s_ics_train"]
    d = np.load(f"data/burger_train_bcs_{args.num_train}.npz")
    u_train_bcs, y_train_bcs, s_train_bcs= d["u_bcs_train"], d["y_bcs_train"], d["s_bcs_train"]
    d = np.load(f"data/burger_train_res_{args.num_train}.npz")
    u_train_res, y_train_res, s_train_res= d["u_res_train"], d["y_res_train"], d["s_res_train"]
    d = np.load(f"data/burger_test_{args.num_test}.npz")
    u_test, y_test, s_test = d["u_test"], d["y_test"], d["s_test"]

    # X_test_trim = trim_to_65535(X_test)[0]
    # y_test_trim = trim_to_65535(y_test)[0]
    
    train_data_res = DataGenerator(u_train_res, y_train_res, s_train_res, enlarge=1)
    train_data_bcs = DataGenerator(u_train_bcs, y_train_bcs, s_train_bcs, enlarge=25)
    train_data_ics = DataGenerator(u_train_ics, y_train_ics, s_train_ics, enlarge=25)
    test_data = DataGenerator(u_test, y_test, s_test)
    train_loader_res = data.DataLoader(train_data_res, batch_size=50000, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    train_loader_bcs = data.DataLoader(train_data_bcs, batch_size=50000, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    train_loader_ics = data.DataLoader(train_data_ics, batch_size=50000, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    test_loader = data.DataLoader(test_data, batch_size=50000, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    # import pdb 
    # pdb.set_trace() 
    optimizer = torch.optim.Adam(model.net.parameters(), 
                                lr, betas=(0.9, 0.999), eps=1e-8)
    schduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/2000))
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    writer = SummaryWriter(log_dir=args.name) 
    step = 0
    for i in range(epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        losses_ics = AverageMeter('Loss_ics', ':.4e')
        losses_bcs = AverageMeter('Loss_bcs', ':.4e')
        losses_res = AverageMeter('Loss_res', ':.4e')
        progress = ProgressMeter(
            len(train_loader_res),
            [batch_time, data_time, losses, losses_ics, losses_bcs, losses_res],
            prefix="[{}]".format(i))
        train_loader_bcs_iterator = iter(train_loader_bcs)
        train_loader_ics_iterator = iter(train_loader_ics)
        model.net.train()
     
        end = time.time()
        for j, (u_res, y_res, s_res) in enumerate(train_loader_res):
            # measure data loading time
            try:
                u_bcs, y_bcs, s_bcs = next(train_loader_bcs_iterator)
            except StopIteration:
                train_loader_bcs_iterator = iter(train_loader_bcs)
                u_bcs, y_bcs, s_bcs = next(train_loader_bcs_iterator)
            
            try:
                u_ics, y_ics, s_ics = next(train_loader_ics_iterator)
            except StopIteration:
                train_loader_ics_iterator = iter(train_loader_ics)
                u_ics, y_ics, s_ics = next(train_loader_ics_iterator)

            u_bcs = u_bcs.cuda(non_blocking=True)
            y_bcs = y_bcs.cuda(non_blocking=True)
            # s_bcs = s_bcs.cuda(non_blocking=True)
            
            u_ics = u_ics.cuda(non_blocking=True)
            y_ics = y_ics.cuda(non_blocking=True)
            s_ics = s_ics.cuda(non_blocking=True)
            
            u_res = u_res.cuda(non_blocking=True)
            y_res = y_res.cuda(non_blocking=True)
            # s_res = s_res.cuda(non_blocking=True)

            t_res = y_res[:, 0:1].requires_grad_()
            x_res = y_res[:, 1:2].requires_grad_()

            t_bcs_1 = y_bcs[:, 0:1]
            x_bcs_1 = y_bcs[:, 1:2].requires_grad_()
            t_bcs_2 = y_bcs[:, 2:3]
            x_bcs_2 = y_bcs[:, 3:4].requires_grad_()
            data_time.update(time.time() - end)
            # print('data', data_time)
            # compute output
            loss_ics = model.loss_ics(u_ics, y_ics, s_ics)
            loss_bcs = model.loss_bcs(u_bcs, t_bcs_1, x_bcs_1, t_bcs_2, x_bcs_2)
            loss_res = model.loss_res(u_res, t_res, x_res)
            loss = 20 * loss_ics + loss_bcs + loss_res 
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print('batch', data_time)
            losses.update(loss, 50000)
            losses_ics.update(loss_ics, 50000)
            losses_bcs.update(loss_bcs, 50000)
            losses_res.update(loss_res, 50000)
            
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            if (step+1) % 50 == 0:
                # logging.info(f'{[step+1]} data: {batch_time:.3f}({data_time:.3f}) loss: {loss:.4e}, loss_ics: {loss_ics:.4e}, loss_bcs: {loss_bcs:.4e}, loss_res: {loss_res:.4e}')
                logging.info(progress.display(j))
                
                writer.add_scalars('loss', {'train_loss_ics': losses_ics.avg.detach()}, step)
                writer.add_scalars('loss', {'train_loss_bcs': losses_bcs.avg.detach()}, step)
                writer.add_scalars('loss', {'train_loss_res': losses_res.avg.detach()}, step)
                writer.add_scalars('loss', {'train_loss': losses.avg.detach()}, step)
            
            if (step+1) % 500 == 0:

                save_checkpoint({
                    'epoch': i + 1,
                    'state_dict': model.net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : schduler.state_dict(), 
                }, args.name, filename=f'checkpoint_{step+1}.pth.tar')

                model.net.eval()
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
                        output = model.net(u,y)
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

                    target = torch.cat(target, dim=0).reshape(-1, 101*101)
                    pred = torch.cat(pred, dim=0).reshape(-1, 101*101)
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

    branch_layer = [m, 100, 100, 100, 100, 100, 100, 100]
    trunk_layer =  [2, 100, 100, 100, 100, 100, 100, 100]
    
    net = PIDeepONet(trunk_layer, branch_layer, activation=args.activation, init=True)#.cuda()
    
    logging.info(net.net)
    
    run(args, problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)


if __name__ == "__main__":
    main()
