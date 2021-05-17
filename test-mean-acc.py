'''Train CIFAR10 with PyTorch
测试 benign acc 和 robust acc（mean acc 随 epoch 的变化情况）
绘制成图
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import cifar10my2
import cifar10my3
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
import matplotlib.pyplot as plt
from models.wideresnet import WideResNet
from torch.autograd import Variable
from time import time
from torch.utils.tensorboard import SummaryWriter
import numpy


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default='0,1,2', type=str, help='GPUs id')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# Model facotrs
parser.add_argument('--depth', type=int, default=34, metavar='N',
                    help='model depth (default: 34)')
parser.add_argument('--widen_factor', type=int, default=10, metavar='N',
                    help='model widen_factor (default: 10)')
parser.add_argument('--droprate', type=float, default=0.0, metavar='N',
                    help='model droprate (default: 0.0)')
# draw imgs
parser.add_argument('--factors', type=str, default='widen_factor', metavar='N',
                    help='tensorboard draw img factors')

# PGD attack
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--num-steps', default=20, help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, help='perturb step size')
parser.add_argument('--random', default=True, help='random initialization for PGD')
args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # 对于 TRADES 提供的 model 注释掉
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# bs = 20
bs = 1000
testset = cifar10my3.CIFAR10MY(
    root='./data', train=False, download=True, transform=transform_test, args=args)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs, shuffle=False, num_workers=2)

# set up data loader
# kwargs = {'num_workers': 1, 'pin_memory': True}
# transform_test = transforms.Compose([transforms.ToTensor(),])
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, **kwargs)
cudnn.benchmark = True

def loadmodel(model_name, i):
    # Model
    print('==> Building model..')
    # ckpt = '/hot-data/niuzh/Mycode/pytorch-cifar-master/checkpoint/model_cifar_wrn.pt'

    # ckpt = '/hot-data/niuzh/Mycode/TRADES-master/model-cifar-wideResNet/ST/model-wideres-epoch87.pt'
    ckpt = '/hot-data/niuzh/Mycode/TRADES-master/model-cifar-wideResNet/AT/' + model_name +'/'
    ckpt += 'model-wideres-epoch' + str(i)+'.pt'
    # net = WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.droprate).cuda()
    net = nn.DataParallel(WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.droprate)).cuda()
    # net.load_state_dict(torch.load(path + ckpt))
    net.load_state_dict(torch.load(ckpt))
    net.eval()
    print(ckpt)
    return net

# PGD Attack
def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
# def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    _, out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd)[1], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd)[1].data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

# input: tensorboard, model, model_name, ith epoch
def test(writer, net, model_name, i):
    global best_acc
    global best_epoch

    accs_batch = []
    acc_robust_label = []
    acc_natural_label = []
    count = 0
    robust_err_total_label = 0
    natural_err_total_label = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            err_natural, err_robust = _pgd_whitebox(net, X, y)
            robust_err_total_label += err_robust
            natural_err_total_label += err_natural

            count = bs + count
            # 计算每个类别下的 err
            if count % 1000 == 0:
                label_index = count/1000-1
                robust_acc = (1-robust_err_total_label/1000).cpu().numpy()
                natural_acc = (1-natural_err_total_label/1000).cpu().numpy()
                # print('robust_acc: {:3f}'.format(robust_acc))
                # print('natural_acc: {:3f}'.format(natural_acc))
                acc_robust_label.append(robust_acc)
                acc_natural_label.append(natural_acc)
                robust_err_total_label = 0
                natural_err_total_label = 0

    # 对 mean acc 绘图
    robust_acc_mean = numpy.mean(acc_robust_label)
    natural_acc_mean = numpy.mean(acc_natural_label)
    graph_name = 'test/' + model_name + '/robust_acc_mean'
    writer.add_scalars(graph_name, {'robust_acc': robust_acc_mean}, i)
    graph_name = 'test/' + model_name + '/benign_acc_mean'
    writer.add_scalars(graph_name, {'natural_acc': natural_acc_mean}, i)

    # 输出各 label 下的 acc
    # print('acc_natural_label：')
    # for i in acc_natural_label:
    #     print('{:3f}'.format(i))
    #
    # print('acc_robust_label：')
    # for i in acc_robust_label:
    #     print('{:3f}'.format(i))
    # return 0

def main():
    start = time()
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # load model
    # model_name = 'e' + str(args.epsilon) + '_depth' + str(args.depth) + '_' + 'widen' + str(
    #     args.widen_factor) + '_' + 'drop' + str(args.droprate)
    # if args.factors == 'widen_factor':
    if True:
        # widen_factors = [4, 6, 8, 10, 12]
        for i in range(100):
            t0 = time()

            model_name = 'e' + str(args.epsilon) + '_depth' + str(args.depth) + '_' + \
                         'widen' + str(args.widen_factor) + '_' + 'drop' + str(args.droprate)
            print("Test " + model_name)
            # load epoch ith model
            net = loadmodel(model_name, i+1)
            # test robust acc & benign acc
            test(writer, net, model_name, i+1)
            t1 = time()
            print('时间:{:3f}'.format((t1 - t0) / 60))

    writer.close()
    end = time()
    print('时间:{:3f}'.format((end-start)/60))

if __name__ == '__main__':
    main()