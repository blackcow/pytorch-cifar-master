'''Train CIFAR10 with PyTorch
测试各 label 的 benign acc，不受限于各 label 的 data 数量
测试各 epoch=100 的 model
绘制成图
计算各 label 的特征中心，然后 label 特征中心的距离
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse

from models import *
# from utils import progress_bar
# from network import create_network

import cifar10my2
import cifar10my3
import models.preactresnet_cl
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
import matplotlib.pyplot as plt
from models.wideresnet import WideResNet
from models.densenet import DenseNet121
from models.preactresnet import create_network
from torch.autograd import Variable
from time import time
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary
from dataset.imagnette import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default='0', type=str, help='GPUs id')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# Model facotrs
parser.add_argument('--depth', type=int, default=34, metavar='N',
                    help='model depth (default: 34)')
parser.add_argument('--widen_factor', type=int, default=10, metavar='N',
                    help='model widen_factor (default: 10)')
parser.add_argument('--droprate', type=float, default=0.0, metavar='N',
                    help='model droprate (default: 0.0)')
# draw imgs
parser.add_argument('--factors', default='model', type=str, metavar='N',
                    choices=['widen_factor', 'depth', 'droprate', 'epsilon', 'model'],
                    help='tensorboard draw img factors')

# PGD attack
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--num-steps', default=20, help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, help='perturb step size')
parser.add_argument('--random', default=True, help='random initialization for PGD')
# test on dataset
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'STL10', 'Imagnette', 'SVHN'],
                    help='train model on dataset')
args = parser.parse_args()
print(args)

# 设定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# a = torch.Tensor([1,2,3])
# b = torch.Tensor([4,5])

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # 对于 TRADES 提供的 model 注释掉
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
bs = 100
if args.dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    # testset = cifar10my3.CIFAR10MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
elif args.dataset == 'CIFAR100':
    testset = cifar10my3.CIFAR100MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
elif args.dataset == 'Imagnette':
    testset = ImagenetteTrain('val')
    # testset = ImagenetteTest()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
elif args.dataset == 'SVHN':
    testset = torchvision.datasets.SVHN(root='../data', split="test", download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

cudnn.benchmark = True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def loadmodel(i, factor):
    # Model
    # ckpt_list = ['model-wideres-epoch75.pt', 'model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']
    ckpt_list = ['model-wideres-epoch76.pt']
    print('==> Building model..')

    ckpt = '/data/niuzh/model/cifar10_rst_adv.pt.ckpt'
    checkpoint = torch.load(ckpt)
    net = nn.DataParallel(WideResNet(depth=factor[1], widen_factor=factor[2], dropRate=factor[3])).cuda()
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint)
    net.eval()
    print(ckpt)
    return net


def loadmodel_preactresnte(i, factor):
    print('==> Building model..')
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST/rmlabel_0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_cl/'
    # ckpt_list = ['percent_0.2', 'percent_0.5', 'percent_0.7', 'percent_0.9']
    # ckpt_list = ['ckpt-epoch76.pt', 'ckpt-epoch100.pt']

    # ST,CIFAR-10
    # ckpt = './'
    # ckpt_list = ['model-wideres-epoch100.pt', 'model-wideres-epoch100.pt']
    # ST-keeplabel, CIFAR 10
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST/kplabel/percent_0.1/'
    # CIFAR 100
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_CIFAR100/kplabel/percent_0.5/'
    # imagnette
    ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_Imagnette/kplabel_seed1/percent_1.0/'
    # SVHN
    ckpt_list = ['model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'Imagnette':
        num_classes = 10
    net = nn.DataParallel(create_network(num_classes)).cuda()
    ckpt += ckpt_list[i]
    checkpoint = torch.load(ckpt)
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(checkpoint)
    net.eval()
    print(ckpt)
    return net


def loadmodel_preactresnte_cl(i, factor):
    print('==> Building model..')
    ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_cl/'
    ckpt_list = ['ckpt-epoch76.pt', 'ckpt-epoch100.pt']
    net = nn.DataParallel(models.preactresnet_cl.create_network()).cuda()
    ckpt += ckpt_list[i]
    # ckpt += '/model-wideres-epoch100.pt'
    checkpoint = torch.load(ckpt)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    print(ckpt)
    return net


# acc test
def _pgd_whitebox(model, X, y, epsilon, num_steps=args.num_steps, step_size=args.step_size):
    rep, out = model(X)
    N, C, H, W = rep.size()
    rep = rep.reshape([N, -1])
    out = out.data.max(1)[1]
    # err = (out.data.max(1)[1] != y.data).float().sum()
    return out, y


# input: tensorboard, model, model_name
def test(writer, net, model_name, epsilon):
    global best_acc
    global best_epoch

    acc_natural_label = []
    target = []
    output = []
    with torch.no_grad():
        # for inputs, targets in testloader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            out, y = _pgd_whitebox(net, X, y, epsilon=epsilon)
            output.append(out)
            target.append(y)

        # 计算每个类别下的 err
        output1 = torch.stack(output[:-1])
        target1 = torch.stack(target[:-1])
        # 最后一行可能不满一列的长度，单独 concat
        output1 = torch.cat((output1.reshape(-1), output[-1]), dim=0).cpu().numpy()
        target1 = torch.cat((target1.reshape(-1), target[-1]), dim=0).cpu().numpy()

        # 获取每个 label 的 out 和 target
        for m in np.unique(target1):
            idx = [i for i, x in enumerate(target1) if x == m]
            output = output1[idx]
            target = target1[idx]
            acc = (output == target).sum() / target.size * 100
            acc_natural_label.append(acc)
            # print(m)

    # 输出各 label 下的 acc
    print('acc_natural_label：')
    for i in acc_natural_label:
        print('{:.1f}'.format(i))
        # print("%d" % i)

    return None


def main():
    start = time()
    seed_everything(1)
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # load model
    # 根据测试的 factor 选择对应的 model
    print('factors:', args.factors)
    logits = [0, 0, 0, 0]
    # logits_robust = [0, 0, 0]
    model_num = 2
    if args.factors == 'model':
        for i in range(model_num):
            print("Test: " + str(i))
            factor = [args.epsilon, args.depth, args.widen_factor, args.droprate]
            # net = loadmodel(i, factor)
            net = loadmodel_preactresnte(i, factor)
            # net = loadmodel_preactresnte_cl(i, factor)
            test(writer, net, 'model_name', factor[0])
    else:
        raise Exception('this should never happen')
    # sum of the dis of the center rep
    # for m in range(model_num):
    #     print('%.2f' % logits[m])
    # for m in range(model_num):
    #     print('%.2f' % logits_robust[m])

    writer.close()
    end = time()
    print('时间:{:3f}'.format((end - start) / 60))


if __name__ == '__main__':
    main()
