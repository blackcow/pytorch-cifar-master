'''Train CIFAR10 with PyTorch
调整 factor，测试各 epoch=76 的 model
测试 benign acc 和 robust acc（在各个 label 下）,不受限于各 label 的 data 数量
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default='0', type=str, help='GPUs id')
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
bs = 100
if args.dataset == 'CIFAR10':
    testset = cifar10my3.CIFAR10MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
elif args.dataset == 'CIFAR100':
    testset = cifar10my3.CIFAR100MY(root='../data', train=False, download=True, transform=transform_test, args=args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

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
    # path = '../Fair-AT/model-cifar-wideResNet/wideresnet/'
    # ckpt = '/hot-data/niuzh/Mycode/pytorch-cifar-master/checkpoint/model_cifar_wrn.pt'
    # ST
    # ckpt = '/hot-data/niuzh/Mycode/Fair-AT/model-cifar-wideResNet/wideresnet/ST' \
    #        '/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '/hot-data/niuzh/Mycode/Fair-AT/model-cifar-wideResNet/wideresnet' \
    # '/ST-ori/e0.031_depth34_widen10_drop0.0/'

    # Fair ST
    # ckpt = '/hot-data/niuzh/Mycode/Fair-AT/model-cifar-wideResNet/wideresnet/' \
    #        'ST_fair_v1/e0.031_depth34_widen10_drop0.0/'

    # TRADES AT
    # ckpt = path + 'TRADES/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/wideresnet/TRADES/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/e0.031_depth34_widen10_drop0.0'
    # ckpt += 'model-wideres-epoch76.pt'

    # ckpt = path + 'ST_fair_v1a_T0.1_L1/e0.031_depth34_widen10_drop0.0/'
    # ckpt = path + 'TRADES_fair_v1a_T0.1_L1/e0.031_depth34_widen10_drop0.0/'

    # ICML
    # ckpt_list = ['trade_10_1.0.pt', 'trade_60_1.0.pt', 'trade_120_1.0.pt']
    # ckpt = '../Robust-Fair/cifar10/models-wideresnet/fair1/'
    # Fair AT
    # ckpt = '../Fair-AT/model-cifar-wideResNet/wideresnet/TRADES/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/e0.031_depth34_widen10_drop0.0/'
    # ckpt += 'model-wideres-epoch76.pt'

    # ckpt += ckpt_list[i]

    ckpt = '/data/niuzh/model/cifar10_rst_adv.pt.ckpt'
    checkpoint = torch.load(ckpt)
    net = nn.DataParallel(WideResNet(depth=factor[1], widen_factor=factor[2], dropRate=factor[3])).cuda()
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint)
    net.eval()
    print(ckpt)
    return net


def loadmodel_preactresnte(i, factor):
    # Model
    # ckpt_list = ['model-wideres-epoch10.pt', 'model-wideres-epoch11.pt', 'model-wideres-epoch12.pt']
    print('==> Building model..')
    # AT preactresnet
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/e0.031_depth34_widen10_drop0.0/model-wideres-epoch76.pt'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_fair_v1a_T0.1_L1/e0.031_depth34_widen10_drop0.0/'
    # ICML-21
    # ckpt_list = ['trade_10_1.0.pt', 'trade_60_1.0.pt', 'trade_120_1.0.pt']
    # ckpt_list = ['trade_120_1.0.pt']
    # ckpt = '../Robust-Fair/cifar10/models-preactresnet/fair1/'
    # net = create_network().cuda()
    # Fair-AT
    # ckpt_list = ['model-wideres-epoch75.pt', 'model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_fair_v1a_T0.1_L1/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_fair_v1a_T0.1_L1-fl1/e0.031_depth34_widen10_drop0.0/'
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/e0.031_depth34_widen10_drop0.0/'
    # ckpt_list = ['model-wideres-epoch75.pt', 'model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    # AT with OPT save
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/AT-opt/'
    # ckpt_list = ['ckpt-epoch75.pt', 'ckpt-epoch76.pt', 'ckpt-epoch100.pt']

    # rm label AT
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/rmlabel_seed2/rmlabel' + str(label) + '/'
    # ckpt_list = ['model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    # Fine-Tune model
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/fine-tune/'
    # ckpt_list = ['ckpt-ft-epoch76.pt', 'ckpt-ft-epoch100.pt', 'ckpt-ft-epoch120.pt']

    # FC Fine-Tune model
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/fine-tune-FC/resum_100/'
    # ckpt_list = ['ckpt-ft-epoch100.pt', 'ckpt-ft-epoch120.pt', 'ckpt-ft-epoch140.pt']

    # 只在某 label 上，做 AT
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES/svlabel_seed1/svlabel_35/'
    # ckpt_list = ['model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    # CIFAR 100, TRADES
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/TRADES_CIFAR100/'
    # imagnette
    # ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_Imagnette/kplabel_seed1/percent_1.0/'
    # SVHN
    ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_SVHN/kplabel_seed1/percent_0.1/'
    ckpt_list = ['model-wideres-epoch100.pt', 'model-wideres-epoch100.pt']
    # ckpt_list = ['model-wideres-epoch76.pt', 'model-wideres-epoch100.pt']

    if args.dataset == 'CIFAR10' or 'STL10' or 'Imagnette' or 'SVHN':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    net = nn.DataParallel(create_network(num_classes)).cuda()
    ckpt += ckpt_list[i]
    # print(net)
    net.load_state_dict(torch.load(ckpt))

    # for AT-opt & Fine-tune model
    # checkpoint = torch.load(ckpt)
    # net.load_state_dict(checkpoint['net'])
    net.eval()
    print(ckpt)
    return net

# PGD Attack
def _pgd_whitebox(model, X, y, epsilon, num_steps=args.num_steps, step_size=args.step_size):
    rep, out = model(X)
    N, C, H, W = rep.size()
    rep = rep.reshape([N, -1])
    out = out.data.max(1)[1]
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
    rep_pgd, out_pgd = model(X_pgd)
    out_pgd = out_pgd.data.max(1)[1]

    rep_pgd = rep_pgd.reshape([N, -1])
    return out, rep, out_pgd, rep_pgd


# input: tensorboard, model, model_name
def test(writer, net, model_name, epsilon):
    global best_acc
    global best_epoch

    acc_natural_label = []
    acc_robust_label = []
    target = []
    output = []
    output_robust = []
    with torch.no_grad():
        # for inputs, targets in testloader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            out, rep, out_pgd, rep_pgd = _pgd_whitebox(net, X, y, epsilon=epsilon)
            output.append(out)
            output_robust.append(out_pgd)
            target.append(y)

        # 计算每个类别下的 err
        output1 = torch.stack(output[:-1])
        output1_pgd = torch.stack(output_robust[:-1])
        target1 = torch.stack(target[:-1])
        # 最后一行可能不满一列的长度，单独 concat
        output1 = torch.cat((output1.reshape(-1), output[-1]), dim=0).cpu().numpy()
        output1_pgd = torch.cat((output1_pgd.reshape(-1), output_robust[-1]), dim=0).cpu().numpy()
        target1 = torch.cat((target1.reshape(-1), target[-1]), dim=0).cpu().numpy()

        # 获取每个 label 的 out 和 target
        for m in np.unique(target1):
            idx = [i for i, x in enumerate(target1) if x == m]
            output = output1[idx]
            output_pgd = output1_pgd[idx]
            target = target1[idx]
            acc = (output == target).sum() / target.size * 100
            acc_robust = (output_pgd == target).sum() / target.size * 100
            acc_natural_label.append(acc)
            acc_robust_label.append(acc_robust)
            # print(m)

    # 输出各 label 下的 acc
    print('acc_natural_label：')
    for i in acc_natural_label:
        print('{:.1f}'.format(i))
        # print("%d" % i)

    print('acc_robust_label：')
    for i in acc_robust_label:
        print('{:3f}'.format(i))

    return None


def main():
    start = time()
    seed_everything(1)
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # load model
    # 根据测试的 factor 选择对应的 model
    print('factors:', args.factors)
    logits = [0, 0, 0]
    logits_robust = [0, 0, 0]
    model_num = 2
    if args.factors == 'model':
        for i in range(model_num):
            print("Test: " + str(i))
            factor = [args.epsilon, args.depth, args.widen_factor, args.droprate]
            # net = loadmodel(i, factor)
            net = loadmodel_preactresnte(i, factor)
            # test robust fair model
            # net = loadmodel_robustfair(i, factor)
            logits[i], logits_robust[i] = test(writer, net, 'model_name', factor[0])
    else:
        raise Exception('this should never happen')
    # sum of the dis of the center rep
    for m in range(model_num):
        print('%.2f' % logits[m])
    for m in range(model_num):
        print('%.2f' % logits_robust[m])

    writer.close()
    end = time()
    print('时间:{:3f}'.format((end - start) / 60))


if __name__ == '__main__':
    main()
