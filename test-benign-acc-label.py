'''Train CIFAR10 with PyTorch
测试各 label 的 benign acc，测试各 epoch=100 的 model
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

bs = 100
testset = cifar10my3.CIFAR10MY(
    root='../data', train=False, download=True, transform=transform_test, args=args)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs, shuffle=False, num_workers=2)

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
    ckpt = '../Fair-AT/model-cifar-wideResNet/preactresnet/ST_cl/'
    # ckpt_list = ['percent_0.2', 'percent_0.5', 'percent_0.7', 'percent_0.9']
    ckpt_list = ['ckpt-epoch76.pt', 'ckpt-epoch100.pt']
    net = nn.DataParallel(create_network()).cuda()
    ckpt += ckpt_list[i]
    # ckpt += '/model-wideres-epoch100.pt'
    checkpoint = torch.load(ckpt)
    net.load_state_dict(checkpoint)
    net.eval()
    print(ckpt)
    return net

# PGD Attack
def _pgd_whitebox(model, X, y, epsilon, num_steps=args.num_steps, step_size=args.step_size):
    rep, out = model(X)
    N, C, H, W = rep.size()
    rep = rep.reshape([N, -1])
    err = (out.data.max(1)[1] != y.data).float().sum()
    # X_pgd = Variable(X.data, requires_grad=True)
    # if args.random:
    #     random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    #     X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    #
    # for _ in range(num_steps):
    #     opt = optim.SGD([X_pgd], lr=1e-3)
    #     opt.zero_grad()
    #     with torch.enable_grad():
    #         loss = nn.CrossEntropyLoss()(model(X_pgd)[1], y)
    #     loss.backward()
    #     eta = step_size * X_pgd.grad.data.sign()
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    #     eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #     X_pgd = Variable(X.data + eta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # rep_pgd, out_pgd = model(X_pgd)
    # err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()

    # rep_pgd = rep_pgd.reshape([N, -1])
    return err, None, rep, None
    # return err, err, rep


# input: tensorboard, model, model_name
def test(writer, net, model_name, epsilon):
    global best_acc
    global best_epoch

    accs_batch = []
    # acc_robust_label = []
    acc_natural_label = []
    count = 0
    # robust_err_total_label = 0
    natural_err_total_label = 0
    tmprep, _ = net(torch.zeros([20, 3, 32, 32]).cuda())
    _, C, H, W = tmprep.size()
    # center of the rep
    rep_label = torch.zeros([10, C * H * W]).cuda()
    # rep_robust_label = torch.zeros([10, C * H * W]).cuda()
    rep_all = torch.zeros([C * H * W]).cuda()
    # rep_pgd_all = torch.zeros([C * H * W]).cuda()
    i = 0
    with torch.no_grad():
        # for inputs, targets in testloader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            X, y = Variable(inputs, requires_grad=True), Variable(targets)
            err_natural, _, rep, _ = _pgd_whitebox(net, X, y, epsilon=epsilon)
            # robust_err_total_label += err_robust
            natural_err_total_label += err_natural
            # 累加 rep
            rep_all = rep_all + rep.sum(dim=0)
            # rep_pgd_all = rep_pgd_all + rep_pgd.sum(dim=0)
            count = bs + count
            # 计算每个类别下的 err
            if count % 1000 == 0:
                rep_label[i] = rep_all/1000 # 计算 rep 中心
                # rep_robust_label[i] = rep_pgd_all/1000
                # 清空
                rep_all = torch.zeros([C * H * W]).cuda()
                # rep_pgd_all = torch.zeros([C * H * W]).cuda()
                i += 1
                # robust_acc = (1 - robust_err_total_label / 1000).cpu().numpy()
                natural_acc = (1 - natural_err_total_label / 1000).cpu().numpy()
                # acc_robust_label.append(robust_acc)
                acc_natural_label.append(natural_acc)
                # robust_err_total_label = 0
                natural_err_total_label = 0

    # 输出各 label 下的 acc
    print('acc_natural_label：')
    for i in acc_natural_label:
        print('{:3f}'.format(i))

    # print('acc_robust_label：')
    # for i in acc_robust_label:
    #     print('{:3f}'.format(i))

    # 各 label 的 Rep 中心归一化，计算余弦相似度
    rep_norm = nn.functional.normalize(rep_label, dim=1)
    logits = torch.mm(rep_norm, torch.transpose(rep_norm, 0, 1))  # [10,HW]*[HW,10]=[10,10]
    logits = logits - torch.diag_embed(torch.diag(logits))  # 去掉对角线的 1
    # logits = logits.abs().sum().cpu().numpy()
    # 只统计 cos sim 大于 0 的
    zero = torch.zeros_like(logits)
    logits1 = torch.where(logits < 0, zero, logits)
    logits1 = logits1.sum().cpu().numpy()
    print('Sum distance of each label rep: {:.2f}'.format(logits1))

    # rep_robust = nn.functional.normalize(rep_robust_label, dim=1)
    # logits_robust = torch.mm(rep_robust, torch.transpose(rep_robust, 0, 1))  # [10,HW]*[HW,10]=[10,10]
    # logits_robust = logits_robust - torch.diag_embed(torch.diag(logits_robust))  # 去掉对角线的 1
    # # logits_robust = logits_robust.abs().sum().cpu().numpy()
    # # 值统计大于 0 的
    # zero = torch.zeros_like(logits_robust)
    # logits2 = torch.where(logits_robust < 0, zero, logits_robust)
    # logits2 = logits2.sum().cpu().numpy()
    # print('Sum distance of robust label rep: {:.2f}'.format(logits2))

    return logits1


def main():
    start = time()
    seed_everything(1)
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # load model
    # 根据测试的 factor 选择对应的 model
    print('factors:', args.factors)
    logits = [0, 0, 0, 0]
    # logits_robust = [0, 0, 0]
    model_num = 4
    if args.factors == 'model':
        for i in range(model_num):
            print("Test: " + str(i))
            factor = [args.epsilon, args.depth, args.widen_factor, args.droprate]
            # net = loadmodel(i, factor)
            net = loadmodel_preactresnte(i, factor)
            # test robust fair model
            # net = loadmodel_robustfair(i, factor)
            logits[i] = test(writer, net, 'model_name', factor[0])
    else:
        raise Exception('this should never happen')
    # sum of the dis of the center rep
    for m in range(model_num):
        print('%.2f' % logits[m])
    # for m in range(model_num):
    #     print('%.2f' % logits_robust[m])

    writer.close()
    end = time()
    print('时间:{:3f}'.format((end - start) / 60))


if __name__ == '__main__':
    main()